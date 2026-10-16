#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace utils
{
  struct CSRMatrix
  {
    using Index = long long;
    using ColIndex = uint32_t;

    // Number of rows. For a square matrix ncols_ == nrows_; for a rectangular
    // (two-basis) matrix nrows_ is the bra dimension and ncols_ the ket dimension.
    Index nrows_;
    Index ncols_;

    struct RowData 
    {
      std::vector<ColIndex> cols;  // 4 bytes/entry
      std::vector<double>   vals;  // 8 bytes/entry
    };                               // = 12 bytes/entry, no padding
    std::vector<RowData> rows;

    // final CSR storage
    std::vector<Index> row_ptr;
    std::vector<ColIndex> col_idx;
    std::vector<double> values;

    bool finalized = false;

    // When true, only the lower triangle (including the diagonal) is stored;
    // the transpose entries are reconstructed on the fly during the
    // matrix-vector product. Set automatically by add_symmetric().
    bool symmetric_ = false;

    // Constructor (square: ncols == dim)
    CSRMatrix(Index dim) : nrows_(dim), ncols_(dim), rows(dim) { }
    // Constructor (rectangular: nrows x ncols, e.g. bra_dim x ket_dim)
    CSRMatrix(Index nrows, Index ncols) : nrows_(nrows), ncols_(ncols), rows(nrows) { }

    // Getters
    const std::vector<Index>& get_row_ptr() const { return row_ptr; }
    const std::vector<ColIndex>& get_col_idx() const { return col_idx; }
    const std::vector<double>& get_values() const { return values; }
    int dim() const { return static_cast<int>(nrows_); }
    Index nrows() const { return nrows_; }
    Index ncols() const { return ncols_; }
    bool is_symmetric() const { return symmetric_; }
    size_t size() const
    {
      assert(finalized);
      return values.size();
    }

    // Add entry
    void add(Index i, Index j, double v)
    {
      rows[i].cols.push_back(static_cast<ColIndex>(j));
      rows[i].vals.push_back(v);
    }

    // Add symmetric entry. Only the lower triangle (row = max, col = min) is
    // stored; the transpose is applied in the symmetric matrix-vector product
    // (symmetric_spmv_lower), halving storage and finalize work.
    void add_symmetric(Index i, Index j, double v)
    {
      symmetric_ = true;
      const Index r = (i >= j) ? i : j;
      const Index c = (i >= j) ? j : i;
      rows[r].cols.push_back(static_cast<ColIndex>(c));
      rows[r].vals.push_back(v);
    }

    // Finalize: sort + merge + CSR build
    void finalize()
    {
      if (finalized) return;

      size_t nnz = 0;

      // (I) sort + merge each row
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:nnz)
#endif
      for (Index i = 0; i < nrows_; i++)
      {
        auto& row = rows[i];
        if (row.cols.empty()) continue;
        auto n = row.cols.size();

        // zip into pairs for sorting (temporary, freed at end of scope)
        std::vector<std::pair<ColIndex, double>> pairs(n);
        for (size_t k = 0; k < n; k++)
          pairs[k] = {row.cols[k], row.vals[k]};

        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // merge duplicates, write back into SoA
        size_t write = 0;
        for (size_t k = 0; k < n;)
        {
          ColIndex col = pairs[k].first;
          double val   = pairs[k].second;

          size_t l = k + 1;
          while (l < n && pairs[l].first == col)
          {
            val += pairs[l].second;
            l++;
          }

          if (std::fabs(val) > 1e-10)
          {
            row.cols[write] = col;
            row.vals[write] = val;
            write++;
          }
          k = l;
        }

        row.cols.resize(write);
        row.cols.shrink_to_fit();
        row.vals.resize(write);
        row.vals.shrink_to_fit();
        nnz += write;
      }

      // (II) build row_ptr from row sizes, then allocate CSR
      row_ptr.resize(nrows_ + 1);
      row_ptr[0] = 0;
      for (Index i = 0; i < nrows_; i++)
        row_ptr[i + 1] = row_ptr[i] + static_cast<Index>(rows[i].cols.size());

      col_idx.resize(nnz);
      values.resize(nnz);

      // (III) fill CSR in parallel (row_ptr already known)
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (Index i = 0; i < nrows_; i++)
      {
        const auto& row = rows[i];
        Index pos = row_ptr[i];
        for (size_t k = 0; k < row.cols.size(); k++)
        {
          col_idx[pos + k] = row.cols[k];
          values[pos + k]  = row.vals[k];
        }
        { std::vector<ColIndex>().swap(rows[i].cols);  // free immediately
          std::vector<double>().swap(rows[i].vals); }
      }
      assert(row_ptr[nrows_] == static_cast<Index>(nnz));

      // (IV) free temporary storage
      rows.clear();
      rows.shrink_to_fit();

      finalized = true;
    }

    // Clear
    void clear()
    {
      row_ptr.clear();
      row_ptr.shrink_to_fit();
      col_idx.clear();
      col_idx.shrink_to_fit();
      values.clear();
      values.shrink_to_fit();
    }

    // y = A*x for a general (possibly rectangular, non-symmetric) matrix.
    // x has ncols_ entries, y has nrows_ (= nrows) entries. Purely local: when
    // the matrix is MPI-block-distributed, sum y (or the downstream scalar)
    // across ranks separately.
    void spmv(const double* x, double* y) const
    {
      assert(finalized);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(runtime)
#endif
      for (Index i = 0; i < nrows_; i++)
      {
        double sum = 0.0;
        const Index kb = row_ptr[i], ke = row_ptr[i + 1];
        for (Index k = kb; k < ke; k++)
          sum += values[k] * x[col_idx[k]];
        y[i] = sum;
      }
    }

    // left^T * A * right, in a single pass with no temporary vector. left has
    // nrows_ entries, right has ncols_. Local only: for an MPI-block-distributed
    // matrix each rank returns its partial scalar; MPI_Allreduce(sum) them to
    // get the full value.
    //
    // When symmetric_ is set only the lower triangle (col <= row) is stored, so
    // each off-diagonal entry A(i,j) contributes both orientations to the form:
    // left[i] A(i,j) right[j] + left[j] A(i,j) right[i]. All accesses to left[]
    // and right[] are reads, so the transpose term is race-free under the
    // reduction (unlike the symmetric SpMV, which writes y[j]).
    double quadratic_form(const double* left, const double* right) const
    {
      assert(finalized);
      double total = 0.0;
      if (symmetric_)
      {
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:total) schedule(runtime)
#endif
        for (Index i = 0; i < nrows_; i++)
        {
          const double li = left[i], ri = right[i];
          double s = 0.0;
          const Index kb = row_ptr[i], ke = row_ptr[i + 1];
          for (Index k = kb; k < ke; k++)
          {
            const Index j = col_idx[k];
            const double v = values[k];
            if (j == i) s += li * v * ri;
            else        s += v * (li * right[j] + left[j] * ri);
          }
          total += s;
        }
      }
      else
      {
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:total) schedule(runtime)
#endif
        for (Index i = 0; i < nrows_; i++)
        {
          double sum = 0.0;
          const Index kb = row_ptr[i], ke = row_ptr[i + 1];
          for (Index k = kb; k < ke; k++)
            sum += values[k] * right[col_idx[k]];
          total += left[i] * sum;
        }
      }
      return total;
    }

    // Convert to Eigen SparseMatrix
    auto toEigen()
    {
      assert(finalized);
      assert(values.size() <= 100000000);

      Eigen::SparseMatrix<double, Eigen::RowMajor, long long> mat(nrows_, ncols_);
      mat.reserve(values.size());

      for (Index i = 0; i < nrows_; i++)
      {
        for (Index k = row_ptr[i]; k < row_ptr[i + 1]; k++)
        {
          const auto& c = col_idx[k];
          const auto& v = values[k];

          assert (c >= 0 && c < ncols_);
          assert(std::isfinite(v));

          mat.insert(i, c) = v;
        }
      }

      mat.makeCompressed();
      return mat;
    }
  };

  // y = A*x for a symmetric matrix whose CSR arrays store only the lower
  // triangle (including the diagonal).
  inline void symmetric_spmv_lower(
      const std::vector<CSRMatrix::Index>& row_ptr,
      const std::vector<CSRMatrix::ColIndex>& col_idx,
      const std::vector<double>& values,
      CSRMatrix::Index dim,
      const double* x, double* y,
      std::vector<std::vector<double>>& tbuf)
  {
    using Index = CSRMatrix::Index;
#ifdef USE_OPENMP
#pragma omp parallel
    {
      const int nt  = omp_get_num_threads();
      const int tid = omp_get_thread_num();

#pragma omp single
      if (tbuf.size() != static_cast<size_t>(nt)) tbuf.resize(nt);

      std::vector<double>& buf = tbuf[tid];
      if (buf.size() != static_cast<size_t>(dim)) buf.assign(dim, 0.0);
      else std::fill(buf.begin(), buf.end(), 0.0);

#pragma omp for schedule(runtime)
      for (Index i = 0; i < dim; i++)
      {
        const double xi = x[i];
        double yi = 0.0;
        const Index kb = row_ptr[i], ke = row_ptr[i + 1];
        for (Index k = kb; k < ke; k++)
        {
          const Index j  = col_idx[k];
          const double v = values[k];
          yi += v * x[j];
          if (j != i) buf[j] += v * xi;   // transpose contribution
        }
        buf[i] += yi;
      }

#pragma omp for schedule(static)
      for (Index t = 0; t < dim; t++)
      {
        double s = 0.0;
        for (int u = 0; u < nt; u++) s += tbuf[u][t];
        y[t] = s;
      }
    }
#else
    std::fill(y, y + dim, 0.0);
    for (Index i = 0; i < dim; i++)
    {
      const double xi = x[i];
      double yi = 0.0;
      for (Index k = row_ptr[i]; k < row_ptr[i + 1]; k++)
      {
        const Index j  = col_idx[k];
        const double v = values[k];
        yi += v * x[j];
        if (j != i) y[j] += v * xi;
      }
      y[i] += yi;
    }
    (void)tbuf;
#endif
  }

  struct CSRMatProd // REQUIRED by Spectra
  {
    using Scalar = double;
    using Index  = CSRMatrix::Index;

    const std::vector<Index>& row_ptr;
    const std::vector<CSRMatrix::ColIndex>& col_idx;
    const std::vector<double>& values;
    Index nrows_;
    bool symmetric_;

    mutable std::vector<std::vector<double>> tbuf_;

    CSRMatProd(const CSRMatrix& csr)
      : row_ptr(csr.get_row_ptr()),
        col_idx(csr.get_col_idx()),
        values(csr.get_values()),
        nrows_(csr.nrows_),
        symmetric_(csr.is_symmetric())
    {}

    int rows() const { return static_cast<int>(nrows_); }
    int cols() const { return static_cast<int>(nrows_); }

    void perform_op(const double* x_in, double* y_out) const
    {
      if (symmetric_)
      {
        symmetric_spmv_lower(row_ptr, col_idx, values, nrows_, x_in, y_out, tbuf_);
      }
      else
      {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(runtime)
#endif
        for (Index i = 0; i < nrows_; i++)
        {
          double sum = 0.0;
          const Index k_start = row_ptr[i];
          const Index k_end = row_ptr[i + 1];
          for (Index k = k_start; k < k_end; k++)
          {
            sum += values[k] * x_in[col_idx[k]];
          }
          y_out[i] = sum;
        }
      }

#ifdef USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, y_out, static_cast<int>(nrows_),
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    }
  };
} // namespace utils

namespace qcx_utils = utils;
