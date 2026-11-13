#ifndef SAM_UTILS_READWRITE_H
#define SAM_UTILS_READWRITE_H

// binary file I/O for SAM.
//
// generic stream helpers come from the u3lib numerical foundation:
//   writeBinary(std::ostream&, value | std::vector)
//   readBinary(std::istream&, value | std::vector)
//
// file-level I/O defined here:
//   writeEigenSparse(filename, Eigen::SparseMatrix<double>) -- (i, j, value)
//   readEigenSparse(filename)   -> Eigen::SparseMatrix<double>   triplets
//   writeCSRMatrix(filename, CSRMatrix)                      -- CSR arrays
//   readCSRMatrix(filename)     -> CSRMatrix
//   writeEigenPairs(filename, evals, evecs)                  -- eigenvalues
//   readEigenPairs(filename)    -> (evals, evecs)               and vectors
//
// usage:
//   auto H = op.matrix_CSR(basis);
//   utils::writeCSRMatrix(op.matrix_file(basis), H);
//   auto H2 = utils::readCSRMatrix(op.matrix_file(basis));

#include_next <readwrite.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <matrix.h>       // CSRMatrix
#include <sparse_utils.h> // check_symmetric

namespace utils
{
  // ── Eigen sparse matrix and eigen solutions (moved from u3lib) ────────────

  // write Eigen::SparseMatrix<double> to binary file
  inline void writeEigenSparse(const std::string& filename,
                   const Eigen::SparseMatrix<double>& mat,
                   double threshold = 1e-12)
  {
    std::ofstream out(filename, std::ios::binary);
    if (!out)
      throw std::runtime_error("Cannot open file for writing: " + filename);
      
    std::cout << "Start writing matrix to binary file" << std::endl;

    Eigen::Index rows = mat.rows();
    Eigen::Index cols = mat.cols();

    writeBinary(out, rows);
    writeBinary(out, cols);

    if (rows == cols) // symmetric case
    {
      assert(check_symmetric(mat));

      // Count number of nonzero elements above threshold
      size_t nnz = 0;
      for (int k = 0; k < mat.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
          if (it.row() >= it.col() && std::fabs(it.value()) > threshold)
            ++nnz;

      writeBinary(out, nnz);

      // Write entries
      for (int k = 0; k < mat.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
        {
          if (std::fabs(it.value()) <= threshold) continue;

          Eigen::Index i = it.row();
          Eigen::Index j = it.col();
          if (i < j) continue;
          
          writeBinary(out, i);
          writeBinary(out, j);
          writeBinary(out, it.value());
        }
    }
    else
    {
      // Count number of nonzero elements above threshold
      size_t nnz = 0;
      for (int k = 0; k < mat.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
          if (std::fabs(it.value()) > threshold)
            ++nnz;

      writeBinary(out, nnz);

      // Write entries
      for (int k = 0; k < mat.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
        {
          if (std::fabs(it.value()) <= threshold) continue;

          Eigen::Index i = it.row();
          Eigen::Index j = it.col();
          
          writeBinary(out, i);
          writeBinary(out, j);
          writeBinary(out, it.value());
        }
    }
    
    std::cout << "Finish writing matrix to binary file" << std::endl;
    return;
  }

  // read Eigen::SparseMatrix<double> from binary file
  inline Eigen::SparseMatrix<double> readEigenSparse(const std::string& filename)
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
      throw std::runtime_error("Cannot open file for reading: " + filename);
    
    std::cout << "Start reading matrix from binary file" << std::endl;
      
    Eigen::Index rows, cols;
    size_t nnz;
    readBinary(in, rows);
    readBinary(in, cols);
    readBinary(in, nnz);

    std::vector<Eigen::Triplet<double>> triplets;

    if (rows == cols)
    {
      triplets.reserve(2 * nnz);

      for (size_t k = 0; k < nnz; ++k)
      {
        Eigen::Index i, j;
        double val;
        readBinary(in, i);
        readBinary(in, j);
        readBinary(in, val);

        triplets.emplace_back(i, j, val);
        if (i > j) triplets.emplace_back(j, i, val);
      }
    }
    else
    {
      triplets.reserve(nnz);

      for (size_t k = 0; k < nnz; ++k)
      {
        Eigen::Index i, j;
        double val;
        readBinary(in, i);
        readBinary(in, j);
        readBinary(in, val);
        triplets.emplace_back(i, j, val);
      }
    }

    Eigen::SparseMatrix<double> mat(rows, cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
      
    std::cout << "Finish reading matrix from binary file" << std::endl;
    return mat;
  }

  // write eigenvalues and eigenvectors to binary file
  inline void writeEigenPairs(const std::string& filename,
                          const std::vector<double>& evals,
                          const Eigen::MatrixXd& evecs)
  {
    std::ofstream out(filename, std::ios::binary);
    if (!out)
      throw std::runtime_error("Cannot open file for writing: " + filename);
      
    std::cout << "Start writing eigen solutions to binary file" << std::endl;

    size_t rows = evecs.rows();
    size_t cols = evecs.cols();

    writeBinary(out, rows);
    writeBinary(out, cols);

    writeBinary(out, evals);

    for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++)
        writeBinary(out, evecs(i, j));

    std::cout << "Finish writing eigen solutions to binary file" << std::endl;
    return;
  }

  // write eigenvalues and eigenvectors to binary file
  inline auto readEigenPairs(const std::string& filename)
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
      throw std::runtime_error("Cannot open file for reading: " + filename);
      
    std::cout << "Start reading eigen solutions from binary file" << std::endl;

    size_t rows, cols;
    readBinary(in, rows);
    readBinary(in, cols);

    std::vector<double> evals(cols);
    readBinary(in, evals);

    Eigen::MatrixXd evecs(rows, cols);
    for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++)
      {
        double x;
        readBinary(in, x);
        evecs(i, j) = x;
      }

    std::cout << "Finish reading eigen solutions from binary file" << std::endl;
    return std::make_pair(evals, evecs);
  }


  // ── CSR matrix ────────────────────────────────────────────────────────────

  // write a finalized CSRMatrix to binary file. layout (native byte order):
  //   int64   nrows
  //   int64   ncols
  //   uint8   symmetric   (1: only the lower triangle, col <= row, is stored)
  //   uint64  nnz
  //   int64   row_ptr[nrows + 1]
  //   uint32  col_idx[nnz]
  //   double  values[nnz]
  // the arrays are the CSRMatrix storage written as is, so a symmetric
  // matrix keeps its lower-triangle-only form.
  inline void writeCSRMatrix(const std::string& filename, const CSRMatrix& mat)
  {
    if (!mat.finalized)
      throw std::invalid_argument(
          "Sorry, only a finalized CSRMatrix can be written; "
          "call finalize() first: " + filename);

#ifdef USE_MPI
    // under MPI each rank holds only its share of the entries (the full
    // product is summed across ranks in CSRMatProd), so a single rank's
    // arrays are not the whole matrix.
    int mpi_initialized = 0, mpi_size = 1;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_size > 1)
      throw std::runtime_error(
          "Sorry, writing a CSRMatrix is only supported on a single MPI "
          "rank, since each rank holds only part of the matrix: " + filename);
#endif

    std::ofstream out(filename, std::ios::binary);
    if (!out)
      throw std::runtime_error("Cannot open file for writing: " + filename);

    std::cout << "Start writing CSR matrix to binary file" << std::endl;

    int64_t nrows = mat.nrows();
    int64_t ncols = mat.ncols();
    uint8_t symmetric = mat.is_symmetric() ? 1 : 0;
    uint64_t nnz = mat.get_values().size();

    assert(mat.get_row_ptr().size() == static_cast<size_t>(nrows + 1));
    assert(mat.get_col_idx().size() == nnz);

    writeBinary(out, nrows);
    writeBinary(out, ncols);
    writeBinary(out, symmetric);
    writeBinary(out, nnz);

    writeBinary(out, mat.get_row_ptr());
    writeBinary(out, mat.get_col_idx());
    writeBinary(out, mat.get_values());

    if (!out)
      throw std::runtime_error("Failed while writing: " + filename);

    std::cout << "Finish writing CSR matrix to binary file" << std::endl;
  }

  // read a CSRMatrix written by writeCSRMatrix; the result
  // is finalized and keeps the symmetric (lower-triangle-only) flag.
  inline CSRMatrix readCSRMatrix(const std::string& filename)
  {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
      throw std::runtime_error("Cannot open file for reading: " + filename);

    std::cout << "Start reading CSR matrix from binary file" << std::endl;

    int64_t nrows, ncols;
    uint8_t symmetric;
    uint64_t nnz;
    readBinary(in, nrows);
    readBinary(in, ncols);
    readBinary(in, symmetric);
    readBinary(in, nnz);

    if (!in || nrows < 0 || ncols < 0 || symmetric > 1)
      throw std::runtime_error("Not a valid CSR matrix file: " + filename);

    CSRMatrix mat(nrows, ncols);
    mat.rows.clear(); // per-row build buffers are not needed
    mat.rows.shrink_to_fit();

    mat.row_ptr.resize(nrows + 1);
    mat.col_idx.resize(nnz);
    mat.values.resize(nnz);

    readBinary(in, mat.row_ptr);
    if (nnz > 0)
    {
      readBinary(in, mat.col_idx);
      readBinary(in, mat.values);
    }

    if (!in)
      throw std::runtime_error("CSR matrix file is truncated: " + filename);
    if (mat.row_ptr.front() != 0 ||
        mat.row_ptr.back() != static_cast<CSRMatrix::Index>(nnz))
      throw std::runtime_error("Inconsistent CSR row pointers: " + filename);

    mat.symmetric_ = (symmetric == 1);
    mat.finalized = true;

    std::cout << "Finish reading CSR matrix from binary file" << std::endl;
    return mat;
  }
} // namespace utils

#endif // SAM_UTILS_READWRITE_H
