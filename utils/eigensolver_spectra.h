#pragma once

#include "qcx_matrix.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace utils
{
  inline auto eigensolver_symmetric_spectra(const Eigen::SparseMatrix<double>& EA,
                                            int nev,
                                            double tol = 1e-8)
  {
    std::cout << "Start diagonalizing" << std::endl;

    if (EA.size() == 1)
    {
      std::vector<double> evals;
      evals.push_back(EA.coeff(0, 0));

      Eigen::MatrixXd evecs(1, 1);
      evecs(0, 0) = 1.0;
      
      return std::pair(std::move(evals), std::move(evecs));
    }

    const auto n = EA.rows();
    if (nev > n) nev = n-1;

    int ncv = std::min((int)n, std::max(3*nev + 20, 2*nev + 10));

    Spectra::SparseSymMatProd<double> op(EA);
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, nev, ncv);

    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 
                             10000,   // max iterations
                             tol);

    if (eigs.info() != Spectra::CompInfo::Successful)
      throw std::runtime_error("Spectra eigensolver did not converge");

    Eigen::VectorXd evals = eigs.eigenvalues();
    Eigen::MatrixXd evecs = eigs.eigenvectors();

    std::vector<double> evals_vec(evals.data(), evals.data() + nconv);
    Eigen::MatrixXd evecs_mat = evecs.leftCols(nconv);

    std::reverse(evals_vec.begin(), evals_vec.end());
    evecs_mat = evecs_mat.rowwise().reverse().eval();

    std::cout << "Finish diagonalizing" << std::endl;

    return std::pair(std::move(evals_vec), std::move(evecs_mat));
  }

  inline auto eigensolver_symmetric_spectra(const utils::CSRMatrix& EA,
                                            int nev,
                                            double tol = 1e-10,
                                            int iter = 100000)
  {
#ifdef USE_MPI
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_rank == 0)
#endif
    std::cout << "Start diagonalizing" << std::endl;

    if (EA.size() == 1)
    {
      std::vector<double> evals;
      evals.push_back(EA.get_values()[0]);

      Eigen::MatrixXd evecs(1, 1);
      evecs(0, 0) = 1.0;

      return std::pair(std::move(evals), std::move(evecs));
    }

    const int n = EA.dim();
    if (nev > n) nev = n - 1;
    int ncv = std::min(n, std::max(5*nev + 50, 3*nev + 20));

    std::vector<double> evals_vec;
    Eigen::MatrixXd    evecs_mat;

#ifdef USE_MPI
    if (mpi_size > 1)
    {
      if (mpi_rank == 0)
      {
        // ── Rank 0: owns the Krylov basis and drives Spectra ─────────────
        struct CSRMatProd_rank0
        {
          using Scalar = double;
          using Index  = utils::CSRMatrix::Index;

          const std::vector<Index>&                          row_ptr;
          const std::vector<utils::CSRMatrix::ColIndex>&     col_idx;
          const std::vector<double>&                         values;
          int dim_;
          bool symmetric_;
          mutable std::vector<std::vector<double>> tbuf_;

          CSRMatProd_rank0(const utils::CSRMatrix& csr)
            : row_ptr(csr.get_row_ptr()), col_idx(csr.get_col_idx()),
              values(csr.get_values()),   dim_(csr.dim()),
              symmetric_(csr.is_symmetric())
          {}

          int rows() const { return dim_; }
          int cols() const { return dim_; }

          void perform_op(const double* x_in, double* y_out) const
          {
            int done = 0;
            MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(const_cast<double*>(x_in), dim_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (symmetric_)
            {
              utils::symmetric_spmv_lower(row_ptr, col_idx, values, dim_,
                                          x_in, y_out, tbuf_);
            }
            else
            {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(runtime)
#endif
              for (int i = 0; i < dim_; i++)
              {
                double sum = 0.0;
                for (auto k = row_ptr[i]; k < row_ptr[i+1]; k++)
                  sum += values[k] * x_in[col_idx[k]];
                y_out[i] = sum;
              }
            }

            std::vector<double> buf(y_out, y_out + dim_);
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), dim_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            std::copy(buf.begin(), buf.end(), y_out);
          }
        };

        CSRMatProd_rank0 op(EA);
        Spectra::SymEigsSolver<CSRMatProd_rank0> eigs(op, nev, ncv);

        Eigen::VectorXd start_v = Eigen::VectorXd::Ones(n);
        start_v.normalize();
        eigs.init(start_v.data());

        int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, iter, tol);

        int done = 1;
        MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int nconv_sig = (eigs.info() == Spectra::CompInfo::Successful) ? nconv : -1;
        MPI_Bcast(&nconv_sig, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (nconv_sig < 0)
          throw std::runtime_error("Spectra eigensolver did not converge");

        Eigen::VectorXd evals = eigs.eigenvalues();
        Eigen::MatrixXd evecs = eigs.eigenvectors();

        evals_vec.assign(evals.data(), evals.data() + nconv);
        evecs_mat = evecs.leftCols(nconv);
        std::reverse(evals_vec.begin(), evals_vec.end());
        evecs_mat = evecs_mat.rowwise().reverse().eval();

        MPI_Bcast(evals_vec.data(), nconv,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(evecs_mat.data(), n * nconv, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
      else
      {
        // ── Ranks 1+: helper loop ────────────────────────────────────────
        const auto& row_ptr = EA.get_row_ptr();
        const auto& col_idx = EA.get_col_idx();
        const auto& values  = EA.get_values();
        const bool  symmetric = EA.is_symmetric();

        std::vector<double> x(n), y(n);
        std::vector<std::vector<double>> tbuf;

        while (true)
        {
          int done = 0;
          MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
          if (done) break;

          MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

          if (symmetric)
          {
            utils::symmetric_spmv_lower(row_ptr, col_idx, values, n,
                                        x.data(), y.data(), tbuf);
          }
          else
          {
            std::fill(y.begin(), y.end(), 0.0);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(runtime)
#endif
            for (int i = 0; i < n; i++)
            {
              double sum = 0.0;
              for (auto k = row_ptr[i]; k < row_ptr[i+1]; k++)
                sum += values[k] * x[col_idx[k]];
              y[i] = sum;
            }
          }

          MPI_Allreduce(MPI_IN_PLACE, y.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        int nconv_sig = 0;
        MPI_Bcast(&nconv_sig, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (nconv_sig < 0)
          throw std::runtime_error("Spectra eigensolver did not converge");

        evals_vec.resize(nconv_sig);
        evecs_mat.resize(n, nconv_sig);
        MPI_Bcast(evals_vec.data(), nconv_sig,     MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(evecs_mat.data(), n * nconv_sig, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
    }
    else
#endif
    {
      // ── Serial / single-rank path ────────────────────────────────────────
      utils::CSRMatProd op(EA);
      Spectra::SymEigsSolver<utils::CSRMatProd> eigs(op, nev, ncv);

      Eigen::VectorXd start_v = Eigen::VectorXd::Ones(n);
      start_v.normalize();
      eigs.init(start_v.data());

      int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, iter, tol);

      if (eigs.info() != Spectra::CompInfo::Successful)
        throw std::runtime_error("Spectra eigensolver did not converge");

      Eigen::VectorXd evals = eigs.eigenvalues();
      Eigen::MatrixXd evecs = eigs.eigenvectors();

      evals_vec.assign(evals.data(), evals.data() + nconv);
      evecs_mat = evecs.leftCols(nconv);
      std::reverse(evals_vec.begin(), evals_vec.end());
      evecs_mat = evecs_mat.rowwise().reverse().eval();
    }

#ifdef USE_MPI
    if (mpi_rank == 0)
#endif
    std::cout << "Finish diagonalizing" << std::endl;

    return std::pair(std::move(evals_vec), std::move(evecs_mat));
  }
} // namespace utils

namespace qcx_utils = utils;
