#ifndef SAM_UTILS_EIGENSOLVER_H
#define SAM_UTILS_EIGENSOLVER_H

#include "eigensolver_spectra.h"

namespace utils
{
  inline auto eigensolver_symmetric(const Eigen::SparseMatrix<double>& EA,
                                    int nev, double tol = 1e-8)
  {
    return eigensolver_symmetric_spectra(EA, nev, tol);
  }
} // namespace utils

#endif
