#pragma once

#include <vector>

#include <Eigen/Dense>

#include <so3Coeffs/SO3_CGC.h>

#include <U3basis.h>
#include <U3U2xU3U2.h>

#include <U4basis.h>
#include <U4U3xU4U3.h>

#include <U3xU4HWSgenerator.h>
#include <U3xU4SingleShellTensorsDecomposition.h>

namespace U3xU4
{
  /* Goal: compute RMEs of single-shell U(3)xU(4) tensors
                                                                   rho_0 U(3)_0
  < alpha_f U(3)_f U(4)_f ||| Tadat			 ||| alpha_i U(3)_i
  U(4)_i >_{rho,eta} eta_0 U(4)_0 where rho_0 and eta_0 are from the structure
  of the tensor itself, whereas U(3)_i x U(3)_0 --> U(3)_f with mutliplicity rho
     U(4)_i x U(4)_0 --> U(4)_f with multiplicity eta

     These RMEs are stored in an array of matrices, the index of the matrix is
                  alpha_i * max(alpha_f) + alpha_f
     For each matrix:
                  row = rho * max(eta) + eta
                  column = rho_0 * max(eta_0) + eta_0

     Inputs from user:
     i3, i4: initial irreps
     f3, f4: final irreps
     ads: number of ad's
     ats: number of a's (or at's)
     prod_ads: coupled irreps of ad's
     prod_ats: coupled irreps of at's
     t3, t4: final tensor characters
     rho0max: multiplicity of t3
     eta0max: multiplicity of t4
     cgcs, wigU3U2db, wigU4U3db: databases for coupling coefficients

     Output:
     RMEs stored in an array of matrices
     ai_max: outer multiplicity of i3
     af_max: outer multiplicity of f3
     rhomax, etamax: multiplicities of RMEs
  */
  std::vector<Eigen::MatrixXd>
  SingleShellRME(std::size_t& ai_max, const u3::irrep& i3, const u4::irrep& i4,
                 std::size_t& af_max, const u3::irrep& f3, const u4::irrep& f4,
                 std::size_t ads, std::size_t ats,
                 const std::vector<std::pair<u3::irrep, u4::irrep>>& prod_ads,
                 const std::vector<std::pair<u3::irrep, u4::irrep>>& prod_ats,
                 const u3::irrep& t3, std::size_t rho0max, const u4::irrep& t4,
                 std::size_t eta0max, std::size_t& rhomax, std::size_t& etamax,
                 u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
                 u4::WigU4U3_db& wigU4U3db);

} // namespace U3xU4