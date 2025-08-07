#include <U3xU4MultiShellTensors.h>

#include <numeric>

#include <qcx_convert.h>
#include <qcx_ND.h>

#define FRONT_CREATE
// #define REAR_CREATE

namespace U3xU4
{
  /* RME: independent of the tensor components
  < bra ||| [Ta x Tb](n13 n23 n33) (n14 n24 n34 n44) ||| ket >_rho,eta

These RMEs are stored in an array of matrices, the index of the matrix is
            alpha_i * max(alpha_f) + alpha_f
For each matrix:
            row = rho * max(eta) + eta
            column = rho_0 * max(eta_0) + eta_0
*/
  std::vector<double>
  OneBodyMultiShellTensor::RME(const U4xSU3::Irrep& bra,
                               const U4xSU3::Irrep& ket) const
  {
    uint16_t rhomax = qcx_utils::itou16(su3::Rho(
        ket.omega.back().su3(), su3::irrep(t.u3), bra.omega.back().su3()));

    uint16_t etamax = qcx_utils::itou16(
        u4::Eta(ket.omega.back().u4(), t.u4, bra.omega.back().u4()));

    // multi-dimensional data for mapping multiplicities to RME
    qcx_utils::NDSpan<uint16_t> multis;

    for (auto Gamma : bra.gamma)
    {
    	std::vector<uint16_t> a(Gamma.alphamax());
    	std::iota(a.begin(), a.end(), 0);
    	multis.add_dimension(a);
    }
    for (auto Omega : bra.omega)
    {
    	std::vector<uint16_t> r(Omega.rhomax());
    	std::iota(r.begin(), r.end(), 0);
    	multis.add_dimension(r);

    	std::vector<uint16_t> e(Omega.etamax());
    	std::iota(e.begin(), e.end(), 0);
    	multis.add_dimension(e);
    }

    for (auto Gamma : ket.gamma)
    {
    	std::vector<uint16_t> a(Gamma.alphamax());
    	std::iota(a.begin(), a.end(), 0);
    	multis.add_dimension(a);
    }
    for (auto Omega : ket.omega)
    {
    	std::vector<uint16_t> r(Omega.rhomax());
    	std::iota(r.begin(), r.end(), 0);
    	multis.add_dimension(r);

    	std::vector<uint16_t> e(Omega.etamax());
    	std::iota(e.begin(), e.end(), 0);
    	multis.add_dimension(e);
    }

    std::vector<uint16_t> r(rhomax);
    std::iota(r.begin(), r.end(), 0);
    multis.add_dimension(r);

    std::vector<uint16_t> e(etamax);
   	std::iota(e.begin(), e.end(), 0);
    multis.add_dimension(e);

    auto total_RME = multis.cardinality();
    std::vector<double> rme(total_RME, 0.0);

    auto num_indices = multis.n_dimensions();
    std::vector<uint16_t> extracted_indices(num_indices);

    

    return rme;
  }
} // namespace U3xU4
