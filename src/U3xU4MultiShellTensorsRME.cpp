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
  */
  std::vector<double> OneBodyMultiShellTensor::RME(
      const U4xSU3::ManyNucleonBasis::Irrep& bra,
      const U4xSU3::ManyNucleonBasis::Irrep& ket, qcx_utils::SO3_CGC& cgcs,
      u3::WigU3U2_db& wigU3U2db, u4::WigU4U3_db& wigU4U3db,
      u3::U3U9_db& U3U9db, u4::U4U9_db& U4U9db) const
  {
  	auto na = a.u3.n13(), nb = b.u3.n13();

    U4xSU3::IrrepUnifier iu;
    const auto& [ubra, uket] = iu.unify(bra, ket);
    // ubra and uket have the same size

    // If there is no OmegaElement, i.e., one GammaElement, after unifying,
    // both bra and ket GammaElements must be in the same shell.
    // Hence this becomes single-shell RME problem.
    if (ubra.omega.empty())
    {
      // everything must be in the same shell
      if (na == nb && na == bra.active_shells[0])
      {
        assert(bra.active_shells[0] == ket.active_shells[0]);

        U3xU4::Tadat tadat(na, 1, 1, {}, {}, t, 1, 1);

        U3xU4::irrep i = uket.gamma[0].toU3xU4(na),
                     f = ubra.gamma[0].toU3xU4(na);

        std::size_t rhomax, etamax, ai_max = uket.gamma[0].alphamax(),
                                    af_max = ubra.gamma[0].alphamax();

        return tadat.RME(ai_max, i, af_max, f, rhomax, etamax, cgcs, wigU3U2db,
                         wigU4U3db);
      }
      else
        return {};
    }

    uint16_t rhomax = qcx_utils::itou16(su3::Rho(
        uket.omega.back().su3(), su3::irrep(t.u3), ubra.omega.back().su3()));

    uint16_t etamax = qcx_utils::itou16(
        u4::Eta(uket.omega.back().u4(), t.u4, ubra.omega.back().u4()));

    if (rhomax == 0 || etamax == 0)
      return {};    

  	// [a+(same shell) x a-(same shell)] 
    if (na == nb) 
      if (ubra.dist[na] == 0 || uket.dist[na] == 0 ||  
      	  ubra.dist[na] != uket.dist[na])
        return {};

    // [a+(lower shell) x a- (higher shell)]
    if (a.u3.n23() < b.u3.n23()) 
      if (ubra.dist[na] != uket.dist[na] + 1 ||
      	  ubra.dist[nb] != uket.dist[nb] - 1)
      	return {};

    // [a-(lower shell) x a+ (higher shell)]
    if (a.u3.n23() > b.u3.n23()) 
      if (ubra.dist[na] != uket.dist[na] - 1 ||
      	  ubra.dist[nb] != uket.dist[nb] + 1)
      	return {};

    // other shells must have the same number of particles
    for (size_t n = 0; n < uket.dist.size() && n != na && n != nb; n++)
      if (ubra.dist[n] != uket.dist[n])
      	return {};

    // Once all these selection rules have been passed, 
    // we can calculate multi-shell RMEs.

    // If there is only one OmegaElement, i.e., two GammaElement's
    if (ubra.omega.size() == 1)
    {
      if (na < nb)
      {
      	U3xU4::Tadat ta(na, (a.u3.n23() == 0) ? 1 : 0, 
      					(a.u3.n23() > 0) ? 1 : 0, {}, {}, a, 1, 1),
      				 tb(nb, (b.u3.n23() == 0) ? 1 : 0, 
      					(b.u3.n23() > 0) ? 1 : 0, {}, {}, b, 1, 1);

      	U3xU4::irrep ket_a = uket.gamma[0].toU3xU4(na),
      				 ket_b = uket.gamma[1].toU3xU4(nb),
      				 ket_omega = uket.omega[0].toU3xU4(),
      				 bra_a = ubra.gamma[0].toU3xU4(na),
      				 bra_b = ubra.gamma[1].toU3xU4(nb),
      				 bra_omega = ubra.omega[0].toU3xU4();

      	std::size_t ket_a_alphamax = uket.gamma[0].alphamax(),
      				ket_b_alphamax = uket.gamma[1].alphamax(),
                    bra_a_alphamax = ubra.gamma[0].alphamax(),
      				bra_b_alphamax = ubra.gamma[1].alphamax(), multi = 1;

      	auto rme_a = ta.RME(ket_a_alphamax, ket_a, bra_a_alphamax, bra_a,
      						multi, multi, cgcs, wigU3U2db, wigU4U3db),
      		 rme_b = tb.RME(ket_b_alphamax, ket_b, bra_b_alphamax, bra_b,
      						multi, multi, cgcs, wigU3U2db, wigU4U3db);

      	auto [allowU3U9, U3U9coeff] = U3U9db.get(
      		ket_a.u3.n13(), ket_a.u3.n23(), ket_a.u3.n33(),
      		a.u3.n13(), a.u3.n23(), a.u3.n33(),
      		ket_b.u3.n13(), ket_b.u3.n23(), ket_b.u3.n33(),
      		b.u3.n13(), b.u3.n23(), b.u3.n33(),
      		bra_a.u3.n13(), bra_a.u3.n23(), bra_a.u3.n33(),
      		bra_b.u3.n13(), bra_b.u3.n23(), bra_b.u3.n33(),
      		ket_omega.u3.n13(), ket_omega.u3.n23(), ket_omega.u3.n33(),
      		t.u3.n13(), t.u3.n23(), t.u3.n33(),
      		bra_omega.u3.n13(), bra_omega.u3.n23(), bra_omega.u3.n33());

      	auto [allowU4U9, U4U9coeff] = U4U9db.get(
      		ket_a.u4.n14(), ket_a.u4.n24(), ket_a.u4.n34(), ket_a.u4.n44(),
      		a.u4.n14(), a.u4.n24(), a.u4.n34(), a.u4.n44(),
      		ket_b.u4.n14(), ket_b.u4.n24(), ket_b.u4.n34(), ket_b.u4.n44(),
      		b.u4.n14(), b.u4.n24(), b.u4.n34(), b.u4.n44(),
      		bra_a.u4.n14(), bra_a.u4.n24(), bra_a.u4.n34(), bra_a.u4.n44(),
      		bra_b.u4.n14(), bra_b.u4.n24(), bra_b.u4.n34(), bra_b.u4.n44(),
      		ket_omega.u4.n14(), ket_omega.u4.n24(), ket_omega.u4.n34(), ket_omega.u4.n44(),
      		t.u4.n14(), t.u4.n24(), t.u4.n34(), t.u4.n44(),
      		bra_omega.u4.n14(), bra_omega.u4.n24(), bra_omega.u4.n34(), bra_omega.u4.n44());

      	
      }
    }

    // multi-dimensional data for mapping multiplicities to RME
    qcx_utils::NDIndex multis;

    for (auto Gamma : bra.gamma)
      multis.add_dimension(Gamma.alphamax());
    for (auto Omega : bra.omega)
      multis.add_dimensions(qcx_utils::itou16(Omega.rhomax()),
                            qcx_utils::itou16(Omega.etamax()));

    for (auto Gamma : ket.gamma)
      multis.add_dimension(Gamma.alphamax());
    for (auto Omega : ket.omega)
      multis.add_dimensions(qcx_utils::itou16(Omega.rhomax()),
                            qcx_utils::itou16(Omega.etamax()));

    multis.add_dimensions(rhomax, etamax);

    auto total_RME = multis.cardinality();
    std::vector<double> RME(total_RME, 0.0);

    auto num_multi = multis.n_dimensions();
    std::vector<uint16_t> multi(num_multi);

    // Case: single-shell tensor between multi-shell irreps
    if (na == nb)
    {
    	uint16_t idx = 0; // to be changed

    	U3xU4::Tadat tadat(na, 1, 1, {}, {}, t, 1, 1);

        U3xU4::irrep gamma_idx_ket = uket.gamma[idx].toU3xU4(na),
                     gamma_idx_bra = ubra.gamma[idx].toU3xU4(na);

        std::size_t rho_idx_max, eta_idx_max, 
        			amax_idx_ket = uket.gamma[idx].alphamax(),
                    amax_idx_bra = ubra.gamma[idx].alphamax();

        auto rme = tadat.RME(amax_idx_ket, gamma_idx_ket, 
        					 amax_idx_bra, gamma_idx_bra, 
        					 rho_idx_max, eta_idx_max, 
        					 cgcs, wigU3U2db, wigU4U3db);

        // Going through the identities prior to our tensor
        for (size_t i = 0; i < idx; i++)
        {
          for (size_t j = 0; j < total_RME; j++)
          {
          	multis.to_multi(j, multi);

          }
        }
    }

    return RME;
  }
} // namespace U3xU4
