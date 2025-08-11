#include <U3xU4MultiShellTensors.h>

#include <numeric>

#include <qcx_convert.h>
#include <qcx_ND.h>

#define FRONT_CREATE
// #define REAR_CREATE

namespace U3xU4
{
  std::vector<double> MultiShellTensor::RME(const U4xSU3::Irrep& bra, 
  	const U4xSU3::Irrep& ket, sam::Databases& dbs) const
  {
    // everything must already be unified
    assert(bra.gamma.size() == ket.gamma.size());
    assert(single_shell_tensors.size() == ket.gamma.size());
    assert(bra.omega.size() == ket.omega.size());
    assert(coupled_irreps.size() = bra.omega.size());

    // If there is no OmegaElement, i.e., only one GammaElement, 
    // both bra and ket GammaElement's are be in the same shell.
    // There must also be only one single-shell tensor U3xU4::Tadat.
    // Hence this becomes single-shell RME problem.
    if (bra.omega.empty())
    {
      const U3xU4::Tadat& tadat = single_shell_tensors[0];

        U3xU4::irrep i = ket.gamma[0].toU3xU4(tadat.n),
                     f = bra.gamma[0].toU3xU4(tadat.n);

        std::size_t rhomax, etamax, ai_max = ket.gamma[0].alphamax(),
                                    af_max = bra.gamma[0].alphamax();

        return tadat.RME(ai_max, i, af_max, f, rhomax, etamax, 
        				 dbs.cgcdb, dbs.wigU3U2db, dbs.wigU4U3db);
    }

    // If there is only one OmegaElement
    else if (bra.omega.size() == 1)
    {
      const U3xU4::Tadat& tadat0 = single_shell_tensors[0],
      					  tadat1 = single_shell_tensors[1];
      const auto& [Omega, Rhomax, Etamax] = coupled_irreps[0];

      U3xU4::irrep ket0 = ket.gamma[0].toU3xU4(tadat0.n),
      			   ket1 = ket.gamma[1].toU3xU4(tadat1.n),
      			   ket_omega = ket.omega[0].toU3xU4(),
      			   bra0 = bra.gamma[0].toU3xU4(tadat0.n),
      			   bra1 = bra.gamma[1].toU3xU4(tadat1.n),
      			   bra_omega = bra.omega[0].toU3xU4();

      std::size_t ket0_alphamax = ket.gamma[0].alphamax(),
      			  ket1_alphamax = ket.gamma[1].alphamax(),
      			  ket_rhomax = ket.omega[0].rhomax(),
      			  ket_etamax = ket.omega[0].etamax(),
                  bra0_alphamax = bra.gamma[0].alphamax(),
      			  bra1_alphamax = bra.gamma[1].alphamax(), 
      			  bra_rhomax = bra.omega[0].rhomax(),
      			  bra_etamax = bra.omega[0].etamax(),
      			  rhomax = u3::Rho(ket_omega.u3, Omega.u3, bra_omega.u3),
      			  etamax = u4::Eta(ket_omega.u4, Omega.u4, bra_omega.u4),
      			  rhomax0, etamax0, rhomax1, etamax1;

      auto rme0 = tadat0.RME(ket0_alphamax, ket0, bra0_alphamax, bra0,
      			rhomax0, etamax0, dbs.cgcdb, dbs.wigU3U2db, dbs.wigU4U3db),
    	   rme1 = tadat1.RME(ket1_alphamax, ket1, bra1_alphamax, bra1,
      			rhomax1, etamax1, dbs.cgcdb, dbs.wigU3U2db, dbs.wigU4U3db);

      auto [allowU3U9, U3U9coeff] = dbs.u3u9db.get(
      		ket0.u3.n13(), ket0.u3.n23(), ket0.u3.n33(),
      		tadat0.T.u3.n13(), tadat0.T.u3.n23(), tadat0.T.u3.n33(),
      		ket1.u3.n13(), ket1.u3.n23(), ket1.u3.n33(),
      		tadat1.T.u3.n13(), tadat1.T.u3.n23(), tadat1.T.u3.n33(),
      		bra0.u3.n13(), bra0.u3.n23(), bra0.u3.n33(),
      		bra1.u3.n13(), bra1.u3.n23(), bra1.u3.n33(),
      		ket_omega.u3.n13(), ket_omega.u3.n23(), ket_omega.u3.n33(),
      		Omega.u3.n13(), Omega.u3.n23(), Omega.u3.n33(),
      		bra_omega.u3.n13(), bra_omega.u3.n23(), bra_omega.u3.n33());
      if (allowU3U9 == false)
      	return {};

      auto [allowU4U9, U4U9coeff] = dbs.u4u9db.get(
      		ket0.u4.n14(), ket0.u4.n24(), ket0.u4.n34(), ket0.u4.n44(),
      		tadat0.T.u4.n14(), tadat0.T.u4.n24(), tadat0.T.u4.n34(), tadat0.T.u4.n44(),
      		ket1.u4.n14(), ket1.u4.n24(), ket1.u4.n34(), ket1.u4.n44(),
      		tadat1.T.u4.n14(), tadat1.T.u4.n24(), tadat1.T.u4.n34(), tadat1.T.u4.n44(),
      		bra0.u4.n14(), bra0.u4.n24(), bra0.u4.n34(), bra0.u4.n44(),
      		bra1.u4.n14(), bra1.u4.n24(), bra1.u4.n34(), bra1.u4.n44(),
      		ket_omega.u4.n14(), ket_omega.u4.n24(), ket_omega.u4.n34(), ket_omega.u4.n44(),
      		Omega.u4.n14(), Omega.u4.n24(), Omega.u4.n34(), Omega.u4.n44(),
      		bra_omega.u4.n14(), bra_omega.u4.n24(), bra_omega.u4.n34(), bra_omega.u4.n44());
      if (allowU4U9 == false)
      	return {};

      qcx_utils::NDIndex multis, multis0, multis1, multisU3, multisU4;

      multis.add_dimensions(ket0_alphamax, ket1_alphamax, ket_rhomax, ket_etamax,
      						bra0_alphamax, bra1_alphamax, bra_rhomax, bra_etamax,
      						tadat0.rho0max, tadat0.eta0max, tadat1.rho0max, tadat1.eta0max,
      						Rhomax, Etamax, rhomax, etamax);

      // These are for getting the single-shell RMEs
      multis0.add_dimensions(ket0_alphamax, bra0_alphamax, 
      		tadat0.rho0max, tadat0.eta0max, rhomax0, etamax0);
      multis1.add_dimensions(ket1_alphamax, bra1_alphamax, 
      		tadat1.rho0max, tadat1.eta0max, rhomax1, etamax1);

      // These are for getting the recoupling coefficients
      multisU3.add_dimensions(ket_rhomax, Rhomax, bra_rhomax,
      						  rhomax0, rhomax1, rhomax);
      multisU4.add_dimensions(ket_etamax, Etamax, bra_etamax,
      						  etamax0, etamax1, etamax);

      auto total_rmes = multis.cardinality();
      std::vector<double> rmes(total_rmes, 0.0);
      std::vector<size_t> multi(16), multi0(6), multi1(6), multiU3(6), multiU4(6);

      double phase = 1.0; // to be implemented later

      for (size_t i = 0; i < total_rmes; i++)
      {
      	multis.to_multi(i, multi);

      	size_t ket0_alpha = multi[0], ket1_alpha = multi[1], 
      		   ket_rho = multi[2], ket_eta = multi[3],
      		   bra0_alpha = multi[4], bra1_alpha = multi[5], 
      		   bra_rho = multi[6], bra_eta = multi[7],
      		   rho0_0 = multi[8], eta0_0 = multi[9], 
      		   rho0_1 = multi[10], eta0_1 = multi[11],
      		   Rho = multi[12], Eta = multi[13], 
      		   rho = multi[14], eta = multi[15];

      	for (size_t rho0 = 0; rho0 < rhomax0; rho0++)
      	  for (size_t eta0 = 0; eta0 < etamax0; eta0++)
      	  	for (size_t rho1 = 0; rho1 < rhomax1; rho1++)
      	  	  for (size_t eta1 = 0; eta1 < etamax1; eta1++)
      	  	  {
      	  	  	multiU3 = {ket_rho, Rho, bra_rho, rho0, rho1, rho};
      	  	  	multiU4 = {ket_eta, Eta, bra_eta, eta0, eta1, eta};
      	  	  	multi0 = {ket0_alpha, bra0_alpha, rho0_0, eta0_0, rho0, eta0};
      	  	  	multi1 = {ket1_alpha, bra1_alpha, rho0_1, eta0_1, rho1, eta1};

      	  	  	rmes[i] += phase 
      	  	  			* U3U9coeff[multisU3.to_linear(multiU3)]
      	  	  			* U4U9coeff[multisU4.to_linear(multiU4)]
      	  	  			* rme0[multis0.to_linear(multi0)]
      	  	  			* rme1[multis1.to_linear(multi1)];
      	  	  }
      }
      return rmes;
    }
  }
} // namespace U3xU4
