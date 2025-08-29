#include <U3xU4irrep.h>
#include <U3xU4MultiShellTensors.h>

#include <numeric>

#include <qcx_convert.h>
#include <qcx_ND.h>

#define FRONT_CREATE
// #define REAR_CREATE

namespace U3xU4
{
  std::vector<double>
  MultiShellTensor::RME(const U3xU4::ManyNucleonBasis::Irrep& ket,
                        const U3xU4::ManyNucleonBasis::Irrep& bra,
                        sam::Databases& dbs) const
  {
    // everything must already be unified
    assert(bra.gamma.size() == ket.gamma.size());
    assert(Gamma.size() == ket.gamma.size());
    assert(bra.omega.size() == ket.omega.size());
    assert(Omega.size() == bra.omega.size());

    // If there is no OmegaElement, i.e., only one GammaElement, 
    // both bra and ket GammaElement's are be in the same shell.
    // There must also be only one single-shell tensor U3xU4::Tadat.
    // Hence this becomes single-shell RME problem.
    if (bra.omega.empty())
    {
      const U3xU4::SingleShellTensor& tadat = Gamma[0];

      assert(tadat.shell() == ket.gamma[0].shell());
      assert(tadat.shell() == bra.gamma[0].shell());

      // U3xU4::irrep i = ket.gamma[0], f = bra.gamma[0];
      const auto& i = ket.gamma[0];
      const auto& f = bra.gamma[0];

      std::size_t rhomax, etamax; // multi for Wigner-Eckart theorem
      //  			  ai_max = ket.gamma[0].alphamax(), // multi for the ket
      //          af_max = bra.gamma[0].alphamax(); // multi for the bra

      // return tadat.RME(ai_max, i, af_max, f, rhomax, etamax, dbs);
      return tadat.RME(i, f, rhomax, etamax, dbs);
    }

    // If there is only one OmegaElement
    else if (bra.omega.size() == 1)
    {
      // extract single-shell tensors
      const U3xU4::SingleShellTensor& tadat0 = Gamma[0];
      const U3xU4::SingleShellTensor& tadat1 = Gamma[1];

      // extract intershell coupling of the total tensor
      auto Rhomax = Omega[0].rhomax(), Etamax = Omega[0].etamax();
      //const auto& [Omega_last, Rhomax, Etamax] = Omega[0];

      assert(tadat0.shell() == ket.gamma[0].shell());
      assert(tadat1.shell() == ket.gamma[1].shell());
      assert(tadat0.shell() == bra.gamma[0].shell());
      assert(tadat1.shell() == bra.gamma[1].shell());

      // irreps in ket
      const auto& ket0 = ket.gamma[0];
      const auto& ket1 = ket.gamma[1];
      // irreps in bra
      const auto& bra0 = bra.gamma[0];
      const auto& bra1 = bra.gamma[1];

      U3xU4::irrep 
        // irreps in ket
        // ket0 = ket.gamma[0], 
      	// ket1 = ket.gamma[1], 
      	ket_omega = ket.omega[0],
        // irreps in bra
      	// bra0 = bra.gamma[0], 
      	// bra1 = bra.gamma[1],
      	bra_omega = bra.omega[0];

      std::size_t 
        // multis in ket
        ket0_alphamax = ket.gamma[0].alphamax(), 
      	ket1_alphamax = ket.gamma[1].alphamax(),
      	ket_rhomax = ket.omega[0].rhomax(),
  		  ket_etamax = ket.omega[0].etamax(),
        // multis in bra
        bra0_alphamax = bra.gamma[0].alphamax(), 
      	bra1_alphamax = bra.gamma[1].alphamax(), 
      	bra_rhomax = bra.omega[0].rhomax(),
      	bra_etamax = bra.omega[0].etamax(),
      	// multi for Wigner-Eckart theorem for the total tensor
      	rhomax = u3::Rho(ket_omega.u3(), Omega[0].u3(), bra_omega.u3()),  
      	etamax = u4::Eta(ket_omega.u4(), Omega[0].u4(), bra_omega.u4()),
      	// multi for Wigner-Eckart theorem for the first tensor
      	rhomax0, etamax0, 
      	// multi for Wigner-Eckart theorem for the second tensor
      	rhomax1, etamax1;

      // RMEs of the individual tensors
      // auto rmes0 = tadat0.RME(ket0_alphamax, ket0, bra0_alphamax, bra0, rhomax0,
      //                         etamax0, dbs);
      auto rmes0 = tadat0.RME(ket0, bra0, rhomax0, etamax0, dbs);
      if (rmes0.empty()) 
        return {};

      // auto rmes1 = tadat1.RME(ket1_alphamax, ket1, bra1_alphamax, bra1, rhomax1,
      //                       etamax1, dbs);
      auto rmes1 = tadat1.RME(ket1, bra1, rhomax1, etamax1, dbs);
      if (rmes1.empty())
      	return{};

      // 9-U(3) recoupling coefficients
      auto [allowU3U9, U3U9coeff] = dbs.u3u9db.get(
          ket0.u3().n13(), ket0.u3().n23(), ket0.u3().n33(),
          tadat0.T().u3().n13(), tadat0.T().u3().n23(), tadat0.T().u3().n33(),
          ket1.u3().n13(), ket1.u3().n23(), ket1.u3().n33(),
          tadat1.T().u3().n13(), tadat1.T().u3().n23(), tadat1.T().u3().n33(),
          bra0.u3().n13(), bra0.u3().n23(), bra0.u3().n33(), bra1.u3().n13(),
          bra1.u3().n23(), bra1.u3().n33(), ket_omega.u3().n13(),
          ket_omega.u3().n23(), ket_omega.u3().n33(), Omega[0].u3().n13(),
          Omega[0].u3().n23(), Omega[0].u3().n33(), bra_omega.u3().n13(),
          bra_omega.u3().n23(), bra_omega.u3().n33());
      if (allowU3U9 == false)
      	return {};

      // 9-U(4) recoupling coefficients
      auto [allowU4U9, U4U9coeff] = dbs.u4u9db.get(
          ket0.u4().n14(), ket0.u4().n24(), ket0.u4().n34(), ket0.u4().n44(),
          tadat0.T().u4().n14(), tadat0.T().u4().n24(), tadat0.T().u4().n34(),
          tadat0.T().u4().n44(), ket1.u4().n14(), ket1.u4().n24(),
          ket1.u4().n34(), ket1.u4().n44(), tadat1.T().u4().n14(),
          tadat1.T().u4().n24(), tadat1.T().u4().n34(), tadat1.T().u4().n44(),
          bra0.u4().n14(), bra0.u4().n24(), bra0.u4().n34(), bra0.u4().n44(),
          bra1.u4().n14(), bra1.u4().n24(), bra1.u4().n34(), bra1.u4().n44(),
          ket_omega.u4().n14(), ket_omega.u4().n24(), ket_omega.u4().n34(),
          ket_omega.u4().n44(), Omega[0].u4().n14(), Omega[0].u4().n24(),
          Omega[0].u4().n34(), Omega[0].u4().n44(), bra_omega.u4().n14(),
          bra_omega.u4().n24(), bra_omega.u4().n34(), bra_omega.u4().n44());
      if (allowU4U9 == false)
        return {};

      // multi-dimensional indices
      qcx_utils::NDIndex multis,   // for the RMEs of the total tensor
       					 multis0,  // for the RMEs of the first tensor
       					 multis1,  // for the RMEs of the second tensor
       					 multisU3, // for the 9-U(3) recoupling coefficient
       					 multisU4; // for the 9-U(4) recoupling coefficient

      multis.add_dimensions(
          ket0_alphamax, ket1_alphamax, ket_rhomax,
          ket_etamax, // multis for ket
          bra0_alphamax, bra1_alphamax, bra_rhomax,
          bra_etamax,                         // multis for bra
          tadat0.rho0max(), tadat0.eta0max(), // multis for the first tensor
          tadat1.rho0max(), tadat1.eta0max(), // multis for the second tensor
          Rhomax, Etamax,                     // multis for the total tensor
          rhomax, etamax); // multis for Wigner-Eckart theorem

      multis0.add_dimensions(ket0_alphamax, bra0_alphamax, tadat0.rho0max(),
                             tadat0.eta0max(), rhomax0, etamax0);

      multis1.add_dimensions(ket1_alphamax, bra1_alphamax, tadat1.rho0max(),
                             tadat1.eta0max(), rhomax1, etamax1);

      multisU3.add_dimensions(ket_rhomax, Rhomax, bra_rhomax,
      						  rhomax0, rhomax1, rhomax);

      multisU4.add_dimensions(ket_etamax, Etamax, bra_etamax,
      						  etamax0, etamax1, etamax);

      auto total_rmes = multis.cardinality();
      std::vector<double> rmes(total_rmes, 0.0);
      std::vector<size_t> multi(16), multi0(6), multi1(6), multiU3(6), multiU4(6);

      // Please double check this phase with lsu3shell
      int phi, N0 = tadat0.ads + tadat0.ats, N1 = tadat1.ads + tadat1.ats,
      	  n0_ket = ket0.particles(), n1_ket = ket1.particles(),
      	  n0_bra = bra0.particles(), n1_bra = bra1.particles();

#ifdef FRONT_CREATE
      phi = n0_ket * N1 + n1_bra * (N0 + n0_ket + n0_bra);
#endif
#ifdef REAR_CREATE
      phi = n1_bra * N0 + n0_ket * (N1 + n1_ket + n1_bra);
#endif

      double phase = (phi % 2 == 0) ? 1.0 : -1.0;

      for (size_t i = 0; i < total_rmes; i++)
      {
      	multis.to_multi(i, multi);

      	size_t ket0_alpha = multi[0], // multis for ket
      		   ket1_alpha = multi[1], 
      		   ket_rho = multi[2], 
      		   ket_eta = multi[3],
      		   bra0_alpha = multi[4], // multis for bra
      		   bra1_alpha = multi[5], 
      		   bra_rho = multi[6], 
      		   bra_eta = multi[7],
      		   rho0_0 = multi[8], eta0_0 = multi[9], // multis for the first tensor
      		   rho0_1 = multi[10], eta0_1 = multi[11], // multis for the second tensor
      		   Rho = multi[12], Eta = multi[13], // multis for the total tensor
      		   rho = multi[14], eta = multi[15]; // multis for Wigner-Eckart theorem

      	for (size_t rho0 = 0; rho0 < rhomax0; rho0++)
      	{
      	  for (size_t rho1 = 0; rho1 < rhomax1; rho1++)
      	  {
      	  	multiU3 = {ket_rho, Rho, bra_rho, rho0, rho1, rho};
      	  	auto u3u9 = U3U9coeff[multisU3.to_linear(multiU3)];

      	  	for (size_t eta0 = 0; eta0 < etamax0; eta0++)
      	  	{
      	  	  for (size_t eta1 = 0; eta1 < etamax1; eta1++)
      	  	  {
      	  	  	multiU4 = {ket_eta, Eta, bra_eta, eta0, eta1, eta};
      	  	  	auto u4u9 = U4U9coeff[multisU4.to_linear(multiU4)];

      	  	  	multi0 = {ket0_alpha, bra0_alpha, rho0_0, eta0_0, rho0, eta0};
      	  	  	auto rme0 = rmes0[multis0.to_linear(multi0)];

      	  	  	multi1 = {ket1_alpha, bra1_alpha, rho0_1, eta0_1, rho1, eta1};
      	  	  	auto rme1 = rmes1[multis1.to_linear(multi1)];

      	  	  	rmes[i] += u3u9 * u4u9 * rme0 * rme1;
      	  	  }
      	  	}
      	  }
      	}

        rmes[i] *= phase;
      }
      return rmes;
    }

    // If there are more than one OmegaElement, there is a recursive cycle
    else
    {
      // extract the last single-shell tensor
      const U3xU4::SingleShellTensor& tadat_last = Gamma.back();
      // extract the last intershell coupling of the total tensor
      auto Rhomax_last = Omega.back().rhomax(), 
           Etamax_last = Omega.back().etamax();
      //const auto& [Omega_last, Rhomax_last, Etamax_last] = Omega.back();

      // extract the rest of the tensor
      auto new_Gamma = Gamma;
      new_Gamma.pop_back();
      auto new_Omega = Omega;
      new_Omega.pop_back();
      auto Rhomax_semilast = new_Omega.back().rhomax(),
           Etamax_semilast = new_Omega.back().etamax();
      //const auto& [Omega_semilast, Rhomax_semilast, Etamax_semilast] = new_Omega.back();
      U3xU4::MultiShellTensor new_tensor(new_Gamma, new_Omega);

      // extract the rest of the ket
      auto new_ket_gamma = ket.gamma;
      new_ket_gamma.pop_back();
      auto new_ket_omega = ket.omega;
      new_ket_omega.pop_back();
      U3xU4::ManyNucleonBasis::Irrep new_ket(new_ket_gamma, new_ket_omega);

      // extract the rest of the bra
      auto new_bra_gamma = bra.gamma;
      new_bra_gamma.pop_back();
      auto new_bra_omega = bra.omega;
      new_bra_omega.pop_back();
      U3xU4::ManyNucleonBasis::Irrep new_bra(new_bra_gamma, new_bra_omega);

      assert(ket.gamma.back().shell() == tadat_last.shell());
      assert(bra.gamma.back().shell() == tadat_last.shell());
      // last irrep in ket
      const auto& ket_gamma_last = ket.gamma.back();
      // last irreps in bra
      const auto& bra_gamma_last = bra.gamma.back();
      U3xU4::irrep 
      	// last irreps in ket
      	// ket_gamma_last = ket.gamma.back(), 
      	ket_omega_last = ket.omega.back(),
      	ket_omega_semilast = ket.omega[ket.omega.size() - 2],
      	// last irreps in bra
      	// bra_gamma_last = bra.gamma.back(), 
      	bra_omega_last = bra.omega.back(),
      	bra_omega_semilast = bra.omega[bra.omega.size() - 2];

      std::size_t 
      	// last multis in ket
      	ket_alphamax_last = ket.gamma.back().alphamax(), 
      	ket_rhomax_last = ket.omega.back().rhomax(),
      	ket_etamax_last = ket.omega.back().etamax(),
      	ket_rhomax_semilast = ket.omega[ket.omega.size() - 2].rhomax(),
      	ket_etamax_semilast = ket.omega[ket.omega.size() - 2].etamax(),
      	// last multis in bra
        bra_alphamax_last = bra.gamma.back().alphamax(), 
      	bra_rhomax_last = bra.omega.back().rhomax(),
      	bra_etamax_last = bra.omega.back().etamax(),
      	bra_rhomax_semilast = bra.omega[bra.omega.size() - 2].rhomax(),
      	bra_etamax_semilast = bra.omega[bra.omega.size() - 2].etamax(),
      	// multi for Wigner-Eckart theorem for the total tensor
      	rhomax = u3::Rho(ket_omega_last.u3(), Omega.back().u3(), bra_omega_last.u3()),  
      	etamax = u4::Eta(ket_omega_last.u4(), Omega.back().u4(), bra_omega_last.u4()),
      	// multi for Wigner-Eckart theorem for the new tensor
      	rhomax_new = u3::Rho(ket_omega_semilast.u3(), new_Omega.back().u3(), bra_omega_semilast.u3()), 
        etamax_new = u4::Eta(ket_omega_semilast.u4(), new_Omega.back().u4(), bra_omega_semilast.u4()), 
      	// multi for Wigner-Eckart theorem for the last tensor
      	rhomax_last = 0, etamax_last = 0;

      // RMEs of the individual tensors			  
      auto rmes_new = new_tensor.RME(new_ket, new_bra, dbs);
      if (rmes_new.empty())
        return {};
      //for (auto& rme_new : rmes_new) std::cout << rme_new << std::endl;

      // auto rmes_last =
      //     tadat_last.RME(ket_alphamax_last, ket_gamma_last, bra_alphamax_last,
      //                    bra_gamma_last, rhomax_last, etamax_last, dbs);
      auto rmes_last =
          tadat_last.RME(ket_gamma_last, bra_gamma_last, rhomax_last, etamax_last, dbs);
      if (rmes_last.empty())
      	return {};
      //for (auto& rme_last : rmes_last) std::cout << rme_last << std::endl;

      //std::cout << rhomax_new << ";" << rhomax_last << ";"
      //          << etamax_new << ";" << etamax_last << std::endl;

      // 9-U(3) recoupling coefficients
      auto [allowU3U9, U3U9coeff] = dbs.u3u9db.get(
          ket_omega_semilast.u3().n13(), ket_omega_semilast.u3().n23(),
          ket_omega_semilast.u3().n33(), new_Omega.back().u3().n13(),
          new_Omega.back().u3().n23(), new_Omega.back().u3().n33(),
          ket_gamma_last.u3().n13(), ket_gamma_last.u3().n23(),
          ket_gamma_last.u3().n33(), tadat_last.T().u3().n13(),
          tadat_last.T().u3().n23(), tadat_last.T().u3().n33(),
          bra_omega_semilast.u3().n13(), bra_omega_semilast.u3().n23(),
          bra_omega_semilast.u3().n33(), bra_gamma_last.u3().n13(),
          bra_gamma_last.u3().n23(), bra_gamma_last.u3().n33(),
          ket_omega_last.u3().n13(), ket_omega_last.u3().n23(),
          ket_omega_last.u3().n33(), Omega.back().u3().n13(),
          Omega.back().u3().n23(), Omega.back().u3().n33(),
          bra_omega_last.u3().n13(), bra_omega_last.u3().n23(),
          bra_omega_last.u3().n33());
      if (allowU3U9 == false)
        return {};
      // for (auto& u3u9 : U3U9coeff) std::cout << u3u9 << std::endl;

      // 9-U(4) recoupling coefficients
      auto [allowU4U9, U4U9coeff] = dbs.u4u9db.get(
          ket_omega_semilast.u4().n14(), ket_omega_semilast.u4().n24(),
          ket_omega_semilast.u4().n34(), ket_omega_semilast.u4().n44(),
          new_Omega.back().u4().n14(), new_Omega.back().u4().n24(),
          new_Omega.back().u4().n34(), new_Omega.back().u4().n44(),
          ket_gamma_last.u4().n14(), ket_gamma_last.u4().n24(),
          ket_gamma_last.u4().n34(), ket_gamma_last.u4().n44(),
          tadat_last.T().u4().n14(), tadat_last.T().u4().n24(),
          tadat_last.T().u4().n34(), tadat_last.T().u4().n44(),
          bra_omega_semilast.u4().n14(), bra_omega_semilast.u4().n24(),
          bra_omega_semilast.u4().n34(), bra_omega_semilast.u4().n44(),
          bra_gamma_last.u4().n14(), bra_gamma_last.u4().n24(),
          bra_gamma_last.u4().n34(), bra_gamma_last.u4().n44(),
          ket_omega_last.u4().n14(), ket_omega_last.u4().n24(),
          ket_omega_last.u4().n34(), ket_omega_last.u4().n44(),
          Omega.back().u4().n14(), Omega.back().u4().n24(),
          Omega.back().u4().n34(), Omega.back().u4().n44(),
          bra_omega_last.u4().n14(), bra_omega_last.u4().n24(),
          bra_omega_last.u4().n34(), bra_omega_last.u4().n44());
      if (allowU4U9 == false)
      	return {};
      //for (auto& u4u9 : U4U9coeff) std::cout << u4u9 << std::endl;

      // multi-dimensional indices
      qcx_utils::NDIndex multis,   	  // for the RMEs of the total tensor
       					 multis_new,  // for the RMEs of the new tensor
       					 multis_last, // for the RMEs of the last tensor
       					 multisU3, 	  // for the 9-U(3) recoupling coefficient
       					 multisU4;    // for the 9-U(4) recoupling coefficient

      // add multis from ket
      for (const auto& gamma : ket.gamma) 
        multis.add_dimension(gamma.alphamax());
      for (const auto& omega : ket.omega) 
        multis.add_dimensions(omega.rhomax(), omega.etamax());
      // add multis from bra
      for (const auto& gamma : bra.gamma) 
        multis.add_dimension(gamma.alphamax());
      for (const auto& omega : bra.omega) 
        multis.add_dimensions(omega.rhomax(), omega.etamax());
      // add multis from tensor
      for (const auto& tensor : Gamma) 
        multis.add_dimensions(tensor.rho0max(), tensor.eta0max());
      for (const auto& o : Omega)
        multis.add_dimensions(o.rhomax(), o.etamax());
      // add multis from Wigner-Eckart theorem
      multis.add_dimensions(rhomax, etamax);

      // add multis_new from new_ket
      for (const auto& gamma : new_ket.gamma) 
        multis_new.add_dimension(gamma.alphamax());
      for (const auto& omega : new_ket.omega) 
        multis_new.add_dimensions(omega.rhomax(), omega.etamax());
      // add multis_new from new_bra
      for (const auto& gamma : new_bra.gamma) 
        multis_new.add_dimension(gamma.alphamax());
      for (const auto& omega : new_bra.omega) 
        multis_new.add_dimensions(omega.rhomax(), omega.etamax());
      // add multis_new from new_tensor
      for (const auto& tensor : new_tensor.Gamma) 
        multis_new.add_dimensions(tensor.rho0max(), tensor.eta0max());
      for (const auto& o : new_tensor.Omega)
        multis_new.add_dimensions(o.rhomax(), o.etamax());
      // add multis_new from Wigner-Eckart theorem
      multis_new.add_dimensions(rhomax_new, etamax_new);

      multis_last.add_dimensions(ket_alphamax_last, bra_alphamax_last,
                                 tadat_last.rho0max(), tadat_last.eta0max(),
                                 rhomax_last, etamax_last);

      multisU3.add_dimensions(ket_rhomax_last, Rhomax_last, bra_rhomax_last,
                              rhomax_new, rhomax_last, rhomax);

      multisU4.add_dimensions(ket_etamax_last, Etamax_last, bra_etamax_last,
                              etamax_new, etamax_last, etamax);

      auto total_rmes = multis.cardinality();
      std::vector<double> rmes(total_rmes, 0.0);
      std::vector<size_t> multi(multis.n_dimensions()), 
        multi_new(multis_new.n_dimensions()), multi_last(6), multiU3(6), multiU4(6);

      // Please double check this phase with lsu3shell
      int phi, N_new = 0, N_last = tadat_last.ads + tadat_last.ats,
          n_new_ket = 0, n_last_ket = ket_gamma_last.particles(),
          n_new_bra = 0, n_last_bra = bra_gamma_last.particles();

      for (const auto& gamma : new_ket.gamma) 
        n_new_ket += gamma.u4().quanta();
      for (const auto& gamma : new_bra.gamma) 
        n_new_bra += gamma.u4().quanta();
      for (const auto& tensor : new_tensor.Gamma)
        N_new += tensor.ads + tensor.ats;

#ifdef FRONT_CREATE
      phi = n_new_ket * N_last + n_last_bra * (N_new + n_new_ket + n_new_bra);
#endif
#ifdef REAR_CREATE
      phi = n_last_bra * N_new + n_new_ket * (N_last + n_last_ket + n_last_bra);
#endif

      double phase = (phi % 2 == 0) ? 1.0 : -1.0;

      for (size_t i = 0; i < total_rmes; i++)
      {
        multis.to_multi(i, multi);

        size_t j = 0, k = 0;
        for (const auto& gamma : new_ket.gamma)
        {
          multi_new[k] = multi[j];
          k++;
          j++;
        }
        multi_last[0] = multi[j];
        j++;
        for (const auto& omega : new_ket.omega)
        {
          multi_new[k] = multi[j];
          multi_new[k + 1] = multi[j + 1];
          k += 2;
          j += 2;
        }
        multiU3[0] = multi[j];
        j++;
        multiU4[0] = multi[j];
        j++;

        for (const auto& gamma : new_bra.gamma)
        {
          multi_new[k] = multi[j];
          k++;
          j++;
        }
        multi_last[1] = multi[j];
        j++;
        for (const auto& omega : new_bra.omega)
        {
          multi_new[k] = multi[j];
          multi_new[k + 1] = multi[j + 1];
          k += 2;
          j += 2;
        }
        multiU3[2] = multi[j];
        j++;
        multiU4[2] = multi[j];
        j++;

        for (const auto& tensor : new_tensor.Gamma)
        {
          multi_new[k] = multi[j];
          multi_new[k + 1] = multi[j + 1];
          k += 2;
          j += 2;
        }
        multi_last[2] = multi[j];
        j++;
        multi_last[3] = multi[j];
        j++;

        for (const auto& o : new_tensor.Omega)
        {
          multi_new[k] = multi[j];
          multi_new[k + 1] = multi[j + 1];
          k += 2;
          j += 2;
        }
        multiU3[1] = multi[j];
        j++;
        multiU4[1] = multi[j];
        j++;

        multiU3[5] = multi[j];
        j++;
        multiU4[5] = multi[j];
        j++;

        for (size_t rho_new = 0; rho_new < rhomax_new; rho_new++)
        {
          for (size_t rho_last = 0; rho_last < rhomax_last; rho_last++)
          {
            multiU3[3] = rho_new;
            multiU3[4] = rho_last;
            auto u3u9 = U3U9coeff[multisU3.to_linear(multiU3)];

            for (size_t eta_new = 0; eta_new < etamax_new; eta_new++)
            {
              for (size_t eta_last = 0; eta_last < etamax_last; eta_last++)
              {
                multiU4[3] = eta_new;
                multiU4[4] = eta_last;
                auto u4u9 = U4U9coeff[multisU4.to_linear(multiU4)];

                multi_new[k] = rho_new;
                multi_new[k + 1] = eta_new;
                auto rme_new = rmes_new[multis_new.to_linear(multi_new)];

                multi_last[4] = rho_last;
                multi_last[5] = eta_last;
                auto rme_last = rmes_last[multis_last.to_linear(multi_last)];

                //std::cout << u3u9 << ";" << u4u9 << ";" 
                //          << rme_new << ";" << rme_last << std::endl; 

                rmes[i] += u3u9 * u4u9 * rme_new * rme_last;
              }
            }
          }
        }

        rmes[i] *= phase;
      }

      return rmes;
    }
  }
} // namespace U3xU4
