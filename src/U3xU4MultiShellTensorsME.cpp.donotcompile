#include <ostream>

#include <qcx_ND.h>

#include <U3xU4MultiShellTensors.h>

namespace U3xU4
{
  /* Goal:
         Compute matrix elements of a JT-coupled operator O, which is
     expanded in terms of many U3xU4::MultiShellTensor_phys objects.
     By nature, these tensors must have the same J,MJ,T,MT quantum
     numbers with the operator O, they only differ in U(3)->SO(3) and
     U(4)->SU(2)xSU(2) labels.

     Structure of tensor strengths: a 2D array
     - 1st dimension: goes with the dimension of subgroup label array
     - 2nd dimension: goes with all the multiplicities of the tensor,
       i.e., Gamma & Omega multiplicities, kappa and zeta.

     In addition, these tensors as input are not unified with the
     basis states; the necessity for unification only arises when
     calculating multi-shell RMEs. The matrix element has the form:
                            < i | O | j >

     The dimension of the matrix must be equal to the total dimension
     of the basis. Each matrix element is a sum over matrix elements of
     all multi-shell tensors.

     How to get the index of a full state (not just irrep)?
     Recall that a state is labeled by
     | gamma, omega, kappa, L, zeta, S, T, MT, J, MJ >
     Here
     - J: fixed by ManyNucleonBasis
     - MJ: zero, fixed implicitly by us
     - MT: (N - Z)/2, fixed by ManyNucleonBasis
     The quantum numbers (kappamax,L,zetamax,S,T) are stored in an array.
     For a specific irrepID, we get:
     - irrep_offset
     - multiplicities of gamma and omega: stored with qcx_utils::NDIndex
     - a list of (kappamax,LL,zetamax,SS,TT) quantum numbers
     Practically, the degeneracy of each irrep can be captured in a 4D array:
     1st dim: gamma & omega multiplicities
     2nd dim: array storing (L,S,T)
     3rd dim: kappa
     4th dim: zeta
     The index of a state is
     offset + ((1st * max(2nd) + 2nd) * max(3rd) + 3rd) * max(4th) + 4th
  */
  Eigen::SparseMatrix<double>
  MatrixElements(std::vector<U3xU4::MultiShellTensor_phys>& tensors,
                 U3xU4::ManyNucleonBasis& basis, sam::Databases& dbs)
  {
  	// for (auto tensor : tensors)
  	//   std::cout << tensor << std::endl;

    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    auto irreps = basis.irreps();
    auto labels = basis.decomposition();
    auto JJ = basis.getJJ();
    int16_t MMJ = (JJ % 2 == 0) ? 0 : 1;
    auto MMT = basis.getMMT();

    auto tensors_count = tensors.size();
    auto JJ_opt = tensors[0].JJ;
    auto MMJ_opt = tensors[0].MMJ;
    auto TT_opt = tensors[0].TT;
    auto MMT_opt = tensors[0].MMT;

    // std::cout << "JJ_opt: " << JJ_opt
    //  	  << " MMJ_opt: " << MMJ_opt
    //		  << " TT_opt: " << TT_opt
    //		  << " MMT_opt: " << MMT_opt << std::endl;

    std::vector<Eigen::Triplet<double>> ija;

    auto cgcJ = cgcs.compute(JJ, JJ_opt, JJ, MMJ, MMJ_opt, MMJ);

    if (std::fabs(cgcJ) > 0.0)
    {
      for (size_t bra_id = 0; bra_id < irreps.size(); bra_id++)
      {
        // std::cout << "bra_id: " << bra_id << std::endl;
        // std::cout << "Construct bra_multis" << std::endl;
        qcx_utils::NDIndex bra_multis;
        for (const auto& gamma : irreps[bra_id].gamma)
          bra_multis.add_dimension(gamma.alphamax());
        for (const auto& omega : irreps[bra_id].omega)
          bra_multis.add_dimensions(omega.rhomax(), omega.etamax());

        // get offset
        auto bra_offset = irreps.offset(bra_id);

        for (size_t ket_id = 0; ket_id < irreps.size(); ket_id++)
        {
          // std::cout << "ket_id: " << ket_id << std::endl;
          // std::cout << "Construct ket_multis" << std::endl;
          qcx_utils::NDIndex ket_multis;
          for (const auto& gamma : irreps[ket_id].gamma)
            ket_multis.add_dimension(gamma.alphamax());
          for (const auto& omega : irreps[ket_id].omega)
            ket_multis.add_dimensions(omega.rhomax(), omega.etamax());

          // get offset
          auto ket_offset = irreps.offset(ket_id);

          // unify bra and ket irreps
          U3xU4::IrrepUnifier iu;
          const auto& [ubra, uket] = iu.unify(irreps[bra_id], irreps[ket_id]);
          // extract the last irreps
          u3::irrep ket_u3, bra_u3;
          u4::irrep ket_u4, bra_u4;
          if (ubra.omega.empty())
          {
            ket_u3 = uket.gamma[0].u3();
            bra_u3 = ubra.gamma[0].u3();

            ket_u4 = uket.gamma[0].u4();
            bra_u4 = ubra.gamma[0].u4();
          }
          else
          {
            ket_u3 = uket.omega.back().u3();
            bra_u3 = ubra.omega.back().u3();

            ket_u4 = uket.omega.back().u4();
            bra_u4 = ubra.omega.back().u4();
          }
          // std::cout << "ket_u3: " << ket_u3
          //           << " bra_u3: " << bra_u3
          //           << " ket_u4: " << ket_u4
          //           << " bra_u4: " << bra_u4 << std::endl;

          // calculate RMEs
          // std::cout << "Start to calculate RMEs" << std::endl;
          std::vector<std::vector<double>> RMEs;
          RMEs.reserve(tensors_count);
          for (auto tensor : tensors)
          {
            // std::cout << tensor << std::endl;
            // std::cout << tensor.T << std::endl;

            // unify the tensor with the irreps
            auto uT = U3xU4::unify_tensor(tensor.T, ubra);

            if (uT.has_value())
            {
              // std::cout << "After unified: " << std::endl;
              // for (auto uG : uT->Gamma)
              //   std::cout << uG << std::endl;
              // for (auto uO : uT->Omega)
              //   std::cout << uO << std::endl;

              auto rmes = uT->RME(uket, ubra, dbs);
              // for (const auto& rme : rmes)
              //   std::cout << rme << std::endl;
              RMEs.push_back(rmes);
            }
            else
            {
              RMEs.push_back({});
            }
          }
          // std::cout << "Finish calculating RMEs" << std::endl;

          // now, we compute full matrix elements
          // std::cout << "Start to calculate FMEs" << std::endl;
          size_t bmax = labels[bra_id].size(), kmax = labels[ket_id].size();
          for (size_t b = 0; b < bmax; b++) // going through bra labels
          {
            auto bra_kappamax = labels[bra_id][b][0],
                 bra_LL = labels[bra_id][b][1],
                 bra_zetamax = labels[bra_id][b][2],
                 bra_SS = labels[bra_id][b][3], bra_TT = labels[bra_id][b][4];

            for (size_t k = 0; k < kmax; k++) // going through ket labels
            {
              auto ket_kappamax = labels[ket_id][k][0],
                   ket_LL = labels[ket_id][k][1],
                   ket_zetamax = labels[ket_id][k][2],
                   ket_SS = labels[ket_id][k][3], ket_TT = labels[ket_id][k][4];

              // std::cout << "TT,MMT: " << ket_TT << "," << TT_opt << "," << bra_TT
              //		<< "," << MMT << "," << MMT_opt << "," << MMT << std::endl;
              auto cgcT =
                  cgcs.compute(ket_TT, TT_opt, bra_TT, MMT, MMT_opt, MMT);
              // std::cout << "cgcT: " << cgcT << std::endl;
              if (std::fabs(cgcT) == 0.0)
                continue;

              std::vector<std::vector<Eigen::MatrixXd>> Coeffs3(tensors_count),
                  										Coeffs4(tensors_count);
              std::vector<std::vector<double>> U9js(tensors_count);
              std::vector<std::pair<int, int>> rhomaxs_etamaxs(tensors_count);

              // std::cout << "Start to calculate coupling coeffs" << std::endl;
              for (size_t tensor_id = 0; tensor_id < tensors_count; tensor_id++)
              {
              	// std::cout << "Tensor: " << tensor << std::endl;

                u3::irrep tensor_u3;
                u4::irrep tensor_u4;
                if (tensors[tensor_id].T.Omega.empty())
                {
                  tensor_u3 = tensors[tensor_id].T.Gamma[0].u3();
                  tensor_u4 = tensors[tensor_id].T.Gamma[0].u4();
                }
                else
                {
                  tensor_u3 = tensors[tensor_id].T.Omega.back().u3();
                  tensor_u4 = tensors[tensor_id].T.Omega.back().u4();
                }

                auto rhomax = u3::Rho(ket_u3, tensor_u3, bra_u3),
                     etamax = u4::Eta(ket_u4, tensor_u4, bra_u4);

                if (rhomax == 0 || etamax == 0)
                  continue;

                rhomaxs_etamaxs[tensor_id].first = rhomax;
                rhomaxs_etamaxs[tensor_id].second = etamax;
                
                for (auto LLSS : tensors[tensor_id].LLSSs)
                {
                  U9js[tensor_id].push_back(
                      racah9.compute(ket_LL, ket_SS, JJ, LLSS[1], LLSS[3],
                                     JJ_opt, bra_LL, bra_SS, JJ));

                  Coeffs3[tensor_id].push_back(
                      wigU3SO3db
                          .get(ket_u3.lam(), ket_u3.mu(), ket_LL / 2,
                               tensor_u3.lam(), tensor_u3.mu(), LLSS[1] / 2,
                               bra_u3.lam(), bra_u3.mu(), bra_LL / 2)
                          .second);

                  Coeffs4[tensor_id].push_back(
                      wigU4SU2SU2db
                          .get(ket_u4.P1(), ket_u4.P2(), ket_u4.P3(), ket_SS,
                               ket_TT, tensor_u4.P1(), tensor_u4.P2(),
                               tensor_u4.P3(), LLSS[3], TT_opt,
                               bra_u4.P1(), bra_u4.P2(), bra_u4.P3(), bra_SS,
                               bra_TT)
                          .second);
                }
              }
              // std::cout << "Finish calculating coupling coeffs" << std::endl;

              for (size_t bra_multi_id = 0;
                   bra_multi_id < bra_multis.cardinality(); bra_multi_id++)
              {
                std::vector<size_t> bra_multi(bra_multis.n_dimensions());
                bra_multis.to_multi(bra_multi_id, bra_multi);

                for (size_t bra_kappa = 0; bra_kappa < bra_kappamax;
                     bra_kappa++)
                {
                  for (size_t bra_zeta = 0; bra_zeta < bra_zetamax; bra_zeta++)
                  {
                    // bra index in the matrix
                    size_t i =
                        bra_offset +
                        ((bra_multi_id * bmax + b) * bra_kappamax + bra_kappa) *
                            bra_zetamax + bra_zeta;

                    for (size_t ket_multi_id = 0;
                         ket_multi_id < ket_multis.cardinality();
                         ket_multi_id++)
                    {
                      std::vector<size_t> ket_multi(ket_multis.n_dimensions());
                      ket_multis.to_multi(ket_multi_id, ket_multi);

                      for (size_t ket_kappa = 0; ket_kappa < ket_kappamax;
                           ket_kappa++)
                      {
                        for (size_t ket_zeta = 0; ket_zeta < ket_zetamax;
                             ket_zeta++)
                        {
                          // ket index in the matrix
                          size_t j = ket_offset +
                                     ((ket_multi_id * kmax + k) * ket_kappamax +
                                      ket_kappa) * ket_zetamax + ket_zeta;

                          double me = 0.0;
                          // std::cout << "i:" << i << " j:" << j << std::endl;

                          for (size_t tensor_id = 0; tensor_id < tensors_count;
                               tensor_id++)
                          {
                          	if (RMEs[tensor_id].empty())
                              continue;

                            auto [rhomax, etamax] = rhomaxs_etamaxs[tensor_id];
                            if (rhomax == 0 || etamax == 0)
                              continue;

                            // std::cout << "rhomax = " << rhomax 
                            // 		   << " etamax = " << etamax << std::endl;

                          	// std::cout << "Tensor: " << tensors[tensor_id] << std::endl;
                          	// for (const auto& rme : RMEs[tensor_id])
              				//   std::cout << "rme: " << rme << std::endl;

                          	// multi-dimensional array for obtaining RMEs
                            qcx_utils::NDIndex RME_multis;
                            // add multis from ket
                            // std::cout << "Construct RME_multis from ket" <<
                            // std::endl;
                            for (const auto& gamma : irreps[ket_id].gamma)
                              RME_multis.add_dimension(gamma.alphamax());
                            for (const auto& omega : irreps[ket_id].omega)
                              RME_multis.add_dimensions(omega.rhomax(),
                                                        omega.etamax());
                            // add multis from bra
                            // std::cout << "Construct RME_multis from bra" <<
                            // std::endl;
                            for (const auto& gamma : irreps[bra_id].gamma)
                              RME_multis.add_dimension(gamma.alphamax());
                            for (const auto& omega : irreps[bra_id].omega)
                              RME_multis.add_dimensions(omega.rhomax(),
                                                        omega.etamax());
                            // add multis from tensor
                            // std::cout << "Construct RME_multis from tensor"
                            // << std::endl;
                            for (const auto& Gamma : tensors[tensor_id].T.Gamma)
                              RME_multis.add_dimensions(Gamma.rho0max(),
                                                        Gamma.eta0max());
                            for (const auto& Omega : tensors[tensor_id].T.Omega)
                              RME_multis.add_dimensions(Omega.rhomax(),
                                                        Omega.etamax());
                            // add multis from Wigner-Eckart
                            // std::cout << "Construct RME_multis from WK" <<
                            // std::endl;
                            RME_multis.add_dimensions(rhomax, etamax);
                            std::vector<size_t> RME_multi(
                                RME_multis.n_dimensions());

                            size_t cnt = 0;
                            for (auto x : ket_multi)
                            {
                              RME_multi[cnt] = x;
                              cnt++;
                            }
                            for (auto x : bra_multi)
                            {
                              RME_multi[cnt] = x;
                              cnt++;
                            }
                            // std::cout << "cnt: " << cnt << std::endl;

                            // going through subgroup labels of the tensor
                            for (size_t ls_id = 0;
                                 ls_id < tensors[tensor_id].LLSSs.size();
                                 ls_id++)
                            {
                              double U9j = U9js[tensor_id][ls_id];
                              if (std::fabs(U9j) == 0.0)
                              	continue;

                              if (Coeffs3[tensor_id][ls_id].size() == 0)
                          	    continue;

                          	  if (Coeffs4[tensor_id][ls_id].size() == 0)
                          	  	continue;

                              auto kappa0max =
                                  tensors[tensor_id].LLSSs[ls_id][0];
                              auto zeta0max =
                                  tensors[tensor_id].LLSSs[ls_id][2];

                              // multi-dimensional array for obtaining strength
                              qcx_utils::NDIndex strength_multis;
                              for (const auto& Gamma :
                                   tensors[tensor_id].T.Gamma)
                                strength_multis.add_dimensions(Gamma.rho0max(),
                                                               Gamma.eta0max());
                              for (const auto& Omega :
                                   tensors[tensor_id].T.Omega)
                                strength_multis.add_dimensions(Omega.rhomax(),
                                                               Omega.etamax());
                              strength_multis.add_dimensions(kappa0max, zeta0max); 

                              std::vector<size_t> strength_multi(
                                  strength_multis.n_dimensions());

                              for (size_t s = 0;
                                   s < strength_multis.cardinality(); s++)
                              {
                              	double strength = 
                              		tensors[tensor_id].strengths[ls_id][s];
                                // std::cout << "strength: " << strength << std::endl;

                                // extract multiplicities
                              	strength_multis.to_multi(s, strength_multi);
                                size_t cnt2 = cnt, cnt3 = 0;
                                for (const auto& Gamma :
                                     tensors[tensor_id].T.Gamma)
                                {
                                  RME_multi[cnt2] = strength_multi[cnt3];
                                  RME_multi[cnt2 + 1] = strength_multi[cnt3 + 1];
                                  cnt2 += 2;
                                  cnt3 += 2;
                                }
                                for (const auto& Omega :
                                     tensors[tensor_id].T.Omega)
                                {
                                  RME_multi[cnt2] = strength_multi[cnt3];
                                  RME_multi[cnt2 + 1] = strength_multi[cnt3 + 1];
                                  cnt2 += 2;
                                  cnt3 += 2;
                                }
                                // std::cout << "cnt2: " << cnt2 
                                // << " cnt3: " << cnt3 << std::endl;
                                auto kappa0 = strength_multi[cnt3];
                                auto zeta0 = strength_multi[cnt3 + 1];

                                auto cgcU3_row = (ket_kappa * kappa0max + kappa0) 
                                			   * bra_kappamax + bra_kappa;
                                auto cgcU4_row = (ket_zeta * zeta0max + zeta0) 
                                			   * bra_zetamax + bra_zeta;

                                // going through multiplicities of 
                                // Wigner-Eckart theorem
                                for (size_t rho = 0; rho < rhomax; rho++)
                                {
                                  double cgcU3 = 
                                  	Coeffs3[tensor_id][ls_id](cgcU3_row,rho);
                                  // std::cout << "cgcU3: " << cgcU3 << std::endl;
                                  RME_multi[cnt2] = rho;

                                  for (size_t eta = 0; eta < etamax; eta++)
                                  {
                                  	double cgcU4 = 
                                      Coeffs4[tensor_id][ls_id](cgcU4_row,eta);
                                    // std::cout << "cgcU4: " << cgcU4 << std::endl;

                                  	RME_multi[cnt2 + 1] = eta;
                                  	// for (const auto& m : RME_multi)
                                  	//   std::cout << m << " ";
                                  	// std::cout << std::endl;

                                  	auto rme =
                                        RMEs[tensor_id]
                                            [RME_multis.to_linear(RME_multi)];
                                    // std::cout << "rme: " << rme << std::endl;

                                    me +=
                                        (strength * U9j * cgcU3 * cgcU4 * rme);
                                  }
                                }
                              }
                            }
                          }

                          me *= (cgcJ * cgcT);

                          if (std::fabs(me) > 1e-12)
                            ija.emplace_back(i, j, me);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          // std::cout << "Finish calculating FMEs" << std::endl;
        }
      }
    }

    Eigen::SparseMatrix<double> mat(basis.dim(), basis.dim());
    mat.setFromTriplets(ija.begin(), ija.end());

    mat.makeCompressed();
    return mat;
  }
} // namespace U3xU4
