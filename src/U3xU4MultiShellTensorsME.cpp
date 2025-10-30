#include <ostream>

#include <qcx_ND.h>

#include <U3xU4MultiShellTensors.h>

std::pair<double,double> sum_mean(const std::vector<double>& vs)
{
  double x = 0.0;
  for (const auto& v : vs)
    x += v;
  return std::make_pair(x,vs.empty() ? 0.0 : x / vs.size());
}

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
     - multiplicities of gamma and omega: alpha
     - a list of (kappamax,LL,zetamax,SS,TT,LST_offset) quantum numbers
     To access the index of a particular state, calculate the following
     LST_length = last(LST_offset) + last(kappamax) * last(zetamax)
     state_id = irrep_offset + alpha * LST_length 
              + LST_offset + kappa * zetamax + zeta
  */

//#define ANALYZE_ME_CONTRIBUTION
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
    auto dim = basis.dim();
    auto JJ = basis.getJJ();
    int16_t MMJ = (JJ % 2 == 0) ? 0 : 1;
    auto MMT = basis.getMMT();

    auto tensors_count = tensors.size();
    // assume that all tensors have the same J, MJ, MT, but T can vary
    auto JJ_opt = tensors[0].JJ;
    auto MMJ_opt = tensors[0].MMJ;
    //auto TT_opt = tensors[0].TT;
    auto MMT_opt = tensors[0].MMT;

    if (tensors_count == 0)
    {
      Eigen::SparseMatrix<double> mat(dim, dim);
      return mat;
    }

    // std::cout << "JJ_opt: " << JJ_opt
    //  	  << " MMJ_opt: " << MMJ_opt
    //		  << " TT_opt: " << TT_opt
    //		  << " MMT_opt: " << MMT_opt << std::endl;

    std::vector<Eigen::Triplet<double>> ija;

    auto cgcJ = cgcs.compute(JJ, JJ_opt, JJ, MMJ, MMJ_opt, MMJ);

#ifdef ANALYZE_ME_CONTRIBUTION
    std::vector<u3::irrep> U3s;
    std::vector<u4::irrep> U4s;
    std::vector<std::vector<double>> contribution_U3s, contribution_U4s;
#endif

    if (std::fabs(cgcJ) > 0.0)
    {
      for (size_t bra_id = 0; bra_id < irreps.size(); bra_id++)
      {
        // std::cout << "bra_id: " << bra_id << std::endl;

        // get alphamax for bra
        size_t bra_alphamax = 1;
        for (const auto& gamma : irreps[bra_id].gamma)
          bra_alphamax *= gamma.alphamax();
        for (const auto& omega : irreps[bra_id].omega)
          bra_alphamax *= (omega.rhomax() * omega.etamax());

        // get offset
        auto bra_offset = irreps.offset(bra_id);

        for (size_t ket_id = 0; ket_id < irreps.size(); ket_id++)
        {
          // std::cout << "ket_id: " << ket_id << std::endl;

          // get alphamax for bra
          size_t ket_alphamax = 1;
          for (const auto& gamma : irreps[ket_id].gamma)
            ket_alphamax *= gamma.alphamax();
          for (const auto& omega : irreps[ket_id].omega)
            ket_alphamax *= (omega.rhomax() * omega.etamax());

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
          std::vector<sam::RME> RMEs;
          RMEs.reserve(tensors_count);
          for (auto tensor : tensors)
          {
            // std::cout << tensor << std::endl;
            // std::cout << tensor.T << std::endl;

            // unify the tensor with the irreps
            auto uT = U3xU4::unify_tensor(tensor.T, ubra);

            if (uT.has_value())
            {
/*
              std::cout << "After unified: " << std::endl;
              for (auto uG : uT->Gamma)
                std::cout << uG << std::endl;
              for (auto uO : uT->Omega)
                std::cout << uO << std::endl; 
*/
              auto rmes = uT->RME(ubra, uket, dbs);
              // rmes.print();
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
          size_t bra_labels_max = labels[bra_id].size(),
                 bra_labels_dim = labels[bra_id].back()[5] +
                 labels[bra_id].back()[0] * labels[bra_id].back()[2],
                 ket_labels_max = labels[ket_id].size(),
                 ket_labels_dim = labels[ket_id].back()[5] +
                 labels[ket_id].back()[0] * labels[ket_id].back()[2];

          for (size_t bra_labels_id = 0; bra_labels_id < bra_labels_max;
               bra_labels_id++) // going through bra labels
          {
            auto bra_kappamax = labels[bra_id][bra_labels_id][0],
                 bra_LL = labels[bra_id][bra_labels_id][1],
                 bra_zetamax = labels[bra_id][bra_labels_id][2],
                 bra_SS = labels[bra_id][bra_labels_id][3], 
                 bra_TT = labels[bra_id][bra_labels_id][4],
                 bra_labels_offset = labels[bra_id][bra_labels_id][5];

            for (size_t ket_labels_id = 0; ket_labels_id < ket_labels_max;
                 ket_labels_id++) // going through ket labels
            {
              auto ket_kappamax = labels[ket_id][ket_labels_id][0],
                   ket_LL = labels[ket_id][ket_labels_id][1],
                   ket_zetamax = labels[ket_id][ket_labels_id][2],
                   ket_SS = labels[ket_id][ket_labels_id][3], 
                   ket_TT = labels[ket_id][ket_labels_id][4],
                   ket_labels_offset = labels[ket_id][ket_labels_id][5];

              // std::cout << "TT,MMT: " << ket_TT << "," << TT_opt << "," << bra_TT
              //		<< "," << MMT << "," << MMT_opt << "," << MMT << std::endl;
              // auto cgcT =
              //    cgcs.compute(ket_TT, TT_opt, bra_TT, MMT, MMT_opt, MMT);
              // std::cout << "cgcT: " << cgcT << std::endl;
              // if (std::fabs(cgcT) == 0.0)
              //   continue;

              std::vector<std::vector<Eigen::MatrixXd>> Coeffs3(tensors_count),
                  										Coeffs4(tensors_count);
              // std::vector<std::vector<double>> U9js(tensors_count);                        
              std::vector<std::vector<double>> cgcJ_cgcT_U9js(tensors_count);

              // std::cout << "Start to calculate coupling coeffs" << std::endl;
              for (size_t tensor_id = 0; tensor_id < tensors_count; tensor_id++)
              {
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
/*                
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
                               tensor_u4.P3(), LLSS[3], tensors[tensor_id].TT,//TT_opt,
                               bra_u4.P1(), bra_u4.P2(), bra_u4.P3(), bra_SS,
                               bra_TT)
                          .second);
                }
*/

                for (auto LLSSTT : tensors[tensor_id].LLSSTTs)
                {
                  auto cgcT = cgcs.compute(ket_TT, LLSSTT[4], bra_TT, MMT, MMT_opt, MMT);
                  //std::cout << "cgcT: " << cgcT << std::endl;

                  cgcJ_cgcT_U9js[tensor_id].push_back(cgcJ * cgcT *
                      racah9.compute(ket_LL, ket_SS, JJ, LLSSTT[1], LLSSTT[3],
                                     JJ_opt, bra_LL, bra_SS, JJ));

                  Coeffs3[tensor_id].push_back(
                      wigU3SO3db
                          .get(ket_u3.lam(), ket_u3.mu(), ket_LL / 2,
                               tensor_u3.lam(), tensor_u3.mu(), LLSSTT[1] / 2,
                               bra_u3.lam(), bra_u3.mu(), bra_LL / 2)
                          .second);

                  Coeffs4[tensor_id].push_back(
                      wigU4SU2SU2db
                          .get(ket_u4.P1(), ket_u4.P2(), ket_u4.P3(), ket_SS,
                               ket_TT, tensor_u4.P1(), tensor_u4.P2(),
                               tensor_u4.P3(), LLSSTT[3], LLSSTT[4],
                               bra_u4.P1(), bra_u4.P2(), bra_u4.P3(), bra_SS,
                               bra_TT)
                          .second);
                }
              }
              // std::cout << "Finish calculating coupling coeffs" << std::endl;

              for (size_t bra_alpha = 0;
                   bra_alpha < bra_alphamax; bra_alpha++)
              {
                for (size_t bra_kappa = 0; bra_kappa < bra_kappamax;
                     bra_kappa++)
                {
                  for (size_t bra_zeta = 0; bra_zeta < bra_zetamax; bra_zeta++)
                  {
                    // bra index in the matrix
                    size_t i = bra_offset + bra_alpha * bra_labels_dim +
                      bra_labels_offset + (bra_kappa * bra_zetamax + bra_zeta);

                    for (size_t ket_alpha = 0;
                         ket_alpha < ket_alphamax; ket_alpha++)
                    {
                      for (size_t ket_kappa = 0; ket_kappa < ket_kappamax;
                           ket_kappa++)
                      {
                        for (size_t ket_zeta = 0; ket_zeta < ket_zetamax;
                             ket_zeta++)
                        {
                          // ket index in the matrix
                          size_t j = ket_offset + ket_alpha * ket_labels_dim +
                            ket_labels_offset + (ket_kappa * ket_zetamax + ket_zeta);

                          double me = 0.0;
                          // std::cout << "i:" << i << " j:" << j << std::endl;

                          for (size_t tensor_id = 0; tensor_id < tensors_count;
                               tensor_id++)
                          {
                          	if (RMEs[tensor_id].empty())
                              continue;

                            auto rhomax = RMEs[tensor_id].rhomax(), 
                                 etamax = RMEs[tensor_id].etamax();
                            if (rhomax == 0 || etamax == 0)
                              continue;

                            // std::cout << "rhomax = " << rhomax 
                            //       << " etamax = " << etamax << std::endl;

                            //std::cout << "Tensor: " << tensors[tensor_id] << std::endl;
                            //RMEs[tensor_id].print();
                        
                            // going through subgroup labels of the tensor
                            for (size_t lst_id = 0;
                                 lst_id < tensors[tensor_id].LLSSTTs.size();
                                 lst_id++)
                            {
                              double cgcJ_cgcT_U9j = cgcJ_cgcT_U9js[tensor_id][lst_id];
                              if (std::fabs(cgcJ_cgcT_U9j) == 0.0)
                              	continue;

                              if (Coeffs3[tensor_id][lst_id].size() == 0)
                          	    continue;

                          	  if (Coeffs4[tensor_id][lst_id].size() == 0)
                          	  	continue;

                              auto kappa0max =
                                  tensors[tensor_id].LLSSTTs[lst_id][0];
                              auto zeta0max =
                                  tensors[tensor_id].LLSSTTs[lst_id][2];

                              size_t tensor_alphamax = 1;
                              for (const auto& Gamma :
                                   tensors[tensor_id].T.Gamma)
                                tensor_alphamax *= 
                                  (Gamma.rho0max() * Gamma.eta0max());
                              for (const auto& Omega :
                                   tensors[tensor_id].T.Omega)
                                tensor_alphamax *= 
                                  (Omega.rhomax() * Omega.etamax());

                              for (size_t tensor_alpha = 0; 
                                   tensor_alpha < tensor_alphamax; tensor_alpha++)
                              {
                                for (size_t kappa0 = 0; kappa0 < kappa0max; kappa0++)
                                {
                                  auto cgcU3_row = (ket_kappa * kappa0max + kappa0) 
                                           * bra_kappamax + bra_kappa;

                                  for (size_t zeta0 = 0; zeta0 < zeta0max; zeta0++)
                                  {
                                    double strength = tensors[tensor_id]
                                      .strengths[lst_id][(tensor_alpha * kappa0max 
                                                        + kappa0) * zeta0max + zeta0];
                                    //std::cout << "strength: " << strength << std::endl;

                                    auto cgcU4_row = (ket_zeta * zeta0max + zeta0) 
                                           * bra_zetamax + bra_zeta;

                                    // going through multiplicities of 
                                    // Wigner-Eckart theorem
                                    for (size_t rho = 0; rho < rhomax; rho++)
                                    {
                                      double cgcU3 = 
                                        Coeffs3[tensor_id][lst_id](cgcU3_row,rho);
                                      //std::cout << "cgcU3: " << cgcU3 << std::endl;

                                      for (size_t eta = 0; eta < etamax; eta++)
                                      {
                                        double cgcU4 = 
                                          Coeffs4[tensor_id][lst_id](cgcU4_row,eta);
                                        //std::cout << "cgcU4: " << cgcU4 << std::endl;

                                  
                                        auto rme = RMEs[tensor_id].get(
                                          bra_alpha, ket_alpha, tensor_alpha, rho, eta);
                                        //std::cout << "rme: " << rme << std::endl;

                                        double x = strength * cgcJ_cgcT_U9j * cgcU3 * cgcU4 * rme;
#ifdef ANALYZE_ME_CONTRIBUTION
                                        // check if the tensor character contributes to me
                                        if (std::fabs(x) > 1e-12)
                                        {
                                          //std::cout << tensors[tensor_id].getLastIrrep() << std::endl;
                                          auto U3 = tensors[tensor_id].getLastIrrep().u3(); 
                                          auto U4 = tensors[tensor_id].getLastIrrep().u4();

                                          auto it3 = std::find(U3s.begin(), U3s.end(), U3);
                                          if (it3 != U3s.end())
                                          {
                                            size_t index = std::distance(U3s.begin(), it3);
                                            contribution_U3s[index][i * dim + j] += x;
                                          }
                                          else
                                          {
                                            U3s.push_back(U3);
                                            contribution_U3s.push_back(std::vector<double>(dim * dim, 0.0));
                                            contribution_U3s.back()[i * dim + j] += x;
                                          }

                                          auto it4 = std::find(U4s.begin(), U4s.end(), U4);
                                          if (it4 != U4s.end())
                                          {
                                            size_t index = std::distance(U4s.begin(), it4);
                                            contribution_U4s[index][i * dim + j] += x;
                                          }
                                          else
                                          {
                                            U4s.push_back(U4);
                                            contribution_U4s.push_back(std::vector<double>(dim * dim, 0.0));
                                            contribution_U4s.back()[i * dim + j] += x;
                                          }
                                        }
#endif
                                        me += x;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }

                          if (std::fabs(me) > 1e-12)
                          {
                            ija.emplace_back(i, j, me);
/*                            
                            if (ket_u3 != bra_u3)
                              std::cout << "Connect different irreps:" 
                                        << " ket: " << ket_id << ket_u3 
                                        << " bra: " << bra_id << bra_u3 << std::endl;
                            if (ket_u4 != bra_u4)
                              std::cout << "Connect different irreps:" 
                                        << " ket: " << ket_id << ket_u4 
                                        << " bra: " << bra_id << bra_u4 << std::endl;
*/
                          }
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

#ifdef ANALYZE_ME_CONTRIBUTION
    std::cout << "Contributions to matrix elements for U(3)" << std::endl;
    for (size_t i = 0; i < U3s.size(); i++)
    {
      auto sm = sum_mean(contribution_U3s[i]);
      std::cout << U3s[i] << ": sum = " << sm.first << " mean = " << sm.second << std::endl;
    }

    std::cout << "Contributions to matrix elements for U(4)" << std::endl;
    for (size_t i = 0; i < U4s.size(); i++)
    {
      auto sm = sum_mean(contribution_U4s[i]);
      std::cout << U4s[i] << ": sum = " << sm.first << " mean = " << sm.second << std::endl;
    }
#endif

    Eigen::SparseMatrix<double> mat(dim, dim);
    mat.setFromTriplets(ija.begin(), ija.end());
    mat.makeCompressed();

    return mat;
  }
} // namespace U3xU4
