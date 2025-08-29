#include <ostream>

#include <qcx_ND.h>

#include <U3xU4MultiShellTensors.h>

namespace U3xU4
{
  /* Goal: Compute matrix elements of a JT-coupled operator O, which is 
  	 expanded in terms of many U3xU4::MultiShellTensor_phys objects.
  	 These tensors as input are not unified with the basis states;
  	 the necessity for unification only arises when calculating 
  	 multi-shell RMEs. The matrix element has the form:
      < i | O | j >
     The dimension of the matrix must be equal to the total dimension
     of the basis. Each matrix element is a sum over matrix elements of
     all multi-shell tensors.
     How to get the index of a state?
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
    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    auto irreps = basis.irreps();
    //auto labels = basis.decomposition();
    //auto jj = basis.getJJ();
    //auto mmt = basis.getMMT();

    auto tensors_count = tensors.size();

    std::vector<Eigen::Triplet<double>> ija;

    for (size_t bra_id = 0; bra_id < irreps.size(); bra_id++)
    {
      //std::cout << "bra_id: " << bra_id << std::endl;
      //std::cout << "Construct bra_multis" << std::endl;
      qcx_utils::NDIndex bra_multis;
      for (const auto& gamma : irreps[bra_id].gamma) 
        bra_multis.add_dimension(gamma.alphamax());
      for (const auto& omega : irreps[bra_id].omega) 
        bra_multis.add_dimensions(omega.rhomax(), omega.etamax());

      // get offset
      auto bra_offset = irreps.offset(bra_id);

      for (size_t ket_id = 0; ket_id < irreps.size(); ket_id++)
      {
        //std::cout << "ket_id: " << ket_id << std::endl;
        //std::cout << "Construct ket_multis" << std::endl;
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

        // calculate RMEs
        //std::cout << "Start to calculate RMEs" << std::endl;
        std::vector<std::vector<double>> RMEs;
        RMEs.reserve(tensors_count);
        for (auto tensor : tensors)
        {
          //std::cout << tensor << std::endl;

          // Check if there is any SingleShellTensor sitting outside of the states
          bool flag = false;
          for (auto G : tensor.T.Gamma)
          {
            //std::cout << "Before unified: " << G << std::endl;
            bool flag2 = false; // check if this tensor belong to any shell
            for (auto gam : ubra.gamma)
              if (gam.shell() == G.shell()) // find the shell
              {
              	flag2 = true;
              	break;
              }

            // if this tensor doesn't belong to any shell
            if (flag2 == false)
            {
              flag = true;
              break;
            }
          }
          if (flag == true)
            RMEs.push_back({});
          else
          {
            // unify the tensor with the irreps
            // auto uT = U3xU4::unify_tensor(tensor.T, ubra);

            //for (auto uG : uT.Gamma)
            //  std::cout << "After unified: " << uG << std::endl;

          	// temporarily using tensor.T
            auto rmes = tensor.T.RME(uket, ubra, dbs);
            RMEs.push_back(rmes);
          }
        }
        //std::cout << "Finish calculating RMEs" << std::endl;
/*
        // now, we compute full matrix elements
        //std::cout << "Start to calculate FMEs" << std::endl;
        size_t bmax = labels[bra_id].size(), kmax = labels[ket_id].size();
        for (size_t b = 0; b < bmax; b++)
        {
          auto bra_kappamax = labels[bra_id][b][0],
               bra_LL = labels[bra_id][b][1],
               bra_zetamax = labels[bra_id][b][2],
               bra_SS = labels[bra_id][b][3],
               bra_TT = labels[bra_id][b][4];

          for (size_t k = 0; k < kmax; k++)
          {
            auto ket_kappamax = labels[ket_id][k][0],
                 ket_LL = labels[ket_id][k][1],
                 ket_zetamax = labels[ket_id][k][2],
                 ket_SS = labels[ket_id][k][3],
                 ket_TT = labels[ket_id][k][4];

            auto cgcJ = cgcs.compute(jj,JJ,jj,0,MMJ,0),
                 cgcT = cgcs.compute(ket_TT,TT,bra_TT,mmt,MMT,mmt);
            //std::cout << "cgcJ, cgcT: " << cgcJ << ", " << cgcT << std::endl;
            if (cgcJ == 0.0 || cgcT == 0.0)
              continue;
            
            su3::irrep ket_su3, bra_su3;
            u4::irrep ket_u4, bra_u4;
            if (ubra.omega.empty())
            {
              ket_su3 = uket.gamma[0].su3();
              bra_su3 = ubra.gamma[0].su3();

              ket_u4 = uket.gamma[0].u4();
              bra_u4 = ubra.gamma[0].u4();
            }
            else
            {
              ket_su3 = uket.omega.back().su3();
              bra_su3 = ubra.omega.back().su3();

              ket_u4 = uket.omega.back().u4();
              bra_u4 = ubra.omega.back().u4();
            }
            //std::cout << "ket_su3: " << ket_su3 
            //          << " bra_su3: " << bra_su3
            //          << " ket_u4: " << ket_u4
            //          << " bra_u4: " << bra_u4 << std::endl;
            
            std::vector<Eigen::MatrixXd> Coeffs3, Coeffs4;
            Coeffs3.reserve(tensors_count);
            Coeffs4.reserve(tensors_count);
            std::vector<double> U9js;
            U9js.reserve(tensors_count);

            //std::cout << "Start to calculate coupling coeffs" << std::endl;
            for (auto tensor : tensors) 
            { 
              //std::cout << "Tensor: " << tensor << std::endl;

              U9js.push_back(racah9.compute(ket_LL, ket_SS, jj,
                                            tensor.LL, tensor.SS, JJ,
                                            bra_LL, bra_SS, jj));
              //std::cout << "Calculated U9j" << std::endl;

              Coeffs3.push_back(wigU3SO3db.get(ket_su3.lam(),ket_su3.mu(),ket_LL/2,
                                        tensor.t.lam(), tensor.t.mu(), tensor.LL/2,
                                       bra_su3.lam(),bra_su3.mu(),bra_LL/2).second);
              //std::cout << "Calculated wigU3SO3" << std::endl;

              //std::cout << ket_u4.P1() << " " << ket_u4.P2() << " " << ket_u4.P3() << "; "
              //          << tensor.t.P1() << " " << tensor.t.P2() << " " << tensor.t.P3() << "; "
              //          << bra_u4.P1() << " " << bra_u4.P2() << " " << bra_u4.P3() << std::endl;

              Coeffs4.push_back(wigU4SU2SU2db.get(
                ket_u4.P1(), ket_u4.P2(), ket_u4.P3(), ket_SS, ket_TT,
                tensor.t.P1(), tensor.t.P2(), tensor.t.P3(), tensor.SS, tensor.TT,
                bra_u4.P1(), bra_u4.P2(), bra_u4.P3(), bra_SS, bra_TT).second);
              //std::cout << "Calculated wigU4SU2SU2" << std::endl;  
            }
            //std::cout << "Finish calculating coupling coeffs" << std::endl;

            for (size_t bra_multi_id = 0; bra_multi_id < bra_multis.cardinality(); bra_multi_id++)
            {
              std::vector<size_t> bra_multi(bra_multis.n_dimensions());
              bra_multis.to_multi(bra_multi_id, bra_multi);

              for (size_t bra_kappa = 0; bra_kappa < bra_kappamax; bra_kappa++)
              {
                for (size_t bra_zeta = 0; bra_zeta < bra_zetamax; bra_zeta++)
                {
                  size_t i = bra_offset + ((bra_multi_id * bmax + b) * bra_kappamax + bra_kappa) * bra_zetamax + bra_zeta;

                  for (size_t ket_multi_id = 0; ket_multi_id < ket_multis.cardinality(); ket_multi_id++)
                  {
                    std::vector<size_t> ket_multi(ket_multis.n_dimensions());
                    ket_multis.to_multi(ket_multi_id, ket_multi);

                    for (size_t ket_kappa = 0; ket_kappa < ket_kappamax; ket_kappa++)
                    {
                      for (size_t ket_zeta = 0; ket_zeta < ket_zetamax; ket_zeta++)
                      {
                        size_t j = ket_offset + ((ket_multi_id * kmax + k) * ket_kappamax + ket_kappa) * ket_zetamax + ket_zeta;

                        double me = 0.0;

                        for (size_t tensor_id = 0; tensor_id < tensors.size(); tensor_id++)
                        { 
                          if (RMEs[tensor_id].empty())
                            continue;

                          size_t rhomax = Coeffs3[tensor_id].cols(),
                                 etamax = Coeffs4[tensor_id].cols();
                          if (rhomax == 0 || etamax == 0)
                            continue;

                          //std::cout << "rhomax = " << rhomax << " etamax = " << etamax << std::endl;

                          qcx_utils::NDIndex RME_multis;
                          // add multis from ket
                          //std::cout << "Construct RME_multis from ket" << std::endl;
                          for (const auto& gamma : irreps[ket_id].gamma) 
                            RME_multis.add_dimension(gamma.alphamax());
                          for (const auto& omega : irreps[ket_id].omega) 
                            RME_multis.add_dimensions(omega.rhomax(), omega.etamax());
                          // add multis from bra
                          //std::cout << "Construct RME_multis from bra" << std::endl;
                          for (const auto& gamma : irreps[bra_id].gamma) 
                            RME_multis.add_dimension(gamma.alphamax());
                          for (const auto& omega : irreps[bra_id].omega) 
                            RME_multis.add_dimensions(omega.rhomax(), omega.etamax());
                          // add multis from Wigner-Eckart
                          //std::cout << "Construct RME_multis from WK" << std::endl;
                          RME_multis.add_dimensions(rhomax,etamax);
                          std::vector<size_t> RME_multi(RME_multis.n_dimensions());

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

                          double me_temp = 0.0;

                          // One-body multishell tensor is only degenerate in kappa
                          size_t kappa0max = tensors[tensor_id].kappamax;
                          for (size_t kappa0 = 0; kappa0 < kappa0max; kappa0++)
                          {
                            double strength = tensors[tensor_id].coeff(kappa0);
                            //std::cout << "strength: " << strength << std::endl;
                            
                            for (size_t rho = 0; rho < rhomax; rho++)
                            {
                              double cgcU3 = Coeffs3[tensor_id]((ket_kappa * kappa0max + kappa0) * bra_kappamax + bra_kappa,rho);
                              //std::cout << "cgcU3: " << cgcU3 << std::endl;

                              for (size_t eta = 0; eta < etamax; eta++)
                              {
                                double cgcU4 = Coeffs4[tensor_id](ket_zeta * bra_zetamax + bra_zeta,eta);
                                //std::cout << "cgcU4: " << cgcU4 << std::endl;

                                // extract RME, fortunately, we have only multiplicities
                                // from bra and ket irreps, and Wigner-Eckart theorem
                                RME_multi[cnt] = rho;
                                RME_multi[cnt + 1] = eta;
                                auto rme = RMEs[tensor_id][RME_multis.to_linear(RME_multi)];

                                //std::cout << "rme: " << rme << std::endl;

                                me_temp += (strength * cgcU3 * cgcU4 * rme);
                              }
                            }
                          }

                          //std::cout << "U9j: " << U9js[tensor_id] << std::endl;
                          me += (U9js[tensor_id] * me_temp);
                        }

                        me *= (cgcJ * cgcT);

                        if (std::fabs(me) > 1e-12)
                          ija.emplace_back(i,j,me);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        //std::cout << "Finish calculating FMEs" << std::endl;
*/
      }
    }

    Eigen::SparseMatrix<double> mat(basis.dim(), basis.dim());
    mat.setFromTriplets(ija.begin(), ija.end());

    return mat;
  }
} // namespace U3xU4
