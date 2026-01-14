#include <ostream>

#include <qcx_ND.h>

#include <U3xU4MultiShellTensors.h>

static std::pair<double,double> sum_mean(const std::vector<double>& vs)
{
  double x = 0.0;
  for (const auto& v : vs)
    x += v;
  return std::make_pair(x, vs.empty() ? 0.0 : x / vs.size());
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

  // Symmetric matrix
  Eigen::SparseMatrix<double> MatrixElements(const MultiShellTensors_phys& tensors, 
    const U3xU4::ManyNucleonBasis& basis, sam::Databases& dbs)
  {
  	// for (auto tensor : tensors) std::cout << tensor << std::endl;

    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    // basis info
    auto irreps = basis.irreps();
    auto dim = basis.dim();
    auto labels = basis.decomposition();
    auto JJ = basis.JJ();
    auto MMJ = JJ;
    auto MMT = basis.MMT();

    // tensors info
    auto tensors_count = tensors.size();
    // all tensors have the same J, MJ, MT, but T may vary
    auto JJ_opt = tensors[0].JJ();
    auto MMJ_opt = tensors[0].MMJ();
    auto MMT_opt = tensors[0].MMT();

    if (tensors_count == 0)
    {
      Eigen::SparseMatrix<double> mat(dim, dim);
      return mat;
    }

    std::vector<Eigen::Triplet<double>> ija;

    auto cgcJ = cgcs.get({JJ, JJ_opt, JJ, MMJ, MMJ_opt, MMJ});

    if (std::fabs(cgcJ) > 0.0)
    {
      for (size_t bra_id = 0; bra_id < irreps.size(); bra_id++)
      {
        // std::cout << "bra_id: " << bra_id << std::endl;

        const auto& bra = irreps[bra_id];

        // get alphamax
        size_t bra_alphamax = bra.alphamax();

        // get offset
        auto bra_offset = irreps.offset(bra_id);

        // get labels
        const auto& bra_labels = labels[bra_id];
        size_t bra_labels_dim = bra_labels.back()[5] +
                  bra_labels.back()[0] * bra_labels.back()[2];

        for (size_t ket_id = 0; ket_id < irreps.size(); ket_id++)
        {
          if (bra_id < ket_id) continue;

          // std::cout << "ket_id: " << ket_id << std::endl;

          const auto& ket = irreps[ket_id];

          // get alphamax 
          size_t ket_alphamax = ket.alphamax();

          // get offset
          auto ket_offset = irreps.offset(ket_id);

          // get labels
          const auto& ket_labels = labels[ket_id];
          size_t ket_labels_dim = ket_labels.back()[5] +
                  ket_labels.back()[0] * ket_labels.back()[2];


          // unify bra and ket irreps
          U3xU4::IrrepUnifier iu;
          const auto& [ubra, uket] = iu.unify(bra, ket);
          // extract the last irreps
          auto ket_irr = (ubra.omega.empty()) 
                       ? uket.gamma[0].u3u4() : uket.omega.back().u3u4();
          auto bra_irr = (ubra.omega.empty())
                       ? ubra.gamma[0].u3u4() : ubra.omega.back().u3u4();

          // calculate RMEs
          // std::cout << "Start to calculate RMEs" << std::endl;
          std::vector<sam::RME> RMEs;
          RMEs.reserve(tensors_count);
          for (const auto& tensor : tensors)
          {
            // std::cout << tensor.T();

            // unify the tensor with the irreps
            auto uT = U3xU4::unify_tensor(tensor.T(), ubra);

            if (uT.has_value())
            {
              /*
              std::cout << "\tAfter unified: " << std::endl;
              for (auto uG : uT->Gammas()) std::cout << uG << std::endl;
              for (auto uO : uT->Omegas()) std::cout << uO << std::endl; 
              */

              auto rmes = uT->RME(ubra, uket, dbs);
              // rmes.print();
              RMEs.push_back(std::move(rmes));
            }
            else
            {
              RMEs.push_back({});
              // std::cout << std::endl;
            }
          }
          // std::cout << "Finish calculating RMEs" << std::endl;

          // now, we compute full matrix elements
          // std::cout << "Start to calculate FMEs" << std::endl;

          // going through bra labels
          for (const auto& bra_label : bra_labels)
          {
            auto bra_kappamax = bra_label[0];
            auto bra_LL = bra_label[1];
            auto bra_zetamax = bra_label[2];
            auto bra_SS = bra_label[3];
            auto bra_TT = bra_label[4];
            auto bra_labels_offset = bra_label[5];

            // going through ket labels
            for (const auto& ket_label : ket_labels)
            {
              auto ket_kappamax = ket_label[0];
              auto ket_LL = ket_label[1];
              auto ket_zetamax = ket_label[2];
              auto ket_SS = ket_label[3];
              auto ket_TT = ket_label[4];
              auto ket_labels_offset = ket_label[5];

              std::vector<std::vector<double>> cgcU2s(tensors_count);

              std::vector<std::vector<Eigen::MatrixXd>> 
                cgcU3s(tensors_count), cgcU4s(tensors_count);

              // std::cout << "Start to calculate coupling coeffs" << std::endl;
              for (size_t tensor_id = 0; tensor_id < tensors_count; tensor_id++)
              {
                if (RMEs[tensor_id].empty()) continue;

                auto irr = tensors[tensor_id].getLastIrrep();

                for (const auto& [k, LL, z, SS, TT, s] : tensors[tensor_id].LLSSTTstrengths())
                {
                  auto cgcT = cgcs.get({ket_TT, TT, bra_TT, MMT, MMT_opt, MMT});
                  auto cgcU2 = cgcJ * cgcT * racah9.get({ket_LL, ket_SS, JJ,
                                                            LL, SS, JJ_opt,
                                                            bra_LL, bra_SS, JJ});
                  cgcU2s[tensor_id].push_back(cgcU2);

                  auto [allowU3, cgcU3] = wigU3SO3db.get(
                            ket_irr.lam(), ket_irr.mu(), ket_LL / 2,
                            irr.lam(), irr.mu(), LL / 2,
                            bra_irr.lam(), bra_irr.mu(), bra_LL / 2);
                  cgcU3s[tensor_id].push_back(std::move(cgcU3));

                  auto [allowU4, cgcU4] = wigU4SU2SU2db.get(
                      ket_irr.P1(), ket_irr.P2(), ket_irr.P3(), ket_SS, ket_TT, 
                      irr.P1(), irr.P2(), irr.P3(), SS, TT,
                      bra_irr.P1(), bra_irr.P2(), bra_irr.P3(), bra_SS, bra_TT);
                  cgcU4s[tensor_id].push_back(std::move(cgcU4));
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

                          if (i < j) continue;

                          double me = 0.0;
                          // std::cout << "i:" << i << " j:" << j << std::endl;

                          for (size_t tensor_id = 0; tensor_id < tensors_count;
                               tensor_id++)
                          {
                            const auto& tensor = tensors[tensor_id];
                            const auto& RME = RMEs[tensor_id];

                          	if (RME.empty()) continue;

                            auto rhomax = RME.rhomax(), etamax = RME.etamax();

                            // std::cout << "Tensor: " << tensor << std::endl;
                            // RME.print();
                        
                            // going through subgroup labels of the tensor
                            for (size_t lst_id = 0;
                                 lst_id < tensor.LLSSTTstrengths().size();
                                 lst_id++)
                            {
                              const auto& cgcU2 = cgcU2s[tensor_id][lst_id];
                              if (std::fabs(cgcU2) == 0.0) continue;
                              // std::cout << "cgcU2: " << cgcU2 << std::endl;

                              const auto& cgcU3 = cgcU3s[tensor_id][lst_id];
                              if (cgcU3.size() == 0) continue;
                              // std::cout << "cgcU3: " << cgcU3 << std::endl;

                              const auto& cgcU4 = cgcU4s[tensor_id][lst_id];
                          	  if (cgcU4.size() == 0) continue;
                              // std::cout << "cgcU4: " << cgcU4 << std::endl;

                              const auto& kappa0max =
                                std::get<0>(tensor.LLSSTTstrengths()[lst_id]);
                              const auto& zeta0max =
                                std::get<2>(tensor.LLSSTTstrengths()[lst_id]);
                              const auto& strengths =
                                std::get<5>(tensor.LLSSTTstrengths()[lst_id]);

                              size_t tensor_alphamax = tensor.alphamax();

                              for (size_t tensor_alpha = 0; 
                                   tensor_alpha < tensor_alphamax; tensor_alpha++)
                              {
                                for (size_t kappa0 = 0; kappa0 < kappa0max; kappa0++)
                                {
                                  auto cgcU3_row = (ket_kappa * kappa0max + kappa0) 
                                                 * bra_kappamax + bra_kappa;

                                  for (size_t zeta0 = 0; zeta0 < zeta0max; zeta0++)
                                  {
                                    auto cgcU4_row = (ket_zeta * zeta0max + zeta0) 
                                                   * bra_zetamax + bra_zeta;

                                    double strength = strengths[
                                      (tensor_alpha * kappa0max + kappa0) * zeta0max + zeta0];

                                    // going through multiplicities of 
                                    // Wigner-Eckart theorem
                                    for (size_t rho = 0; rho < rhomax; rho++)
                                    {
                                      for (size_t eta = 0; eta < etamax; eta++)
                                      {                                  
                                        auto rme = RME.get(
                                          bra_alpha, ket_alpha, tensor_alpha, rho, eta);
                                        //std::cout << "rme: " << rme << std::endl;

                                        double x = strength * cgcU2 * cgcU3(cgcU3_row,rho) 
                                                 * cgcU4(cgcU4_row,eta) * rme;

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
                            ija.emplace_back(i, j, me); // for i >= j
                            if (i > j) ija.emplace_back(j, i, me);
/*                            
                            if (ket_irr.u3() != bra_irr.u3())
                              std::cout << "Connect different irreps:" 
                                        << " ket: " << ket_id << ket_irr.u3() 
                                        << " bra: " << bra_id << bra_irr.u3() << std::endl;

                            if (ket_irr.u4() != bra_irr.u4())
                              std::cout << "Connect different irreps:" 
                                        << " ket: " << ket_id << ket_irr.u4() 
                                        << " bra: " << bra_id << bra_irr.u4() << std::endl;
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

    Eigen::SparseMatrix<double> mat(dim, dim);
    mat.setFromTriplets(ija.begin(), ija.end());
    mat.makeCompressed();

    return mat;
  }

  // Rectangular matrix
  Eigen::SparseMatrix<double> MatrixElements(const MultiShellTensors_phys& tensors, 
    const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets, 
    sam::Databases& dbs)
  {
    // for (auto tensor : tensors) std::cout << tensor << std::endl;

    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    // bra basis info
    auto bras_irreps = bras.irreps();
    auto bras_dim = bras.dim();
    auto bras_labels = bras.decomposition();
    auto bras_JJ = bras.JJ();
    auto bras_MMJ = bras_JJ;
    auto bras_MMT = bras.MMT();

    // ket basis info
    auto kets_irreps = kets.irreps();
    auto kets_dim = kets.dim();
    auto kets_labels = kets.decomposition();
    auto kets_JJ = kets.JJ();
    auto kets_MMJ = kets_JJ;
    auto kets_MMT = kets.MMT();

    // tensors info
    auto tensors_count = tensors.size();
    // all tensors have the same J, MJ, MT, but T may vary
    auto JJ_opt = tensors[0].JJ();
    auto MMJ_opt = tensors[0].MMJ();
    auto MMT_opt = tensors[0].MMT();

    if (tensors_count == 0)
    {
      Eigen::SparseMatrix<double> mat(bras_dim, kets_dim);
      return mat;
    }

    std::vector<Eigen::Triplet<double>> ija;

    auto cgcJ = cgcs.get({kets_JJ, JJ_opt, bras_JJ,
                             kets_MMJ, MMJ_opt, bras_MMJ});

    if (std::fabs(cgcJ) > 0.0)
    {
      for (size_t bra_id = 0; bra_id < bras_irreps.size(); bra_id++)
      {
        // std::cout << "bra_id: " << bra_id << std::endl;

        const auto& bra = bras_irreps[bra_id];

        // get alphamax
        size_t bra_alphamax = bra.alphamax();

        // get offset
        auto bra_offset = bras_irreps.offset(bra_id);

        // get labels
        const auto& bra_labels = bras_labels[bra_id];
        size_t bra_labels_dim = bra_labels.back()[5] +
                  bra_labels.back()[0] * bra_labels.back()[2];

        for (size_t ket_id = 0; ket_id < kets_irreps.size(); ket_id++)
        {
          // std::cout << "ket_id: " << ket_id << std::endl;

          const auto& ket = kets_irreps[ket_id];

          // get alphamax 
          size_t ket_alphamax = ket.alphamax();

          // get offset
          auto ket_offset = kets_irreps.offset(ket_id);

          // get labels
          const auto& ket_labels = kets_labels[ket_id];
          size_t ket_labels_dim = ket_labels.back()[5] +
                  ket_labels.back()[0] * ket_labels.back()[2];


          // unify bra and ket irreps
          U3xU4::IrrepUnifier iu;
          const auto& [ubra, uket] = iu.unify(bra, ket);
          // extract the last irreps
          auto ket_irr = (ubra.omega.empty()) 
                       ? uket.gamma[0].u3u4() : uket.omega.back().u3u4();
          auto bra_irr = (ubra.omega.empty())
                       ? ubra.gamma[0].u3u4() : ubra.omega.back().u3u4();

          // calculate RMEs
          // std::cout << "Start to calculate RMEs" << std::endl;
          std::vector<sam::RME> RMEs;
          RMEs.reserve(tensors_count);
          for (const auto& tensor : tensors)
          {
            // std::cout << tensor.T();

            // unify the tensor with the irreps
            auto uT = U3xU4::unify_tensor(tensor.T(), ubra);

            if (uT.has_value())
            {
              /*
              std::cout << "\tAfter unified: " << std::endl;
              for (auto uG : uT->Gammas()) std::cout << uG << std::endl;
              for (auto uO : uT->Omegas()) std::cout << uO << std::endl; 
              */

              auto rmes = uT->RME(ubra, uket, dbs);
              // rmes.print();
              RMEs.push_back(std::move(rmes));
            }
            else
            {
              RMEs.push_back({});
              // std::cout << std::endl;
            }
          }
          // std::cout << "Finish calculating RMEs" << std::endl;

          // now, we compute full matrix elements
          // std::cout << "Start to calculate FMEs" << std::endl;

          // going through bra labels
          for (const auto& bra_label : bra_labels)
          {
            auto bra_kappamax = bra_label[0];
            auto bra_LL = bra_label[1];
            auto bra_zetamax = bra_label[2];
            auto bra_SS = bra_label[3];
            auto bra_TT = bra_label[4];
            auto bra_labels_offset = bra_label[5];

            // going through ket labels
            for (const auto& ket_label : ket_labels)
            {
              auto ket_kappamax = ket_label[0];
              auto ket_LL = ket_label[1];
              auto ket_zetamax = ket_label[2];
              auto ket_SS = ket_label[3];
              auto ket_TT = ket_label[4];
              auto ket_labels_offset = ket_label[5];

              std::vector<std::vector<double>> cgcU2s(tensors_count);

              std::vector<std::vector<Eigen::MatrixXd>> 
                cgcU3s(tensors_count), cgcU4s(tensors_count);

              // std::cout << "Start to calculate coupling coeffs" << std::endl;
              for (size_t tensor_id = 0; tensor_id < tensors_count; tensor_id++)
              {
                if (RMEs[tensor_id].empty()) continue;

                auto irr = tensors[tensor_id].getLastIrrep();

                for (const auto& [k, LL, z, SS, TT, s] : tensors[tensor_id].LLSSTTstrengths())
                {
                  auto cgcT = cgcs.get({ket_TT, TT, bra_TT,
                                           kets_MMT, MMT_opt, bras_MMT});
                  auto cgcU2 = cgcJ * cgcT * racah9.get({ket_LL, ket_SS, kets_JJ,
                                                            LL, SS, JJ_opt,
                                                            bra_LL, bra_SS, bras_JJ});
                  cgcU2s[tensor_id].push_back(cgcU2);

                  auto [allowU3, cgcU3] = wigU3SO3db.get(
                            ket_irr.lam(), ket_irr.mu(), ket_LL / 2,
                            irr.lam(), irr.mu(), LL / 2,
                            bra_irr.lam(), bra_irr.mu(), bra_LL / 2);
                  cgcU3s[tensor_id].push_back(std::move(cgcU3));

                  auto [allowU4, cgcU4] = wigU4SU2SU2db.get(
                      ket_irr.P1(), ket_irr.P2(), ket_irr.P3(), ket_SS, ket_TT, 
                      irr.P1(), irr.P2(), irr.P3(), SS, TT,
                      bra_irr.P1(), bra_irr.P2(), bra_irr.P3(), bra_SS, bra_TT);
                  cgcU4s[tensor_id].push_back(std::move(cgcU4));
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
                            const auto& tensor = tensors[tensor_id];
                            const auto& RME = RMEs[tensor_id];

                            if (RME.empty()) continue;

                            auto rhomax = RME.rhomax(), etamax = RME.etamax();

                            // std::cout << "Tensor: " << tensors[tensor_id] << std::endl;
                            // RME.print();
                        
                            // going through subgroup labels of the tensor
                            for (size_t lst_id = 0;
                                 lst_id < tensor.LLSSTTstrengths().size();
                                 lst_id++)
                            {
                              const auto& cgcU2 = cgcU2s[tensor_id][lst_id];
                              if (std::fabs(cgcU2) == 0.0) continue;
                              // std::cout << "cgcU2: " << cgcU2 << std::endl;

                              const auto& cgcU3 = cgcU3s[tensor_id][lst_id];
                              if (cgcU3.size() == 0) continue;
                              // std::cout << "cgcU3: " << cgcU3 << std::endl;

                              const auto& cgcU4 = cgcU4s[tensor_id][lst_id];
                              if (cgcU4.size() == 0) continue;
                              // std::cout << "cgcU4: " << cgcU4 << std::endl;

                              const auto& kappa0max =
                                std::get<0>(tensor.LLSSTTstrengths()[lst_id]);
                              const auto& zeta0max =
                                std::get<2>(tensor.LLSSTTstrengths()[lst_id]);
                              const auto& strengths =
                                std::get<5>(tensor.LLSSTTstrengths()[lst_id]);

                              size_t tensor_alphamax = tensor.alphamax();

                              for (size_t tensor_alpha = 0; 
                                   tensor_alpha < tensor_alphamax; tensor_alpha++)
                              {
                                for (size_t kappa0 = 0; kappa0 < kappa0max; kappa0++)
                                {
                                  auto cgcU3_row = (ket_kappa * kappa0max + kappa0) 
                                                 * bra_kappamax + bra_kappa;

                                  for (size_t zeta0 = 0; zeta0 < zeta0max; zeta0++)
                                  {
                                    auto cgcU4_row = (ket_zeta * zeta0max + zeta0) 
                                                   * bra_zetamax + bra_zeta;

                                    double strength = strengths[
                                      (tensor_alpha * kappa0max + kappa0) * zeta0max + zeta0];

                                    // going through multiplicities of 
                                    // Wigner-Eckart theorem
                                    for (size_t rho = 0; rho < rhomax; rho++)
                                    {
                                      for (size_t eta = 0; eta < etamax; eta++)
                                      {                                  
                                        auto rme = RME.get(
                                          bra_alpha, ket_alpha, tensor_alpha, rho, eta);
                                        //std::cout << "rme: " << rme << std::endl;

                                        double x = strength * cgcU2 * cgcU3(cgcU3_row,rho) 
                                                 * cgcU4(cgcU4_row,eta) * rme;
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
                            if (ket_irr.u3() != bra_irr.u3())
                              std::cout << "Connect different irreps:" 
                                        << " ket: " << ket_id << ket_irr.u3() 
                                        << " bra: " << bra_id << bra_irr.u3() << std::endl;

                            if (ket_irr.u4() != bra_irr.u4())
                              std::cout << "Connect different irreps:" 
                                        << " ket: " << ket_id << ket_irr.u4() 
                                        << " bra: " << bra_id << bra_irr.u4() << std::endl;
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

    Eigen::SparseMatrix<double> mat(bras_dim, kets_dim);
    mat.setFromTriplets(ija.begin(), ija.end());
    mat.makeCompressed();

    return mat;
  }
} // namespace U3xU4
