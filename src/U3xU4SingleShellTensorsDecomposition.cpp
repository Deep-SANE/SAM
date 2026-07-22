#include <cassert>
#include <stdexcept>
#include <utility>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Fano.h>

#include <U3U2xU3U2.h>
#include <U3SO3xU3SO3.h>

#include <U4U3xU4U3.h>
#include <U4SU2SU2xU4SU2SU2.h>

#include <U3xU4SingleShellTensors.h>
#include <U3xU4SingleShellTensorsDecomposition_Database.h>

namespace U3xU4
{
  /////////////////// ANNIHILATION OPERATOR/TENSOR /////////////////

  // annihilation operator
  double at_gelfand::turnToAnnihilation()
  {
    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps3_conj: Gelfand array for [n,n,0]

    sps4: Gelfand array for [1,0,0,0]
    sps4_conj: Gelfand array for [1,1,1,0]
    */

    assert(tensor);

    const auto& g3_conj = sps3_conj_[c_];
    const auto& g4_conj = sps4_conj_[r_];

    int n13b = g3_conj.n13(), n22b = g3_conj.n22(),
        n11b = g3_conj.n11(), n33p = g4_conj.n33(),
        n22p = g4_conj.n22(), n11p = g4_conj.n11();

    u3::gelfand g3(n13b, 0, 0, n13b - n22b, 0, n13b - n11b);
    c_ = qcx_utils::itou8(u3::findGelfand(g3, sps3_));

    u4::gelfand g4(1, 0, 0, 0, 1 - n33p, 0, 0, 1 - n22p, 0, 1 - n11p);
    r_ = qcx_utils::itou8(u4::findGelfand(g4, sps4_));

    int phase = n22b + n11b + n33p + n22p + n11p;

    tensor = false;

    // std::cout << isTensor() << " " << static_cast<int>(r_) 
    //           << " " << static_cast<int>(c_) << std::endl;

    return (phase % 2 == 0) ? 1 : -1;
  }

  double at_phys::turnToAnnihilation()
  {
    const auto& [n, ll, jj, mmj, mmt] = labels_.getLabels(index);
#ifdef ADD_SU3LIB_PHASE_U3SO3
    int phi = (2 * n + 1 - jj - mmj - mmt) / 2;
#else
    int phi = (n + ll / 2 + 1 - jj - mmj - mmt) / 2;
#endif
    // mmj *= -1;
    // mmt *= -1;
    // index = labels_.getIndex(n, ll, jj, mmj, mmt);

    index = labels_.getIndex(n, ll, jj, -mmj, -mmt);

    tensor = false;
    return (phi % 2 == 0) ? 1.0 : -1.0;
  }

  /////////////// PRODUCT OF CREATION OPERATORS/TENSORS /////////////////

  // expand coupled product of creation tensors (ad's)
  // in terms of creation operators (no phase needed)
  std::pair<std::vector<double>, std::vector<U3xU4::ads_gelfand>>
  Tad_gelfand::expand() const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();
    auto& wigU4U3db = u4::wigU4U3db();

    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps4: Gelfand array for [1,0,0,0]
    */

    std::vector<double> coeffs;
    std::vector<U3xU4::ads_gelfand> tensorsArr;

    auto num = num_ads();
    int shell_dim = 2 * (n_ + 1) * (n_ + 2);

    // one ad
    if (num == 1)
    {
      coeffs.push_back(1.0);
      int c = u3::findGelfand(g3_, sps3_); 
      int r = u4::findGelfand(g4_, sps4_);
      U3xU4::ad_gelfand ad(c, r, sps3_, sps4_);
      //U3xU4::ads_gelfand Tad(1, ad);
      tensorsArr.push_back({ad});
    }

    // two ad's
    else if (num == 2)
    {
      size_t estimate = shell_dim * shell_dim;
      coeffs.reserve(estimate);
      tensorsArr.reserve(estimate);

      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3_, sps3_, sps3_);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4_, sps4_, sps4_);

      int jj3 = g3_.Lambdax2(), mm3 = g3_.MLambdax2(), 
          jj4 = g4_.Lambdax2(), mm4 = g4_.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        int g3_1 = uncouple_g3[i].first;
        const auto& sp3_1 = sps3_[g3_1];

        int g3_2 = uncouple_g3[i].second;
        const auto& sp3_2 = sps3_[g3_2];

        int jj3_1 = sp3_1.Lambdax2(), mm3_1 = sp3_1.MLambdax2(),
            jj3_2 = sp3_2.Lambdax2(), mm3_2 = sp3_2.MLambdax2();

        const auto accU3 = wigU3U2db.get(
          sp3_1.n13(), sp3_1.n23(), sp3_1.n33(), sp3_1.n12(), sp3_1.n22(), 
          sp3_2.n13(), sp3_2.n23(), sp3_2.n33(), sp3_2.n12(), sp3_2.n22(), 
          g3_.n13(), g3_.n23(), g3_.n33(), g3_.n12(), g3_.n22());

        if (accU3.size() == 0) continue;

        double cgc3 = cgcs.get({jj3_1, jj3_2, jj3, mm3_1, mm3_2, mm3});

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          int g4_1 = uncouple_g4[j].first;
          const auto& sp4_1 = sps4_[g4_1];

          int g4_2 = uncouple_g4[j].second;
          const auto& sp4_2 = sps4_[g4_2];

          int jj4_1 = sp4_1.Lambdax2(), mm4_1 = sp4_1.MLambdax2(),
              jj4_2 = sp4_2.Lambdax2(), mm4_2 = sp4_2.MLambdax2();

          const auto accU4 = wigU4U3db.get(
            sp4_1.n14(), sp4_1.n24(), sp4_1.n34(), sp4_1.n44(), 
            sp4_1.n13(), sp4_1.n23(), sp4_1.n33(), 
            sp4_2.n14(), sp4_2.n24(), sp4_2.n34(), sp4_2.n44(), 
            sp4_2.n13(), sp4_2.n23(), sp4_2.n33(), 
            g4_.n14(), g4_.n24(), g4_.n34(), g4_.n44(), 
            g4_.n13(), g4_.n23(), g4_.n33());

          if (accU4.size() == 0) continue;        

          const auto accU3_4 = wigU3U2db.get(
            sp4_1.n13(), sp4_1.n23(), sp4_1.n33(), sp4_1.n12(), sp4_1.n22(), 
            sp4_2.n13(), sp4_2.n23(), sp4_2.n33(), sp4_2.n12(), sp4_2.n22(), 
            g4_.n13(), g4_.n23(), g4_.n33(), g4_.n12(), g4_.n22());

          if (accU3_4.size() == 0) continue;

          double cgc4 = cgcs.get({jj4_1, jj4_2, jj4, mm4_1, mm4_2, mm4});

          double cgcU4 = 0.0;
          for (size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15) continue;

          U3xU4::ad_gelfand ad1(g3_1, g4_1, sps3_, sps4_),
                            ad2(g3_2, g4_2, sps3_, sps4_);
          U3xU4::ads_gelfand adad = {std::move(ad1), std::move(ad2)};

          coeffs.push_back(cgcU3 * cgcU4);
          tensorsArr.push_back(std::move(adad));
        }
      }
    }

    // three ad's
    else if (num == 3)
    {
      size_t estimate = shell_dim * shell_dim * shell_dim;
      coeffs.reserve(estimate);
      tensorsArr.reserve(estimate);

      u3::gelfands g3Arr0 = u3::gelfandBasis(Omegas_[0].u3());
      u4::gelfands g4Arr0 = u4::gelfandBasis(Omegas_[0].u4());

      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3_, g3Arr0, sps3_);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4_, g4Arr0, sps4_);

      int jj3 = g3_.Lambdax2(), mm3 = g3_.MLambdax2(), 
          jj4 = g4_.Lambdax2(), mm4 = g4_.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        const auto& G3_0 = g3Arr0[uncouple_g3[i].first];

        int g3_3 = uncouple_g3[i].second;
        const auto& sp3_3 = sps3_[g3_3];

        int jj3_0 = G3_0.Lambdax2(), mm3_0 = G3_0.MLambdax2(),
            jj3_3 = sp3_3.Lambdax2(), mm3_3 = sp3_3.MLambdax2();

        const auto accU3 = wigU3U2db.get(
          G3_0.n13(), G3_0.n23(), G3_0.n33(), G3_0.n12(), G3_0.n22(), 
          sp3_3.n13(), sp3_3.n23(), sp3_3.n33(), sp3_3.n12(), sp3_3.n22(), 
          g3_.n13(), g3_.n23(), g3_.n33(), g3_.n12(), g3_.n22());

        if (accU3.size() == 0) continue;   

        double cgc3 = cgcs.get({jj3_0, jj3_3, jj3, mm3_0, mm3_3, mm3});

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          const auto& G4_0 = g4Arr0[uncouple_g4[j].first];

          int g4_3 = uncouple_g4[j].second;
          const auto& sp4_3 = sps4_[g4_3];

          int jj4_0 = G4_0.Lambdax2(), mm4_0 = G4_0.MLambdax2(),
              jj4_3 = sp4_3.Lambdax2(), mm4_3 = sp4_3.MLambdax2();

          const auto accU4 = wigU4U3db.get(
            G4_0.n14(), G4_0.n24(), G4_0.n34(), G4_0.n44(), 
            G4_0.n13(), G4_0.n23(), G4_0.n33(), 
            sp4_3.n14(), sp4_3.n24(), sp4_3.n34(), sp4_3.n44(), 
            sp4_3.n13(), sp4_3.n23(), sp4_3.n33(), 
            g4_.n14(), g4_.n24(), g4_.n34(), g4_.n44(), 
            g4_.n13(), g4_.n23(), g4_.n33());

          if (accU4.size() == 0) continue;       

          const auto accU3_4 = wigU3U2db.get(
              G4_0.n13(), G4_0.n23(), G4_0.n33(), G4_0.n12(), G4_0.n22(), 
              sp4_3.n13(), sp4_3.n23(), sp4_3.n33(), sp4_3.n12(), sp4_3.n22(), 
              g4_.n13(), g4_.n23(), g4_.n33(), g4_.n12(), g4_.n22());

          if (accU3_4.size() == 0) continue;

          double cgc4 = cgcs.get({jj4_0, jj4_3, jj4, mm4_0, mm4_3, mm4});

          double cgcU4 = 0.0;
          for (size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15) continue;

          U3xU4::Tad_gelfand Tad(n_, sps3_, sps4_, {Omegas_[0]}, G3_0, G4_0);
          auto [coeffs_0, tensors_0] = Tad.expand();

          U3xU4::ad_gelfand ad3(g3_3, g4_3, sps3_, sps4_);

          cgcU4 *= cgcU3;

          for (size_t k = 0; k < coeffs_0.size(); k++)
          {
            coeffs.push_back(coeffs_0[k] * cgcU4);
            U3xU4::ads_gelfand tensorsArr_temp = tensors_0[k];
            tensorsArr_temp.push_back(ad3);
            tensorsArr.push_back(std::move(tensorsArr_temp));
          }
        }
      }
    }

    return std::make_pair(std::move(coeffs), std::move(tensorsArr));
  }

  // Expand in terms of uncoupled products of ad's
  Expansion Tad_phys::expand() const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    // Flat expansion: coeff rows are indexed by kappa only, every operator is a
    // creation ad. Built straight into Expansion (no Eigen rows).
    Expansion result;
    result.labels = &labels_;
    result.coeffs.ncols = kappamax();
    result.nops = num_ads();
    std::vector<double> scratch;

    auto num = num_ads();
    int ll_max = 2 * n_, ll_min = 2 * (n_ % 2);

    // one ad
    if (num == 1)
    {
      double c = 1.0;
      result.coeffs.push_row(&c);
      U3xU4::ad_phys ad(n_, LL_, JJ_, MMJ_, MMT_, labels_);
      result.append_op(ad.getIndex(), true);
    }

    // two ad's
    else if (num == 2)
    {
      int lam = Omegas_[0].lam(), mu = Omegas_[0].mu(),
          P1 = Omegas_[0].P1(), P2 = Omegas_[0].P2(), P3 = Omegas_[0].P3();

      const auto cgcU4 = wigU4SU2SU2db.get(
        1, 0, 0, 1, 1, 1, 0, 0, 1, 1, P1, P2, P3, SS(), TT());

      assert(cgcU4.size() == 1);

      for (int lla = ll_max; lla >= ll_min; lla -= 4)
      {
        for (int llb = ll_max; llb >= ll_min; llb -= 4)
        {
          // if (abs(lla - llb) > LL_ || (lla + llb) < LL_) continue;

          const auto cgcU3 = wigU3SO3db.get(
            n_, 0, lla / 2, n_, 0, llb / 2, lam, mu, LL() / 2);

          if (cgcU3.size() == 0) continue;

          for (int jja = lla + 1; jja >= abs(lla - 1); jja -= 2)
          {
            for (int jjb = llb + 1; jjb >= abs(llb - 1); jjb -= 2)
            {
              if (abs(jja - jjb) > JJ_ || (jja + jjb) < JJ_) continue;

              double U9j = racah9.get({lla, llb, LL(),
                                          1, 1, SS(),
                                          jja, jjb, JJ()});

              for (int mmja = jja; mmja >= -jja; mmja -= 2)
              {
                int mmjb = MMJ() - mmja;
                if (abs(mmjb) > jjb) continue;

                for (int mmta = 1; mmta >= -1; mmta -= 2)
                {
                  int mmtb = MMT() - mmta;
                  if (abs(mmtb) > 1) continue;

                  double cgcJ = cgcs.get({jja, jjb, JJ(), mmja, mmjb, MMJ()}),
                         cgcT = cgcs.get({1, 1, TT(), mmta, mmtb, MMT()}),
                         factor = cgcU4[0] * U9j * cgcJ * cgcT;

                  scratch.resize(kappamax());
                  double s2 = 0.0;
                  for (int k = 0; k < kappamax(); k++)
                  {
                    scratch[k] = cgcU3[k] * factor;
                    s2 += scratch[k] * scratch[k];
                  }
                  if (s2 < 1e-24) continue;

                  result.coeffs.push_row(scratch.data());

                  U3xU4::ad_phys ad1(n_, lla, jja, mmja, mmta, labels_),
                                 ad2(n_, llb, jjb, mmjb, mmtb, labels_);
                  result.append_op(ad1.getIndex(), true);
                  result.append_op(ad2.getIndex(), true);
                }
              }
            }
          }
        }
      }
    }

    // three ad's
    else if (num == 3)
    {
      std::vector<u3::physicalvector>
        pv3s0 = u3::physicalBasis(Omegas_[0].u3());
      std::vector<u4::physicalvector> 
        pv4s0 = u4::physicalBasis(Omegas_[0].u4());

      int lam = Omegas_[1].lam(), mu = Omegas_[1].mu(),
          P1 = Omegas_[1].P1(), P2 = Omegas_[1].P2(), P3 = Omegas_[1].P3(),
          lam0 = Omegas_[0].lam(), mu0 = Omegas_[0].mu(),
          P10 = Omegas_[0].P1(), P20 = Omegas_[0].P2(), P30 = Omegas_[0].P3();

      for (int llc = ll_max; llc >= ll_min; llc -= 4)
      {
        for (size_t pv30 = 0; pv30 < pv3s0.size(); pv30++)
        {
          int LL0 = 2 * pv3s0[pv30].L();

          // if (abs(LL0 - llc) > LL_ || (LL0 + llc) < LL_) continue;

          const auto cgcU3 = wigU3SO3db.get(
            lam0, mu0, LL0 / 2, n_, 0, llc / 2, lam, mu, LL() / 2);

          if (cgcU3.size() == 0) continue;   

          for (size_t pv40 = 0; pv40 < pv4s0.size(); pv40++)
          {
            int SS0 = pv4s0[pv40].SS(), TT0 = pv4s0[pv40].TT();

            // if (abs(SS0 - 1) > SS_ || (SS0 + 1) < SS_ ||
            //     abs(TT0 - 1) > TT_ || (TT0 + 1) < TT_)
            //   continue;

            const auto cgcU4 = wigU4SU2SU2db.get(
              P10, P20, P30, SS0, TT0, 1, 0, 0, 1, 1, P1, P2, P3, SS(), TT());

            if (cgcU4.size() == 0) continue;   
            
            for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
            {
              for (int JJ0 = LL0 + SS0; JJ0 >= abs(LL0 - SS0); JJ0 -= 2)
              {
                if (abs(JJ0 - jjc) > JJ_ || (JJ0 + jjc) < JJ_) continue;

                double U9j = racah9.get({LL0, llc, LL(),
                                         SS0, 1, SS(),
                                         JJ0, jjc, JJ()});

                for (int mmjc = jjc; mmjc >= -jjc; mmjc -= 2)
                {
                  int MMJ0 = MMJ() - mmjc;
                  if (abs(MMJ0) > JJ0) continue;

                  for (int mmtc = 1; mmtc >= -1; mmtc -= 2)
                  {
                    int MMT0 = MMT() - mmtc;
                    if (abs(MMT0) > TT0) continue;

                    double cgcJ = cgcs.get({JJ0, jjc, JJ(), MMJ0, mmjc, MMJ()}),
                           cgcT = cgcs.get({TT0, 1, TT(), MMT0, mmtc, MMT()}),
                           factor = cgcU4[0] * U9j * cgcJ * cgcT;

                    U3xU4::Tad_phys Tad0(n_, {Omegas_[0]}, LL0, SS0, TT0, MMT0,
                                         JJ0, MMJ0, labels_);
                    Expansion e0 = Tad0.expand();

                    U3xU4::ad_phys ad3(n_, llc, jjc, mmjc, mmtc, labels_);
                    uint16_t ad3_idx = ad3.getIndex();

                    for (size_t i = 0; i < e0.rows(); i++)
                    {
                      const double* c0 = e0.coeff_row(i);
                      scratch.assign(kappamax(), 0.0);
                      double s2 = 0.0;
                      for (int k = 0; k < kappamax(); k++)
                      {
                        for (int k0 = 0; k0 < Tad0.kappamax(); k0++)
                          scratch[k] += cgcU3[k0 * kappamax() + k] * factor * c0[k0];
                        s2 += scratch[k] * scratch[k];
                      }
                      if (s2 < 1e-24) continue;

                      result.coeffs.push_row(scratch.data());
                      result.append_ops_from(e0, i);    // the first two ad's
                      result.append_op(ad3_idx, true);  // the third ad
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    return result;
  }

  /////////////// PRODUCT OF ANNIHILATION OPERATORS/TENSORS ///////////////

  // expand coupled product of annihilation tensors (at's)
  // in terms of annihilation operators (a's) via a phase
  std::pair<std::vector<double>, std::vector<U3xU4::ats_gelfand>>
  Tat_gelfand::expand() const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();
    auto& wigU4U3db = u4::wigU4U3db();

    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps3_conj: Gelfand array for [n,n,0]

    sps4: Gelfand array for [1,0,0,0]
    sps4_conj: Gelfand array for [1,1,1,0]
    */

    std::vector<double> coeffs;
    std::vector<U3xU4::ats_gelfand> tensorsArr;

    auto num = num_ats();
    int shell_dim = 2 * (n_ + 1) * (n_ + 2);

    // one at
    if (num == 1)
    {
      auto c = u3::findGelfand(g3_, sps3_conj_),
           r = u4::findGelfand(g4_, sps4_conj_);
      U3xU4::at_gelfand at(c, r, sps3_, sps3_conj_, sps4_, sps4_conj_);
      auto phase = at.turnToAnnihilation();

      assert(!at.isTensor());

      coeffs.push_back(phase);
      tensorsArr.push_back({at});
    }

    // two at's
    else if (num == 2)
    {
      size_t estimate = shell_dim * shell_dim;
      coeffs.reserve(estimate);
      tensorsArr.reserve(estimate);

      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3_, sps3_conj_, sps3_conj_);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4_, sps4_conj_, sps4_conj_);

      int jj3 = g3_.Lambdax2(), mm3 = g3_.MLambdax2(), 
          jj4 = g4_.Lambdax2(), mm4 = g4_.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        int g3_1 = uncouple_g3[i].first;
        const auto& sp3_1 = sps3_conj_[g3_1];

        int g3_2 = uncouple_g3[i].second;
        const auto& sp3_2 = sps3_conj_[g3_2];

        int jj3_1 = sp3_1.Lambdax2(), mm3_1 = sp3_1.MLambdax2(),
            jj3_2 = sp3_2.Lambdax2(), mm3_2 = sp3_2.MLambdax2();

        const auto accU3 = wigU3U2db.get(
          sp3_1.n13(), sp3_1.n23(), sp3_1.n33(), sp3_1.n12(), sp3_1.n22(), 
          sp3_2.n13(), sp3_2.n23(), sp3_2.n33(), sp3_2.n12(), sp3_2.n22(), 
          g3_.n13(), g3_.n23(), g3_.n33(), g3_.n12(), g3_.n22());

        if (accU3.size() == 0) continue;   

        double cgc3 = cgcs.get({jj3_1, jj3_2, jj3, mm3_1, mm3_2, mm3});

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          int g4_1 = uncouple_g4[j].first;
          const auto& sp4_1 = sps4_conj_[g4_1];

          int g4_2 = uncouple_g4[j].second;
          const auto& sp4_2 = sps4_conj_[g4_2];

          int jj4_1 = sp4_1.Lambdax2(), mm4_1 = sp4_1.MLambdax2(),
              jj4_2 = sp4_2.Lambdax2(), mm4_2 = sp4_2.MLambdax2();

          const auto accU4 = wigU4U3db.get(
            sp4_1.n14(), sp4_1.n24(), sp4_1.n34(), sp4_1.n44(),
            sp4_1.n13(), sp4_1.n23(), sp4_1.n33(), 
            sp4_2.n14(), sp4_2.n24(), sp4_2.n34(), sp4_2.n44(), 
            sp4_2.n13(), sp4_2.n23(), sp4_2.n33(), 
            g4_.n14(), g4_.n24(), g4_.n34(), g4_.n44(), 
            g4_.n13(), g4_.n23(), g4_.n33());

          if (accU4.size() == 0) continue;   

          const auto accU3_4 = wigU3U2db.get(
            sp4_1.n13(), sp4_1.n23(), sp4_1.n33(), sp4_1.n12(), sp4_1.n22(), 
            sp4_2.n13(), sp4_2.n23(), sp4_2.n33(), sp4_2.n12(), sp4_2.n22(),
            g4_.n13(), g4_.n23(), g4_.n33(), g4_.n12(), g4_.n22());

          if (accU3_4.size() == 0) continue;

          double cgc4 = cgcs.get({jj4_1, jj4_2, jj4, mm4_1, mm4_2, mm4});

          double cgcU4 = 0.0;
          for (size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15) continue;

          U3xU4::at_gelfand at1(g3_1, g4_1, sps3_, sps3_conj_, sps4_, sps4_conj_),
                            at2(g3_2, g4_2, sps3_, sps3_conj_, sps4_, sps4_conj_);
          auto phase1 = at1.turnToAnnihilation();
          auto phase2 = at2.turnToAnnihilation();

          assert(!at1.isTensor());
          assert(!at2.isTensor());

          U3xU4::ats_gelfand atat = {std::move(at1), std::move(at2)};

          coeffs.push_back(phase1 * phase2 * cgcU3 * cgcU4);
          tensorsArr.push_back(std::move(atat));
        }
      }
    }

    // three at's
    else if (num == 3)
    {
      size_t estimate = shell_dim * shell_dim * shell_dim;
      coeffs.reserve(estimate);
      tensorsArr.reserve(estimate);

      std::vector<u3::gelfand> 
        g3Arr0 = u3::gelfandBasis(Omegas_[0].u3());
      std::vector<u4::gelfand> 
        g4Arr0 = u4::gelfandBasis(Omegas_[0].u4());

      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3_, g3Arr0, sps3_conj_);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4_, g4Arr0, sps4_conj_);

      int jj3 = g3_.Lambdax2(), mm3 = g3_.MLambdax2(), 
          jj4 = g4_.Lambdax2(), mm4 = g4_.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        const auto& G3_0 = g3Arr0[uncouple_g3[i].first];

        int g3_3 = uncouple_g3[i].second;
        const auto& sp3_3 = sps3_conj_[g3_3];

        int jj3_0 = G3_0.Lambdax2(), mm3_0 = G3_0.MLambdax2(),
            jj3_3 = sp3_3.Lambdax2(), mm3_3 = sp3_3.MLambdax2();

        const auto accU3 = wigU3U2db.get(
          G3_0.n13(), G3_0.n23(), G3_0.n33(), G3_0.n12(), G3_0.n22(), 
          sp3_3.n13(), sp3_3.n23(), sp3_3.n33(), sp3_3.n12(), sp3_3.n22(), 
          g3_.n13(), g3_.n23(), g3_.n33(), g3_.n12(), g3_.n22());

        if (accU3.size() == 0) continue;

        double cgc3 = cgcs.get({jj3_0, jj3_3, jj3, mm3_0, mm3_3, mm3});

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          const auto& G4_0 = g4Arr0[uncouple_g4[j].first];

          int g4_3 = uncouple_g4[j].second;
          const auto& sp4_3 = sps4_conj_[g4_3];

          int jj4_0 = G4_0.Lambdax2(), mm4_0 = G4_0.MLambdax2(),
              jj4_3 = sp4_3.Lambdax2(), mm4_3 = sp4_3.MLambdax2();

          const auto accU4 = wigU4U3db.get(
            G4_0.n14(), G4_0.n24(), G4_0.n34(), G4_0.n44(), 
            G4_0.n13(), G4_0.n23(), G4_0.n33(), 
            sp4_3.n14(), sp4_3.n24(), sp4_3.n34(), sp4_3.n44(),
            sp4_3.n13(), sp4_3.n23(), sp4_3.n33(), 
            g4_.n14(), g4_.n24(), g4_.n34(), g4_.n44(),
            g4_.n13(), g4_.n23(), g4_.n33());

          if (accU4.size() == 0) continue;       

          const auto accU3_4 = wigU3U2db.get(
            G4_0.n13(), G4_0.n23(), G4_0.n33(), G4_0.n12(), G4_0.n22(), 
            sp4_3.n13(), sp4_3.n23(), sp4_3.n33(), sp4_3.n12(), sp4_3.n22(), 
            g4_.n13(), g4_.n23(), g4_.n33(), g4_.n12(), g4_.n22());

          if (accU3_4.size() == 0) continue;

          double cgc4 = cgcs.get({jj4_0, jj4_3, jj4, mm4_0, mm4_3, mm4});

          double cgcU4 = 0.0;
          for (size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15) continue;

          U3xU4::at_gelfand at3(g3_3, g4_3, sps3_, sps3_conj_, sps4_, sps4_conj_);
          auto phase3 = at3.turnToAnnihilation();

          assert(!at3.isTensor());

          U3xU4::Tat_gelfand Tat(n_, sps3_, sps3_conj_, sps4_, sps4_conj_, 
                                 {Omegas_[0]}, G3_0, G4_0);

          auto [coeffs_0, tensors_0] = Tat.expand();

          cgcU4 *= (phase3 * cgcU3);

          for (size_t k = 0; k < coeffs_0.size(); k++)
          {
            coeffs.push_back(coeffs_0[k]  * cgcU4);
            U3xU4::ats_gelfand tensorsArr_temp = tensors_0[k];
            tensorsArr_temp.push_back(at3);
            tensorsArr.push_back(std::move(tensorsArr_temp));
          }
        }
      }
    }

    return std::make_pair(std::move(coeffs), std::move(tensorsArr));
  }

  // expand in terms of uncoupled products of at's
  Expansion Tat_phys::expand() const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    // Flat expansion: coeff rows are indexed by kappa only, every operator is an
    // annihilation a~ (turnToAnnihilation already applied). No Eigen rows.
    Expansion result;
    result.labels = &labels_;
    result.coeffs.ncols = kappamax();
    result.nops = num_ats();
    std::vector<double> scratch;

    auto num = num_ats();
    int ll_max = 2 * n_, ll_min = 2 * (n_ % 2);

    // one at
    if (num == 1)
    {
      U3xU4::at_phys at(n_, LL_, JJ_, MMJ_, MMT_, labels_);
      double c = at.turnToAnnihilation();
      result.coeffs.push_row(&c);
      result.append_op(at.getIndex(), false);
    }

    // two at's
    else if (num == 2)
    {
      int lam = Omegas_[0].lam(), mu = Omegas_[0].mu(),
          P1 = Omegas_[0].P1(), P2 = Omegas_[0].P2(), P3 = Omegas_[0].P3();

      const auto cgcU4 = wigU4SU2SU2db.get(
        0, 0, 1, 1, 1, 0, 0, 1, 1, 1, P1, P2, P3, SS(), TT());

      assert(cgcU4.size() > 0);

      for (int lla = ll_max; lla >= ll_min; lla -= 4)
      {
        for (int llb = ll_max; llb >= ll_min; llb -= 4)
        {
          // if (abs(lla - llb) > LL_ || (lla + llb) < LL_) continue;

          const auto cgcU3 = wigU3SO3db.get(
            0, n_, lla / 2, 0, n_, llb / 2, lam, mu, LL() / 2);

          if (cgcU3.size() == 0) continue;          

          for (int jja = lla + 1; jja >= abs(lla - 1); jja -= 2)
          {
            for (int jjb = llb + 1; jjb >= abs(llb - 1); jjb -= 2)
            {
              if (abs(jja - jjb) > JJ_ || (jja + jjb) < JJ_) continue;

              double U9j = racah9.get({lla, llb, LL(),
                                          1, 1, SS(),
                                          jja, jjb, JJ()});

              for (int mmja = jja; mmja >= -jja; mmja -= 2)
              {
                int mmjb = MMJ() - mmja;
                if (abs(mmjb) > jjb) continue;

                for (int mmta = 1; mmta >= -1; mmta -= 2)
                {
                  int mmtb = MMT() - mmta;
                  if (abs(mmtb) > 1) continue;

                  double cgcJ = cgcs.get({jja, jjb, JJ(), mmja, mmjb, MMJ()}),
                         cgcT = cgcs.get({1, 1, TT(), mmta, mmtb, MMT()});

                  U3xU4::at_phys at1(n_, lla, jja, mmja, mmta, labels_),
                                 at2(n_, llb, jjb, mmjb, mmtb, labels_);
                  double phase1 = at1.turnToAnnihilation(),
                         phase2 = at2.turnToAnnihilation(),
                         factor = cgcU4[0] * U9j * cgcJ * cgcT * phase1 * phase2;

                  scratch.resize(kappamax());
                  double s2 = 0.0;
                  for (int k = 0; k < kappamax(); k++)
                  {
                    scratch[k] = cgcU3[k] * factor;
                    s2 += scratch[k] * scratch[k];
                  }
                  if (s2 < 1e-24) continue;

                  result.coeffs.push_row(scratch.data());

                  assert(!at1.isTensor());
                  assert(!at2.isTensor());
                  result.append_op(at1.getIndex(), false);
                  result.append_op(at2.getIndex(), false);
                }
              }
            }
          }
        }
      }
    }

    // three at's
    else if (num == 3)
    {
      std::vector<u3::physicalvector>
        pv3s0 = u3::physicalBasis(Omegas_[0].u3());
      std::vector<u4::physicalvector> 
        pv4s0 = u4::physicalBasis(Omegas_[0].u4());

      int lam = Omegas_[1].lam(), mu = Omegas_[1].mu(),
          P1 = Omegas_[1].P1(), P2 = Omegas_[1].P2(), P3 = Omegas_[1].P3(),
          lam0 = Omegas_[0].lam(), mu0 = Omegas_[0].mu(),
          P10 = Omegas_[0].P1(), P20 = Omegas_[0].P2(), P30 = Omegas_[0].P3();

      for (int llc = ll_max; llc >= ll_min; llc -= 4)
      {
        for (size_t pv30 = 0; pv30 < pv3s0.size(); pv30++)
        {
          int LL0 = 2 * pv3s0[pv30].L();

          // if (abs(LL0 - llc) > LL_ || (LL0 + llc) < LL_) continue;

          const auto cgcU3 = wigU3SO3db.get(
            lam0, mu0, LL0 / 2, 0, n_, llc / 2, lam, mu, LL() / 2);

          if (cgcU3.empty()) continue;

          for (size_t pv40 = 0; pv40 < pv4s0.size(); pv40++)
          {
            int SS0 = pv4s0[pv40].SS(), TT0 = pv4s0[pv40].TT();

            // if (abs(SS0 - 1) > SS_ || (SS0 + 1) < SS_ ||
            //     abs(TT0 - 1) > TT_ || (TT0 + 1) < TT_)
            //   continue;

            const auto cgcU4 = wigU4SU2SU2db.get(
              P10, P20, P30, SS0, TT0, 0, 0, 1, 1, 1, P1, P2, P3, SS(), TT());

            if (cgcU4.empty()) continue;

            for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
            {
              for (int JJ0 = LL0 + SS0; JJ0 >= abs(LL0 - SS0); JJ0 -= 2)
              {
                if (abs(JJ0 - jjc) > JJ_ || (JJ0 + jjc) < JJ_) continue;

                double U9j = racah9.get({LL0, llc, LL(),
                                            SS0, 1, SS(),
                                            JJ0, jjc, JJ()});

                for (int mmjc = jjc; mmjc >= -jjc; mmjc -= 2)
                {
                  int MMJ0 = MMJ() - mmjc;
                  if (abs(MMJ0) > JJ0) continue;

                  for (int mmtc = 1; mmtc >= -1; mmtc -= 2)
                  {
                    int MMT0 = MMT() - mmtc;
                    if (abs(MMT0) > TT0) continue;

                    double cgcJ = cgcs.get({JJ0, jjc, JJ(), MMJ0, mmjc, MMJ()}),
                           cgcT = cgcs.get({TT0, 1, TT(), MMT0, mmtc, MMT()}),
                           factor = cgcU4[0] * U9j * cgcJ * cgcT;

                    U3xU4::Tat_phys Tat0(n_, {Omegas_[0]}, LL0, SS0, TT0, MMT0,
                                         JJ0, MMJ0, labels_);
                    Expansion e0 = Tat0.expand();

                    U3xU4::at_phys at3(n_, llc, jjc, mmjc, mmtc, labels_);
                    double phase3 = at3.turnToAnnihilation();
                    factor *= phase3;
                    uint16_t at3_idx = at3.getIndex();

                    for (size_t i = 0; i < e0.rows(); i++)
                    {
                      const double* c0 = e0.coeff_row(i);
                      scratch.assign(kappamax(), 0.0);
                      double s2 = 0.0;
                      for (int k = 0; k < kappamax(); k++)
                      {
                        for (int k0 = 0; k0 < Tat0.kappamax(); k0++)
                          scratch[k] += cgcU3[k0 * kappamax() + k] * factor * c0[k0];
                        s2 += scratch[k] * scratch[k];
                      }
                      if (s2 < 1e-24) continue;

                      result.coeffs.push_row(scratch.data());
                      result.append_ops_from(e0, i);     // the first two a's
                      result.append_op(at3_idx, false);  // the third a
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    return result;
  }

  ////// PRODUCT OF CREATION AND ANNIHILATION OPERATORS/TENSORS ///////

  // expand coupled product of creation and annihilation tensors
  // in terms of creation and annihilation operators (ad's and a's)
  std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
             std::vector<U3xU4::ads_gelfand>, // ad's
             std::vector<U3xU4::ats_gelfand>> // a's
  SingleShellTensor_gelfand::expand() const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();
    auto& wigU4U3db = u4::wigU4U3db();

    std::vector<Eigen::RowVectorXd> coeffs;
    std::vector<U3xU4::ads_gelfand> adsArr;
    std::vector<U3xU4::ats_gelfand> atsArr;

    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps3_conj: Gelfand array for [n,n,0]

    sps4: Gelfand array for [1,0,0,0]
    sps4_conj: Gelfand array for [1,1,1,0]
    */

    auto n = shell();
    auto rhomax = rho0max();
    auto etamax = eta0max();

    // only ads
    if (num_ats() == 0)
    {
      U3xU4::Tad_gelfand Tad(n, sps3_, sps4_, Omega_ads(), g3_, g4_);

      auto [coeffs_, adsArr_] = Tad.expand();

      for (size_t i = 0; i < coeffs_.size(); i++)
      {
        Eigen::RowVectorXd x(1);
        x(0) = coeffs_[i];
        coeffs.push_back(x);
        adsArr.push_back(std::move(adsArr_[i]));
      }
    }

    // only ats
    else if (num_ads() == 0)
    {
      U3xU4::Tat_gelfand Tat(n, sps3_, sps3_conj_, sps4_, sps4_conj_, 
                             Omega_ats(), g3_, g4_);

      auto [coeffs_, atsArr_] = Tat.expand();

      for (size_t i = 0; i < coeffs_.size(); i++)
      {
        Eigen::RowVectorXd x(1);
        x(0) = coeffs_[i];
        coeffs.push_back(x);
        atsArr.push_back(std::move(atsArr_[i]));
      }
    }

    else
    {
      int shell_dim = 2 * (n + 1) * (n + 2), estimate = 1;
      for (int i = 0; i < num_ads(); i++) estimate *= shell_dim;
      for (int i = 0; i < num_ats(); i++) estimate *= shell_dim;
      coeffs.reserve(estimate);
      adsArr.reserve(estimate);
      atsArr.reserve(estimate);

      u3::gelfands g3Arr1, g3Arr2;
      u4::gelfands g4Arr1, g4Arr2;

      if (num_ads() > 1)
      {
        g3Arr1 = u3::gelfandBasis(Omega_ads().back().u3());
        g4Arr1 = u4::gelfandBasis(Omega_ads().back().u4());
      }
      else
      {
        g3Arr1 = sps3_;
        g4Arr1 = sps4_;
      }

      if (num_ats() > 1)
      {
        g3Arr2 = u3::gelfandBasis(Omega_ats().back().u3());
        g4Arr2 = u4::gelfandBasis(Omega_ats().back().u4());
      }
      else
      {
        g3Arr2 = sps3_conj_;
        g4Arr2 = sps4_conj_;
      }

      // uncouple the final tensor to Tad and Tat
      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3_, g3Arr1, g3Arr2);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4_, g4Arr1, g4Arr2);

      int jj3 = g3_.Lambdax2(), mm3 = g3_.MLambdax2(), 
          jj4 = g4_.Lambdax2(), mm4 = g4_.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        const auto& g3_1 = g3Arr1[uncouple_g3[i].first];
        const auto& g3_2 = g3Arr2[uncouple_g3[i].second];

        int jj3_1 = g3_1.Lambdax2(), mm3_1 = g3_1.MLambdax2(),
            jj3_2 = g3_2.Lambdax2(), mm3_2 = g3_2.MLambdax2();

        const auto accU3 = wigU3U2db.get(
          g3_1.n13(), g3_1.n23(), g3_1.n33(), g3_1.n12(), g3_1.n22(), 
          g3_2.n13(), g3_2.n23(), g3_2.n33(), g3_2.n12(), g3_2.n22(), 
          g3_.n13(), g3_.n23(), g3_.n33(), g3_.n12(), g3_.n22());
        
        if (accU3.size() == 0) continue;

        double cgc3 = cgcs.get({jj3_1, jj3_2, jj3, mm3_1, mm3_2, mm3});

        auto cgcU3 = accU3 * cgc3;

        if (cgcU3.norm() <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          const auto& g4_1 = g4Arr1[uncouple_g4[j].first]; 
          const auto& g4_2 = g4Arr2[uncouple_g4[j].second];

          int jj4_1 = g4_1.Lambdax2(), mm4_1 = g4_1.MLambdax2(),
              jj4_2 = g4_2.Lambdax2(), mm4_2 = g4_2.MLambdax2();

          const auto accU4 = wigU4U3db.get(
            g4_1.n14(), g4_1.n24(), g4_1.n34(), g4_1.n44(), 
            g4_1.n13(), g4_1.n23(), g4_1.n33(), 
            g4_2.n14(), g4_2.n24(), g4_2.n34(), g4_2.n44(), 
            g4_2.n13(), g4_2.n23(), g4_2.n33(), 
            g4_.n14(), g4_.n24(), g4_.n34(), g4_.n44(), 
            g4_.n13(), g4_.n23(), g4_.n33());

          if (accU4.size() == 0) continue;

          const auto accU3_4 = wigU3U2db.get(
            g4_1.n13(), g4_1.n23(), g4_1.n33(), g4_1.n12(), g4_1.n22(), 
            g4_2.n13(), g4_2.n23(), g4_2.n33(), g4_2.n12(), g4_2.n22(), 
            g4_.n13(), g4_.n23(), g4_.n33(), g4_.n12(), g4_.n22());

          if (accU3_4.size() == 0) continue;

          double cgc4 = cgcs.get({jj4_1, jj4_2, jj4, mm4_1, mm4_2, mm4});

          Eigen::RowVectorXd cgcU4(etamax);
          cgcU4.setZero();

          for (size_t e = 0; e < etamax; e++)
            for (size_t r = 0; r < accU4.rows(); r++)
              cgcU4(e) += accU4(r, e) * accU3_4(r) * cgc4;

          if (cgcU4.norm() <= 1e-15) continue;

          // uncouple Tad
          U3xU4::Tad_gelfand Tad(n, sps3_, sps4_, Omega_ads(), g3_1, g4_1);
          auto [coeffs_ads, adsArr_] = Tad.expand();

          // uncouple Tat
          U3xU4::Tat_gelfand Tat(n, sps3_, sps3_conj_, sps4_, sps4_conj_,
                                 Omega_ats(), g3_2, g4_2);
          auto [coeffs_ats, atsArr_] = Tat.expand();

          for (size_t k = 0; k < coeffs_ads.size(); k++)
          {
            for (size_t l = 0; l < coeffs_ats.size(); l++)
            {
              Eigen::RowVectorXd coeffs_(rhomax * etamax);
              coeffs_.setZero();

              for (size_t r = 0; r < rhomax; r++)
                for (size_t e = 0; e < etamax; e++)
                {
                  coeffs_(r * etamax + e) +=
                      cgcU3(r) * cgcU4(e) * coeffs_ads[k] * coeffs_ats[l];
                }

              coeffs.push_back(std::move(coeffs_));
              adsArr.push_back(adsArr_[k]);
              atsArr.push_back(atsArr_[l]);
            }
          }
        }
      }
    }

    return std::make_tuple(std::move(coeffs), 
                           std::move(adsArr), std::move(atsArr));
  }

  // expand coupled product of creation and annihilation tensors
  // in terms of creation and annihilation operators (ad's and a's)
  Expansion SingleShellTensor_phys::expand() const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    auto n = shell();
    auto rhomax = rho0max();
    auto etamax = eta0max();

    // pure creation tensor: the whole expansion is Tad's (all ad operators)
    if (num_ats() == 0)
    {
      U3xU4::Tad_phys Tad(n, Omega_ads(), LL_, SS_, TT_, MMT_, JJ_, MMJ_, labels_);
      return Tad.expand();
    }

    // pure annihilation tensor: the whole expansion is Tat's (all a operators)
    if (num_ads() == 0)
    {
      U3xU4::Tat_phys Tat(n, Omega_ats(), LL_, SS_, TT_, MMT_, JJ_, MMJ_, labels_);
      return Tat.expand();
    }

    // both ad's and a's: couple Tad x Tat into the flat Expansion directly.
    Expansion result;
    result.labels = &labels_;
    result.coeffs.ncols = rhomax * etamax * kappamax_ * zetamax_;
    result.nops = num_ads() + num_ats();
    std::vector<double> scratch;

    {
      u3::physicalvectors pv3s1, pv3s2;
      u4::physicalvectors pv4s1, pv4s2;

      if (num_ads() > 1)
      {
        pv3s1 = u3::physicalBasis(Omega_ads().back().u3());
        pv4s1 = u4::physicalBasis(Omega_ads().back().u4());
      }
      else
      {
        pv3s1 = u3::physicalBasis(u3::irrep(n, 0));
        pv4s1 = u4::physicalBasis(u4::irrep(1, 0, 0));
      }

      if (num_ats() > 1)
      {
        pv3s2 = u3::physicalBasis(Omega_ats().back().u3());
        pv4s2 = u4::physicalBasis(Omega_ats().back().u4());
      }
      else
      {
        pv3s2 = u3::physicalBasis(u3::irrep(0, n));
        pv4s2 = u4::physicalBasis(u4::irrep(0, 0, 1));
      }

      // uncouple the final tensor to Tad and Tat
      for (const auto& pv31 : pv3s1)
      {
        int lam1 = pv31.lam(), mu1 = pv31.mu();
        int kappa1 = pv31.kappa(), LL1 = 2 * pv31.L();

        for (const auto& pv32 : pv3s2)
        {
          int lam2 = pv32.lam(), mu2 = pv32.mu();
          int kappa2 = pv32.kappa(), LL2 = 2 * pv32.L();

          if (abs(LL1 - LL2) > LL_ || (LL1 + LL2) < LL_) continue;

          const auto cgcU3 = wigU3SO3db.get_U3SO3(lam1, mu1, LL1 / 2, 
            lam2, mu2, LL2 / 2, lam(), mu(), LL() / 2);
          if (cgcU3.empty()) continue;

          for (const auto& pv41 : pv4s1)
          {
            int P11 = pv41.P1(), P21 = pv41.P2(), P31 = pv41.P3(); 
            int SS1 = pv41.SS(), TT1 = pv41.TT();

            for (const auto& pv42 : pv4s2)
            {
              int P12 = pv42.P1(), P22 = pv42.P2(), P32 = pv42.P3(); 
              int SS2 = pv42.SS(), TT2 = pv42.TT();

              if (abs(SS1 - SS2) > SS_ || (SS1 + SS2) < SS_ ||
                  abs(TT1 - TT2) > TT_ || (TT1 + TT2) < TT_)
                continue;

              const auto cgcU4 = wigU4SU2SU2db.get_U4SU2SU2(
                                  P11, P21, P31, SS1, TT1, 
                                  P12, P22, P32, SS2, TT2, 
                                  P1(), P2(), P3(), SS(), TT());
              if (cgcU4.empty()) continue;

              for (int JJ1 = LL1 + SS1; JJ1 >= abs(LL1 - SS1); JJ1 -= 2)
              {
                for (int MMJ1 = JJ1; MMJ1 >= -JJ1; MMJ1 -= 2)
                {
                  int MMJ2 = MMJ() - MMJ1;
                  
                  for (int MMT1 = TT1; MMT1 >= -TT1; MMT1 -= 2)
                  {
                    int MMT2 = MMT() - MMT1;
                    if (abs(MMT2) > TT2) continue;
                    double cgcT = cgcs.get({TT1, TT2, TT(), MMT1, MMT2, MMT()});

                    U3xU4::Tad_phys 
                      Tad(n, Omega_ads(), LL1, SS1, TT1, MMT1, JJ1, MMJ1, labels_);

                    Expansion ead;   // lazy Tad expansion (computed once)

                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ_) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;

                      double U9j = racah9.get({LL1, LL2, LL(),
                                                SS1, SS2, SS(),
                                                JJ1, JJ2, JJ()});

                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (ead.empty()) ead = Tad.expand();

                      U3xU4::Tat_phys
                        Tat(n, Omega_ats(), LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);
                      Expansion eat = Tat.expand();

                      for (size_t i = 0; i < ead.rows(); i++)
                      {
                        const double* c1 = ead.coeff_row(i);
                        for (size_t j = 0; j < eat.rows(); j++)
                        {
                          const double* c2 = eat.coeff_row(j);
                          scratch.assign(result.coeffs.ncols, 0.0);
                          for (int rho = 0; rho < rhomax; rho++)
                            for (int eta = 0; eta < etamax; eta++)
                              for (int k = 0; k < kappamax_; k++)
                                for (int z = 0; z < zetamax_; z++)
                                {
                                  int index =
                                      ((rho * etamax + eta) * kappamax_ + k) *
                                          zetamax_ +
                                      z;
                                  for (int k1 = 0; k1 < kappa1; k1++)
                                    for (int k2 = 0; k2 < kappa2; k2++)
                                      scratch[index] +=
                                          cgcU3(k1, k2, k, rho) *
                                          cgcU4(0, 0, z, eta) * cgcU2 *
                                          c1[k1] * c2[k2];
                                }

                          double s2 = 0.0;
                          for (double v : scratch) s2 += v * v;
                          if (s2 < 1e-24) continue;

                          result.coeffs.push_row(scratch.data());
                          result.append_ops_from(ead, i);   // the ad's
                          result.append_ops_from(eat, j);   // the a's
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    return result;
  }

  // get two-body strength directly during expand()
  std::vector<double> SingleShellTensor_phys::getStrengths(
        const HO::TwBME_scalar& TBME, SSTDecomp_phys_db& db) const
  {
    assert(num_ads() == 2 && num_ats() == 2);

    int ncols = rho0max() * eta0max() * kappamax_ * zetamax_;
    std::vector<double> strengths(ncols, 0.0);

    // The cached expansion already carries the full cgcU3*cgcU4*cgcU2 *
    // coeffs1*coeffs2 contraction in coeffs[t]; the strength is just the sum
    // over expansion terms of me(t) * coeffs[t]. (Equivalent to the fused
    // version below, up to expand()'s 1e-12 norm filtering of null terms.)
    // Zero-copy borrowed view: coeff rows are flat double arrays and the ad/at
    // operators are stored as Table_nljmjmt indices (turned into labels on
    // demand) -- no Eigen row allocation, no Record copy, no ad_phys rebuild.
    auto e = db.get_view(*this);
    assert(static_cast<int>(e->ncols()) == ncols);

    for (size_t t = 0; t < e->rows(); t++)
    {
      // operators are stored ad's-then-a's, so slots 0,1 are ad's and 2,3 are
      // a's (no phase, no permutation); labels come straight from the table.
      std::array<int, 4> ns, lls, jjs, mmjs, mmts;
      for (int k = 0; k < 4; k++)
        std::tie(ns[k], lls[k], jjs[k], mmjs[k], mmts[k]) = e->op_labels(t, k);

      double me = TBME.getME(
        ns[0], lls[0], jjs[0], mmjs[0], mmts[0],
        ns[1], lls[1], jjs[1], mmjs[1], mmts[1],
        ns[3], lls[3], jjs[3], mmjs[3], mmts[3],
        ns[2], lls[2], jjs[2], mmjs[2], mmts[2]) / 4.0;

      if (std::fabs(me) < 1e-10) continue;

      const double* row = e->coeff_row(t);
      for (int c = 0; c < ncols; c++)
        strengths[c] += me * row[c];
    }

    return strengths;
/*
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    assert(num_ads() == 2 && num_ats() == 2);

    auto n = shell();
    auto rhomax = rho0max();
    auto etamax = eta0max();

    std::vector<double> strengths(rhomax * etamax * kappamax_ * zetamax_, 0.0);

    u3::physicalvectors
      pv3s1 = u3::physicalBasis(Omega_ads().back().u3()),
      pv3s2 = u3::physicalBasis(Omega_ats().back().u3());

    u4::physicalvectors
      pv4s1 = u4::physicalBasis(Omega_ads().back().u4()),
      pv4s2 = u4::physicalBasis(Omega_ats().back().u4());

    for (const auto& pv31 : pv3s1)
    {
      int lam1 = pv31.lam(), mu1 = pv31.mu(),
          kappa1 = pv31.kappa(), LL1 = 2 * pv31.L();

      for (const auto& pv32 : pv3s2)
      {
        int lam2 = pv32.lam(), mu2 = pv32.mu(),
            kappa2 = pv32.kappa(), LL2 = 2 * pv32.L();

        const auto cgcU3 = wigU3SO3db.get_U3SO3(lam1, mu1, LL1 / 2,
                        lam2, mu2, LL2 / 2, lam(), mu(), LL() / 2);
        if (cgcU3.empty()) continue;

        for (const auto& pv41 : pv4s1)
        {
          int P11 = pv41.P1(), P21 = pv41.P2(), P31 = pv41.P3(),
              SS1 = pv41.SS(), TT1 = pv41.TT();

          for (const auto& pv42 : pv4s2)
          {
            int P12 = pv42.P1(), P22 = pv42.P2(), P32 = pv42.P3(),
                SS2 = pv42.SS(), TT2 = pv42.TT();

            const auto cgcU4 = wigU4SU2SU2db.get_U4SU2SU2(
                                  P11, P21, P31, SS1, TT1,
                                  P12, P22, P32, SS2, TT2,
                                  P1(), P2(), P3(), SS(), TT());
            if (cgcU4.empty()) continue;

            for (int JJ1 = LL1 + SS1; JJ1 >= abs(LL1 - SS1); JJ1 -= 2)
            {
              for (int MMJ1 = JJ1; MMJ1 >= -JJ1; MMJ1 -= 2)
              {
                int MMJ2 = MMJ() - MMJ1;

                for (int MMT1 = TT1; MMT1 >= -TT1; MMT1 -= 2)
                {
                  int MMT2 = MMT() - MMT1;
                  if (abs(MMT2) > TT2) continue;
                  double cgcT = cgcs.get({TT1, TT2, TT(), MMT1, MMT2, MMT()});

                  U3xU4::Tad_phys
                    Tad(n, Omega_ads(), LL1, SS1, TT1, MMT1, JJ1, MMJ1, labels_);

                  std::vector<Eigen::RowVectorXd> coeffs1;
                  std::vector<U3xU4::ads_phys> adArr;

                  for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                  {
                    if (u2::mult(JJ1, JJ2, JJ_) == 0) continue;
                    if (abs(MMJ2) > JJ2) continue;

                    double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                    double U9j = racah9.get({LL1, LL2, LL(),
                                              SS1, SS2, SS(),
                                              JJ1, JJ2, JJ()});

                    double cgcU2 = U9j * cgcJ * cgcT;

                    if (coeffs1.empty())
                      std::tie(coeffs1, adArr) = Tad.expand();

                    U3xU4::Tat_phys Tat(n, Omega_ats(), LL2, SS2, TT2, MMT2, JJ2,
                                          MMJ2, labels_);

                    auto [coeffs2, atArr] = Tat.expand();

                    for (size_t i = 0; i < coeffs1.size(); i++)
                    {
                      for (size_t j = 0; j < coeffs2.size(); j++)
                      {
                        std::array<int, 4> ns, lls, jjs, mmjs, mmts;
                        // no phase, no permutation
                        for (int o = 0; o < 2; o++)
                        {
                          std::tie(ns[o], lls[o], jjs[o], mmjs[o], mmts[o]) =
                            adArr[i][o].getLabels();

                          std::tie(ns[o + 2], lls[o + 2], jjs[o + 2], mmjs[o + 2], mmts[o + 2]) =
                            atArr[j][o].getLabels();
                        }

                        double me = TBME.getME(
                          ns[0], lls[0], jjs[0], mmjs[0], mmts[0],
                          ns[1], lls[1], jjs[1], mmjs[1], mmts[1],
                          ns[3], lls[3], jjs[3], mmjs[3], mmts[3],
                          ns[2], lls[2], jjs[2], mmjs[2], mmts[2]) / 4.0;

                        if (std::fabs(me) < 1e-10) continue;

                        for (int rho = 0; rho < rhomax; rho++)
                          for (int eta = 0; eta < etamax; eta++)
                            for (int k = 0; k < kappamax_; k++)
                              for (int z = 0; z < zetamax_; z++)
                              {
                                int index =
                                      ((rho * etamax + eta) * kappamax_ + k) *
                                          zetamax_ +
                                      z;
                                for (int k1 = 0; k1 < kappa1; k1++)
                                  for (int k2 = 0; k2 < kappa2; k2++)
                                    strengths[index] += me *
                                      cgcU3(k1 , k2, k, rho) *
                                      cgcU4(0, 0, z, eta) * cgcU2 *
                                      coeffs1[i](k1) * coeffs2[j](k2);
                              }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    return strengths;
*/
  }

  // get two-body strength directly during expand()
  std::vector<double> SingleShellTensor_phys::getStrengths(
        const HO::TwBME_general& TBME, SSTDecomp_phys_db& db) const
  {
    assert(num_ads() == 2 && num_ats() == 2);

    int ncols = rho0max() * eta0max() * kappamax_ * zetamax_;
    std::vector<double> strengths(ncols, 0.0);

    // The cached expansion already carries the full cgcU3*cgcU4*cgcU2 *
    // coeffs1*coeffs2 contraction in coeffs[t]; the strength is just the sum
    // over expansion terms of me(t) * coeffs[t]. (Equivalent to the fused
    // version below, up to expand()'s 1e-12 norm filtering of null terms.)
    // Zero-copy borrowed view: coeff rows are flat double arrays and the ad/at
    // operators are stored as Table_nljmjmt indices (turned into labels on
    // demand) -- no Eigen row allocation, no Record copy, no ad_phys rebuild.
    auto e = db.get_view(*this);
    assert(static_cast<int>(e->ncols()) == ncols);

    for (size_t t = 0; t < e->rows(); t++)
    {
      // operators are stored ad's-then-a's, so slots 0,1 are ad's and 2,3 are
      // a's (no phase, no permutation); labels come straight from the table.
      std::array<int, 4> ns, lls, jjs, mmjs, mmts;
      for (int k = 0; k < 4; k++)
        std::tie(ns[k], lls[k], jjs[k], mmjs[k], mmts[k]) = e->op_labels(t, k);

      double me = TBME.getME(
        ns[0], lls[0], jjs[0], mmjs[0], mmts[0],
        ns[1], lls[1], jjs[1], mmjs[1], mmts[1],
        ns[3], lls[3], jjs[3], mmjs[3], mmts[3],
        ns[2], lls[2], jjs[2], mmjs[2], mmts[2]) / 4.0;

      if (std::fabs(me) < 1e-10) continue;

      const double* row = e->coeff_row(t);
      for (int c = 0; c < ncols; c++)
        strengths[c] += me * row[c];
    }

    return strengths;
  }


} // namespace U3xU4
