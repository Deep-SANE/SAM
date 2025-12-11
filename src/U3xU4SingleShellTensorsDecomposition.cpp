#include <cassert>
#include <stdexcept>
#include <utility>

#include <Databases.h>
#include <U3xU4SingleShellTensors.h>

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

    const auto& g3_conj = sps3_conj[c];
    const auto& g4_conj = sps4_conj[r];

    int n13b = g3_conj.n13(), n22b = g3_conj.n22(),
        n11b = g3_conj.n11(), n33p = g4_conj.n33(),
        n22p = g4_conj.n22(), n11p = g4_conj.n11();

    u3::gelfand g3(n13b, 0, 0, n13b - n22b, 0, n13b - n11b);
    c = qcx_utils::itou8(u3::findGelfand(g3, sps3));

    u4::gelfand g4(1, 0, 0, 0, 1 - n33p, 0, 0, 1 - n22p, 0, 1 - n11p);
    r = qcx_utils::itou8(u4::findGelfand(g4, sps4));

    int phase = n22b + n11b + n33p + n22p + n11p;

    tensor = false;

    return (phase % 2 == 0) ? 1 : -1;
  }

  double at_phys::turnToAnnihilation()
  {
    auto [n, ll, jj, mmj, mmt] = labels.getLabels(index);
#ifdef ADD_SU3LIB_PHASE_U3SO3
    int phi = (2 * n + 1 - jj - mmj - mmt) / 2;
#else
    int phi = (n + ll / 2 + 1 - jj - mmj - mmt) / 2;
#endif
    mmj *= -1;
    mmt *= -1;
    tensor = false;
    index = labels.getIndex(n, ll, jj, mmj, mmt);
    return (phi % 2 == 0) ? 1.0 : -1.0;
  }

  /////////////// PRODUCT OF CREATION OPERATORS/TENSORS /////////////////

  // expand coupled product of creation tensors (ad's)
  // in terms of creation operators (no phase needed)
  std::pair<std::vector<double>, std::vector<U3xU4::ads_gelfand>>
  Tad_gelfand::expand(sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& wigU3U2db = dbs.wigU3U2db;
    auto& wigU4U3db = dbs.wigU4U3db;

    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps4: Gelfand array for [1,0,0,0]
    */

    std::vector<double> coeffs;
    std::vector<U3xU4::ads_gelfand> tensorsArr;

    auto num = num_ads();

    // one ad
    if (num == 1)
    {
      coeffs.push_back(1.0);
      int c = u3::findGelfand(g3, sps3); 
      int r = u4::findGelfand(g4, sps4);
      U3xU4::ad_gelfand ad(c, r, sps3, sps4);
      //U3xU4::ads_gelfand Tad(1, ad);
      tensorsArr.push_back({ad});
    }

    // two ad's
    else if (num == 2)
    {
      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3, sps3, sps3);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4, sps4, sps4);

      int jj3 = g3.Lambdax2(), mm3 = g3.MLambdax2(), 
          jj4 = g4.Lambdax2(), mm4 = g4.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        int g3_1 = uncouple_g3[i].first;
        const auto& sp3_1 = sps3[g3_1];

        int g3_2 = uncouple_g3[i].second;
        const auto& sp3_2 = sps3[g3_2];

        int jj3_1 = sp3_1.Lambdax2(), mm3_1 = sp3_1.MLambdax2(),
            jj3_2 = sp3_2.Lambdax2(), mm3_2 = sp3_2.MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
          sp3_1.n13(), sp3_1.n23(), sp3_1.n33(), sp3_1.n12(), sp3_1.n22(), 
          sp3_2.n13(), sp3_2.n23(), sp3_2.n33(), sp3_2.n12(), sp3_2.n22(), 
          g3.n13(), g3.n23(), g3.n33(), g3.n12(), g3.n22()); 
        // accU3 should have only one element

        if (allowU3 == false) continue;

        double cgc3 = cgcs.compute(jj3_1, jj3_2, jj3, mm3_1, mm3_2, mm3);

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          int g4_1 = uncouple_g4[j].first;
          const auto& sp4_1 = sps4[g4_1];

          int g4_2 = uncouple_g4[j].second;
          const auto& sp4_2 = sps4[g4_2];

          int jj4_1 = sp4_1.Lambdax2(), mm4_1 = sp4_1.MLambdax2(),
              jj4_2 = sp4_2.Lambdax2(), mm4_2 = sp4_2.MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
            sp4_1.n14(), sp4_1.n24(), sp4_1.n34(), sp4_1.n44(), 
            sp4_1.n13(), sp4_1.n23(), sp4_1.n33(), 
            sp4_2.n14(), sp4_2.n24(), sp4_2.n34(), sp4_2.n44(), 
            sp4_2.n13(), sp4_2.n23(), sp4_2.n33(), 
            g4.n14(), g4.n24(), g4.n34(), g4.n44(), 
            g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false) continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
            sp4_1.n13(), sp4_1.n23(), sp4_1.n33(), sp4_1.n12(), sp4_1.n22(), 
            sp4_2.n13(), sp4_2.n23(), sp4_2.n33(), sp4_2.n12(), sp4_2.n22(), 
            g4.n13(), g4.n23(), g4.n33(), g4.n12(), g4.n22());

          if (allowU3_4 == false) continue;

          double cgc4 = cgcs.compute(jj4_1, jj4_2, jj4, mm4_1, mm4_2, mm4);

          double cgcU4 = 0.0;
          for (size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15) continue;

          U3xU4::ad_gelfand ad1(g3_1, g4_1, sps3, sps4),
                            ad2(g3_2, g4_2, sps3, sps4);
          U3xU4::ads_gelfand adad = {std::move(ad1), std::move(ad2)};

          coeffs.push_back(cgcU3 * cgcU4);
          tensorsArr.push_back(std::move(adad));
        }
      }
    }

    // three ad's
    else if (num == 3)
    {
      std::vector<u3::gelfand> g3Arr0 =
          u3::gelfandBasis(Omegas[0].u3());
      std::vector<u4::gelfand> g4Arr0 =
          u4::gelfandBasis(Omegas[0].u4());

      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3, g3Arr0, sps3);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4, g4Arr0, sps4);

      int jj3 = g3.Lambdax2(), mm3 = g3.MLambdax2(), 
          jj4 = g4.Lambdax2(), mm4 = g4.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        const auto& G3_0 = g3Arr0[uncouple_g3[i].first];

        int g3_3 = uncouple_g3[i].second;
        const auto& sp3_3 = sps3[g3_3];

        int jj3_0 = G3_0.Lambdax2(), mm3_0 = G3_0.MLambdax2(),
            jj3_3 = sp3_3.Lambdax2(), mm3_3 = sp3_3.MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
          G3_0.n13(), G3_0.n23(), G3_0.n33(), G3_0.n12(), G3_0.n22(), 
          sp3_3.n13(), sp3_3.n23(), sp3_3.n33(), sp3_3.n12(), sp3_3.n22(), 
          g3.n13(), g3.n23(), g3.n33(), g3.n12(), g3.n22()); 
        // accU3 should have only one element

        if (allowU3 == false) continue;

        double cgc3 = cgcs.compute(jj3_0, jj3_3, jj3, mm3_0, mm3_3, mm3);

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          const auto& G4_0 = g4Arr0[uncouple_g4[j].first];

          int g4_3 = uncouple_g4[j].second;
          const auto& sp4_3 = sps4[g4_3];

          int jj4_0 = G4_0.Lambdax2(), mm4_0 = G4_0.MLambdax2(),
              jj4_3 = sp4_3.Lambdax2(), mm4_3 = sp4_3.MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
            G4_0.n14(), G4_0.n24(), G4_0.n34(), G4_0.n44(), 
            G4_0.n13(), G4_0.n23(), G4_0.n33(), 
            sp4_3.n14(), sp4_3.n24(), sp4_3.n34(), sp4_3.n44(), 
            sp4_3.n13(), sp4_3.n23(), sp4_3.n33(), 
            g4.n14(), g4.n24(), g4.n34(), g4.n44(), 
            g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false) continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
              G4_0.n13(), G4_0.n23(), G4_0.n33(), G4_0.n12(), G4_0.n22(), 
              sp4_3.n13(), sp4_3.n23(), sp4_3.n33(), sp4_3.n12(), sp4_3.n22(), 
              g4.n13(), g4.n23(), g4.n33(), g4.n12(), g4.n22());

          if (allowU3_4 == false) continue;

          double cgc4 = cgcs.compute(jj4_0, jj4_3, jj4, mm4_0, mm4_3, mm4);

          double cgcU4 = 0.0;
          for (size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15) continue;

          U3xU4::Tad_gelfand Tad(n, sps3, sps4, {Omegas[0]}, G3_0, G4_0);
          auto [coeffs_0, tensors_0] = Tad.expand(dbs);

          U3xU4::ad_gelfand ad3(g3_3, g4_3, sps3, sps4);

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
  std::pair<std::vector<Eigen::RowVectorXd>, // each element has size == kappa
            std::vector<U3xU4::ads_phys>>
  Tad_phys::expand(sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    std::vector<Eigen::RowVectorXd> coeffsArr;
    std::vector<U3xU4::ads_phys> tensorsArr;

    auto num = num_ads();
    int8_t ll_max = 2 * n, ll_min = 2 * (n % 2);

    // one ad
    if (num == 1)
    {
      Eigen::RowVectorXd c(1);
      c(0) = 1.0;
      coeffsArr.push_back(c);
      U3xU4::ad_phys ad(n, LL, JJ, MMJ, MMT, labels);
      //U3xU4::ads_phys Tad(1, ad);
      tensorsArr.push_back({ad});
    }

    // two ad's
    else if (num == 2)
    {
      int lam = Omegas[0].lam(), mu = Omegas[0].mu(),
          P1 = Omegas[0].P1(), P2 = Omegas[0].P2(), P3 = Omegas[0].P3();

      for (int8_t lla = ll_max; lla >= ll_min; lla -= 4)
      {
        for (int8_t llb = ll_max; llb >= ll_min; llb -= 4)
        {
          if (std::abs(lla - llb) > LL || (lla + llb) < LL)
            continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(
            n, 0, lla / 2, n, 0, llb / 2, lam, mu, LL / 2);
          if (allowU3 == false) continue;

          auto [allowU4, cgcU4] = wigU4SU2SU2db.get(
            1, 0, 0, 1, 1, 1, 0, 0, 1, 1, P1, P2, P3, SS, TT);
          if (allowU4 == false) continue;

          for (int8_t jja = lla + 1; jja >= std::abs(lla - 1); jja -= 2)
          {
            for (int8_t jjb = llb + 1; jjb >= std::abs(llb - 1); jjb -= 2)
            {
              if (std::abs(jja - jjb) > JJ || (jja + jjb) < JJ) continue;

              double U9j = racah9.compute(lla, llb, LL, 
                                          1, 1, SS, 
                                          jja, jjb, JJ);

              for (int8_t mmja = jja; mmja >= -jja; mmja -= 2)
              {
                int8_t mmjb = MMJ - mmja;
                if (std::abs(mmjb) > jjb) continue;

                for (int8_t mmta = 1; mmta >= -1; mmta -= 2)
                {
                  int8_t mmtb = MMT - mmta;
                  if (std::abs(mmtb) > 1) continue;

                  double cgcJ = cgcs.compute(jja, jjb, JJ, mmja, mmjb, MMJ),
                         cgcT = cgcs.compute(1, 1, TT, mmta, mmtb, MMT),
                         factor = cgcU4(0, 0) * U9j * cgcJ * cgcT;

                  Eigen::RowVectorXd coeffs(kappamax);
                  coeffs = cgcU3.col(0) * factor;
                  // coeffs.setZero();
                  // for (int k = 0; k < kappamax; k++)
                  //   coeffs(k) += cgcU3(k, 0) * factor;

                  if (coeffs.norm() < 1e-12) continue;

                  coeffsArr.push_back(std::move(coeffs));

                  U3xU4::ad_phys ad1(n, lla, jja, mmja, mmta, labels),
                                 ad2(n, llb, jjb, mmjb, mmtb, labels);
                  U3xU4::ads_phys Tad = {std::move(ad1), std::move(ad2)};
                  tensorsArr.push_back(std::move(Tad));
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
        pv3s0 = u3::physicalBasis(Omegas[0].u3());
      std::vector<u4::physicalvector> 
        pv4s0 = u4::physicalBasis(Omegas[0].u4());

      int lam = Omegas[1].lam(), mu = Omegas[1].mu(),
          P1 = Omegas[1].P1(), P2 = Omegas[1].P2(), P3 = Omegas[1].P3(),
          lam0 = Omegas[0].lam(), mu0 = Omegas[0].mu(),
          P10 = Omegas[0].P1(), P20 = Omegas[0].P2(), P30 = Omegas[0].P3();

      for (int8_t llc = ll_max; llc >= ll_min; llc -= 4)
      {
        for (int pv30 = 0; pv30 < pv3s0.size(); pv30++)
        {
          int8_t LL0 = 2 * pv3s0[pv30].L;

          if (std::abs(LL0 - llc) > LL || (LL0 + llc) < LL)
            continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(
            lam0, mu0, LL0 / 2, n, 0, llc / 2, lam, mu, LL / 2);
          if (allowU3 == false) continue;

          for (int pv40 = 0; pv40 < pv4s0.size(); pv40++)
          {
            int8_t SS0 = pv4s0[pv40].SS, TT0 = pv4s0[pv40].TT;

            if (std::abs(SS0 - 1) > SS || (SS0 + 1) < SS ||
                std::abs(TT0 - 1) > TT || (TT0 + 1) < TT)
              continue;

            auto [allowU4, cgcU4] = wigU4SU2SU2db.get(
                P10, P20, P30, SS0, TT0, 1, 0, 0, 1, 1, P1, P2, P3, SS, TT);
            if (allowU4 == false) continue;

            for (int8_t jjc = llc + 1; jjc >= std::abs(llc - 1); jjc -= 2)
            {
              for (int8_t JJ0 = LL0 + SS0; JJ0 >= std::abs(LL0 - SS0); JJ0 -= 2)
              {
                if (std::abs(JJ0 - jjc) > JJ || (JJ0 + jjc) < JJ) continue;

                double U9j = racah9.compute(LL0, llc, LL, 
                                            SS0, 1, SS, 
                                            JJ0, jjc, JJ);

                for (int8_t mmjc = jjc; mmjc >= -jjc; mmjc -= 2)
                {
                  int8_t MMJ0 = MMJ - mmjc;
                  if (std::abs(MMJ0) > JJ0) continue;

                  for (int8_t mmtc = 1; mmtc >= -1; mmtc -= 2)
                  {
                    int8_t MMT0 = MMT - mmtc;
                    if (std::abs(MMT0) > TT0) continue;

                    double cgcJ = cgcs.compute(JJ0, jjc, JJ, MMJ0, mmjc, MMJ),
                           cgcT = cgcs.compute(TT0, 1, TT, MMT0, mmtc, MMT),
                           factor = cgcU4(0, 0) * U9j * cgcJ * cgcT;

                    U3xU4::Tad_phys Tad0(n, {Omegas[0]}, LL0, SS0, TT0, MMT0, JJ0,
                                         MMJ0, labels);
                    auto [coeffsArr0, adArr0] = Tad0.expand(dbs);

                    U3xU4::ad_phys ad3(n, llc, jjc, mmjc, mmtc, labels);

                    for (int i = 0; i < coeffsArr0.size(); i++)
                    {
                      Eigen::RowVectorXd coeffs(kappamax);
                      coeffs.setZero();
                      for (int k = 0; k < kappamax; k++)
                        for (int k0 = 0; k0 < Tad0.kappamax; k0++)
                          coeffs(k) += cgcU3(k0 * kappamax + k, 0) 
                                     * factor * coeffsArr0[i](k0);

                      if (coeffs.norm() < 1e-12) continue;

                      coeffsArr.push_back(std::move(coeffs));

                      U3xU4::ads_phys adad = adArr0[i];
                      adad.push_back(ad3);
                      tensorsArr.push_back(std::move(adad));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    return std::make_pair(std::move(coeffsArr), std::move(tensorsArr));
  }

  /////////////// PRODUCT OF ANNIHILATION OPERATORS/TENSORS ///////////////

  // expand coupled product of annihilation tensors (at's)
  // in terms of annihilation operators (a's) via a phase
  std::pair<std::vector<double>, std::vector<U3xU4::ats_gelfand>>
  Tat_gelfand::expand(sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& wigU3U2db = dbs.wigU3U2db;
    auto& wigU4U3db = dbs.wigU4U3db;

    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps3_conj: Gelfand array for [n,n,0]

    sps4: Gelfand array for [1,0,0,0]
    sps4_conj: Gelfand array for [1,1,1,0]
    */

    std::vector<double> coeffs;
    std::vector<std::vector<U3xU4::at_gelfand>> tensorsArr;

    auto num = num_ats();

    // one at
    if (num == 1)
    {
      auto c = u3::findGelfand(g3, sps3_conj),
           r = u4::findGelfand(g4, sps4_conj);
      U3xU4::at_gelfand at(c, r, sps3, sps3_conj, sps4, sps4_conj);
      auto phase = at.turnToAnnihilation();

      assert(!at.isTensor());

      coeffs.push_back(phase);
      tensorsArr.push_back({at});
    }

    // two at's
    else if (num == 2)
    {
      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3, sps3_conj, sps3_conj);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4, sps4_conj, sps4_conj);

      int jj3 = g3.Lambdax2(), mm3 = g3.MLambdax2(), 
          jj4 = g4.Lambdax2(), mm4 = g4.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        int g3_1 = uncouple_g3[i].first;
        const auto& sp3_1 = sps3_conj[g3_1];

        int g3_2 = uncouple_g3[i].second;
        const auto& sp3_2 = sps3_conj[g3_2];

        int jj3_1 = sp3_1.Lambdax2(), mm3_1 = sp3_1.MLambdax2(),
            jj3_2 = sp3_2.Lambdax2(), mm3_2 = sp3_2.MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
          sp3_1.n13(), sp3_1.n23(), sp3_1.n33(), sp3_1.n12(), sp3_1.n22(), 
          sp3_2.n13(), sp3_2.n23(), sp3_2.n33(), sp3_2.n12(), sp3_2.n22(), 
          g3.n13(), g3.n23(), g3.n33(), g3.n12(), g3.n22()); 
        // accU3 should have only one element

        if (allowU3 == false) continue;

        double cgc3 = cgcs.compute(jj3_1, jj3_2, jj3, mm3_1, mm3_2, mm3);

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          int g4_1 = uncouple_g4[j].first;
          const auto& sp4_1 = sps4_conj[g4_1];

          int g4_2 = uncouple_g4[j].second;
          const auto& sp4_2 = sps4_conj[g4_2];

          int jj4_1 = sp4_1.Lambdax2(), mm4_1 = sp4_1.MLambdax2(),
              jj4_2 = sp4_2.Lambdax2(), mm4_2 = sp4_2.MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
            sp4_1.n14(), sp4_1.n24(), sp4_1.n34(), sp4_1.n44(),
            sp4_1.n13(), sp4_1.n23(), sp4_1.n33(), 
            sp4_2.n14(), sp4_2.n24(), sp4_2.n34(), sp4_2.n44(), 
            sp4_2.n13(), sp4_2.n23(), sp4_2.n33(), 
            g4.n14(), g4.n24(), g4.n34(), g4.n44(), 
            g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false) continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
            sp4_1.n13(), sp4_1.n23(), sp4_1.n33(), sp4_1.n12(), sp4_1.n22(), 
            sp4_2.n13(), sp4_2.n23(), sp4_2.n33(), sp4_2.n12(), sp4_2.n22(),
            g4.n13(), g4.n23(), g4.n33(), g4.n12(), g4.n22());

          if (allowU3_4 == false) continue;

          double cgc4 = cgcs.compute(jj4_1, jj4_2, jj4, mm4_1, mm4_2, mm4);

          double cgcU4 = 0.0;
          for (size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15) continue;

          U3xU4::at_gelfand at1(g3_1, g4_1, sps3, sps3_conj, sps4, sps4_conj),
                            at2(g3_2, g4_2, sps3, sps3_conj, sps4, sps4_conj);
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
      std::vector<u3::gelfand> 
        g3Arr0 = u3::gelfandBasis(Omegas[0].u3());
      std::vector<u4::gelfand> 
        g4Arr0 = u4::gelfandBasis(Omegas[0].u4());

      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3, g3Arr0, sps3_conj);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4, g4Arr0, sps4_conj);

      int jj3 = g3.Lambdax2(), mm3 = g3.MLambdax2(), 
          jj4 = g4.Lambdax2(), mm4 = g4.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        const auto& G3_0 = g3Arr0[uncouple_g3[i].first];

        int g3_3 = uncouple_g3[i].second;
        const auto& sp3_3 = sps3_conj[g3_3];

        int jj3_0 = G3_0.Lambdax2(), mm3_0 = G3_0.MLambdax2(),
            jj3_3 = sp3_3.Lambdax2(), mm3_3 = sp3_3.MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
          G3_0.n13(), G3_0.n23(), G3_0.n33(), G3_0.n12(), G3_0.n22(), 
          sp3_3.n13(), sp3_3.n23(), sp3_3.n33(), sp3_3.n12(), sp3_3.n22(), 
          g3.n13(), g3.n23(), g3.n33(), g3.n12(), g3.n22()); 
        // accU3 should have only one element

        if (allowU3 == false) continue;

        double cgc3 = cgcs.compute(jj3_0, jj3_3, jj3, mm3_0, mm3_3, mm3);

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          const auto& G4_0 = g4Arr0[uncouple_g4[j].first];

          int g4_3 = uncouple_g4[j].second;
          const auto& sp4_3 = sps4_conj[g4_3];

          int jj4_0 = G4_0.Lambdax2(), mm4_0 = G4_0.MLambdax2(),
              jj4_3 = sp4_3.Lambdax2(), mm4_3 = sp4_3.MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
            G4_0.n14(), G4_0.n24(), G4_0.n34(), G4_0.n44(), 
            G4_0.n13(), G4_0.n23(), G4_0.n33(), 
            sp4_3.n14(), sp4_3.n24(), sp4_3.n34(), sp4_3.n44(),
            sp4_3.n13(), sp4_3.n23(), sp4_3.n33(), 
            g4.n14(), g4.n24(), g4.n34(), g4.n44(),
            g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false) continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
            G4_0.n13(), G4_0.n23(), G4_0.n33(), G4_0.n12(), G4_0.n22(), 
            sp4_3.n13(), sp4_3.n23(), sp4_3.n33(), sp4_3.n12(), sp4_3.n22(), 
            g4.n13(), g4.n23(), g4.n33(), g4.n12(), g4.n22());

          if (allowU3_4 == false) continue;

          double cgc4 = cgcs.compute(jj4_0, jj4_3, jj4, mm4_0, mm4_3, mm4);

          double cgcU4 = 0.0;
          for (size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15) continue;

          U3xU4::at_gelfand at3(g3_3, g4_3, sps3, sps3_conj, sps4, sps4_conj);
          auto phase3 = at3.turnToAnnihilation();

          assert(!at3.isTensor());

          U3xU4::Tat_gelfand Tat(n, sps3, sps3_conj, sps4, sps4_conj, 
                                 {Omegas[0]}, G3_0, G4_0);

          auto [coeffs_0, tensors_0] = Tat.expand(dbs);

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
  std::pair<std::vector<Eigen::RowVectorXd>, // each element has size == kappa
            std::vector<U3xU4::ats_phys>>
  Tat_phys::expand(sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    std::vector<Eigen::RowVectorXd> coeffsArr;
    std::vector<U3xU4::ats_phys> tensorsArr;

    auto num = num_ats();
    int8_t ll_max = 2 * n, ll_min = 2 * (n % 2);

    // one at
    if (num == 1)
    {
      Eigen::RowVectorXd c(1);
      U3xU4::at_phys at(n, LL, JJ, MMJ, MMT, labels);
      c(0) = at.turnToAnnihilation();
      coeffsArr.push_back(c);
      tensorsArr.push_back({at});
    }

    // two at's
    else if (num == 2)
    {
      int lam = Omegas[0].lam(), mu = Omegas[0].mu(),
          P1 = Omegas[0].P1(), P2 = Omegas[0].P2(), P3 = Omegas[0].P3();

      for (int8_t lla = ll_max; lla >= ll_min; lla -= 4)
      {
        for (int8_t llb = ll_max; llb >= ll_min; llb -= 4)
        {
          if (std::abs(lla - llb) > LL || (lla + llb) < LL) continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(
            0, n, lla / 2, 0, n, llb / 2, lam, mu, LL / 2);
          if (allowU3 == false) continue;

          auto [allowU4, cgcU4] = wigU4SU2SU2db.get(
            0, 0, 1, 1, 1, 0, 0, 1, 1, 1, P1, P2, P3, SS, TT);
          if (allowU4 == false) continue;

          for (int8_t jja = lla + 1; jja >= std::fabs(lla - 1); jja -= 2)
          {
            for (int8_t jjb = llb + 1; jjb >= std::fabs(llb - 1); jjb -= 2)
            {
              if (std::abs(jja - jjb) > JJ || (jja + jjb) < JJ) continue;

              double U9j = racah9.compute(lla, llb, LL, 
                                          1, 1, SS, 
                                          jja, jjb, JJ);

              for (int8_t mmja = jja; mmja >= -jja; mmja -= 2)
              {
                int8_t mmjb = MMJ - mmja;
                if (std::abs(mmjb) > jjb) continue;

                for (int8_t mmta = 1; mmta >= -1; mmta -= 2)
                {
                  int8_t mmtb = MMT - mmta;
                  if (std::abs(mmtb) > 1) continue;

                  double cgcJ = cgcs.compute(jja, jjb, JJ, mmja, mmjb, MMJ),
                         cgcT = cgcs.compute(1, 1, TT, mmta, mmtb, MMT);

                  U3xU4::at_phys at1(n, lla, jja, mmja, mmta, labels),
                                 at2(n, llb, jjb, mmjb, mmtb, labels);
                  double phase1 = at1.turnToAnnihilation(),
                         phase2 = at2.turnToAnnihilation(),
                         factor =
                             cgcU4(0, 0) * U9j * cgcJ * cgcT * phase1 * phase2;

                  Eigen::RowVectorXd coeffs(kappamax);
                  coeffs = cgcU3.col(0) * factor;
                  // coeffs.setZero();
                  // for (int k = 0; k < kappamax; k++)
                  //   coeffs(k) += cgcU3(k, 0) * factor;

                  if (coeffs.norm() < 1e-12) continue;

                  coeffsArr.push_back(std::move(coeffs));

                  assert(!at1.isTensor());
                  assert(!at2.isTensor());

                  U3xU4::ats_phys Tat = {std::move(at1), std::move(at2)};
                  tensorsArr.push_back(std::move(Tat));
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
        pv3s0 = u3::physicalBasis(Omegas[0].u3());
      std::vector<u4::physicalvector> 
        pv4s0 = u4::physicalBasis(Omegas[0].u4());

      int lam = Omegas[1].lam(), mu = Omegas[1].mu(),
          P1 = Omegas[1].P1(), P2 = Omegas[1].P2(), P3 = Omegas[1].P3(),
          lam0 = Omegas[0].lam(), mu0 = Omegas[0].mu(),
          P10 = Omegas[0].P1(), P20 = Omegas[0].P2(), P30 = Omegas[0].P3();

      for (int8_t llc = ll_max; llc >= ll_min; llc -= 4)
      {
        for (int pv30 = 0; pv30 < pv3s0.size(); pv30++)
        {
          int8_t LL0 = 2 * pv3s0[pv30].L;

          if (std::abs(LL0 - llc) > LL || (LL0 + llc) < LL) continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(
            lam0, mu0, LL0 / 2, 0, n, llc / 2, lam, mu, LL / 2);
          if (allowU3 == false) continue;

          for (int pv40 = 0; pv40 < pv4s0.size(); pv40++)
          {
            int8_t SS0 = pv4s0[pv40].SS, TT0 = pv4s0[pv40].TT;

            if (std::abs(SS0 - 1) > SS || (SS0 + 1) < SS ||
                std::abs(TT0 - 1) > TT || (TT0 + 1) < TT)
              continue;

            auto [allowU4, cgcU4] = wigU4SU2SU2db.get(
              P10, P20, P30, SS0, TT0, 0, 0, 1, 1, 1, P1, P2, P3, SS, TT);
            if (allowU4 == false) continue;

            for (int8_t jjc = llc + 1; jjc >= std::abs(llc - 1); jjc -= 2)
            {
              for (int8_t JJ0 = LL0 + SS0; JJ0 >= std::abs(LL0 - SS0); JJ0 -= 2)
              {
                if (std::abs(JJ0 - jjc) > JJ || (JJ0 + jjc) < JJ) continue;

                double U9j = racah9.compute(LL0, llc, LL, 
                                            SS0, 1, SS, 
                                            JJ0, jjc, JJ);

                for (int8_t mmjc = jjc; mmjc >= -jjc; mmjc -= 2)
                {
                  int8_t MMJ0 = MMJ - mmjc;
                  if (std::abs(MMJ0) > JJ0) continue;

                  for (int8_t mmtc = 1; mmtc >= -1; mmtc -= 2)
                  {
                    int8_t MMT0 = MMT - mmtc;
                    if (std::abs(MMT0) > TT0) continue;

                    double cgcJ = cgcs.compute(JJ0, jjc, JJ, MMJ0, mmjc, MMJ),
                           cgcT = cgcs.compute(TT0, 1, TT, MMT0, mmtc, MMT),
                           factor = cgcU4(0, 0) * U9j * cgcJ * cgcT;

                    U3xU4::Tat_phys Tat0(n, {Omegas[0]}, LL0, SS0, TT0, MMT0, JJ0,
                                         MMJ0, labels);

                    auto [coeffsArr0, aArr0] = Tat0.expand(dbs);

                    U3xU4::at_phys at3(n, llc, jjc, mmjc, mmtc, labels);
                    double phase3 = at3.turnToAnnihilation();
                    factor *= phase3;

                    for (int i = 0; i < coeffsArr0.size(); i++)
                    {                    
                      Eigen::RowVectorXd coeffs(kappamax);
                      coeffs.setZero();
                      for (int k = 0; k < kappamax; k++)
                        for (int k0 = 0; k0 < Tat0.kappamax; k0++)
                          coeffs(k) += cgcU3(k0 * kappamax + k, 0) 
                                     * factor * coeffsArr0[i](k0);

                      if (coeffs.norm() < 1e-12) continue;

                      coeffsArr.push_back(std::move(coeffs));

                      U3xU4::ats_phys atat = aArr0[i];
                      atat.push_back(at3);
                      tensorsArr.push_back(std::move(atat));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    return std::make_pair(std::move(coeffsArr), std::move(tensorsArr));
  }

  ////// PRODUCT OF CREATION AND ANNIHILATION OPERATORS/TENSORS ///////

  // expand coupled product of creation and annihilation tensors
  // in terms of creation and annihilation operators (ad's and a's)
  std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
             std::vector<U3xU4::ads_gelfand>, // ad's
             std::vector<U3xU4::ats_gelfand>> // a's
  SingleShellTensor_gelfand::expand(sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& wigU3U2db = dbs.wigU3U2db;
    auto& wigU4U3db = dbs.wigU4U3db;

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
    if (ats == 0)
    {
      U3xU4::Tad_gelfand Tad(n, sps3, sps4, Omega_ads, g3, g4);

      auto [coeffs_, adsArr_] = Tad.expand(dbs);

      for (size_t i = 0; i < coeffs_.size(); i++)
      {
        Eigen::RowVectorXd x(1);
        x(0) = coeffs_[i];
        coeffs.push_back(x);
        adsArr.push_back(std::move(adsArr_[i]));
      }
    }

    // only ats
    else if (ads == 0)
    {
      U3xU4::Tat_gelfand Tat(n, sps3, sps3_conj, sps4, sps4_conj, 
                             Omega_ats, g3, g4);

      auto [coeffs_, atsArr_] = Tat.expand(dbs);

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
      std::vector<u3::gelfand> g3Arr1, g3Arr2;
      std::vector<u4::gelfand> g4Arr1, g4Arr2;

      if (ads > 1)
      {
        g3Arr1 = u3::gelfandBasis(Omega_ads.back().u3());
        g4Arr1 = u4::gelfandBasis(Omega_ads.back().u4());
      }
      else
      {
        g3Arr1 = sps3;
        g4Arr1 = sps4;
      }

      if (ats > 1)
      {
        g3Arr2 = u3::gelfandBasis(Omega_ats.back().u3());
        g4Arr2 = u4::gelfandBasis(Omega_ats.back().u4());
      }
      else
      {
        g3Arr2 = sps3_conj;
        g4Arr2 = sps4_conj;
      }

      // uncouple the final tensor to Tad and Tat
      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3, g3Arr1, g3Arr2);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4, g4Arr1, g4Arr2);

      int jj3 = g3.Lambdax2(), mm3 = g3.MLambdax2(), jj4 = g4.Lambdax2(),
          mm4 = g4.MLambdax2();

      for (size_t i = 0; i < uncouple_g3.size(); i++)
      {
        const auto& g3_1 = g3Arr1[uncouple_g3[i].first];
        const auto& g3_2 = g3Arr2[uncouple_g3[i].second];

        int jj3_1 = g3_1.Lambdax2(), mm3_1 = g3_1.MLambdax2(),
            jj3_2 = g3_2.Lambdax2(), mm3_2 = g3_2.MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
          g3_1.n13(), g3_1.n23(), g3_1.n33(), g3_1.n12(), g3_1.n22(), 
          g3_2.n13(), g3_2.n23(), g3_2.n33(), g3_2.n12(), g3_2.n22(), 
          g3.n13(), g3.n23(), g3.n33(), g3.n12(), g3.n22()); 
        // accU3 should have only rhomax elements

        if (allowU3 == false) continue;

        double cgc3 = cgcs.compute(jj3_1, jj3_2, jj3, mm3_1, mm3_2, mm3);

        auto cgcU3 = accU3 * cgc3;

        if (cgcU3.norm() <= 1e-15) continue;

        for (size_t j = 0; j < uncouple_g4.size(); j++)
        {
          const auto& g4_1 = g4Arr1[uncouple_g4[j].first]; 
          const auto& g4_2 = g4Arr2[uncouple_g4[j].second];


          int jj4_1 = g4_1.Lambdax2(), mm4_1 = g4_1.MLambdax2(),
              jj4_2 = g4_2.Lambdax2(), mm4_2 = g4_2.MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
            g4_1.n14(), g4_1.n24(), g4_1.n34(), g4_1.n44(), 
            g4_1.n13(), g4_1.n23(), g4_1.n33(), 
            g4_2.n14(), g4_2.n24(), g4_2.n34(), g4_2.n44(), 
            g4_2.n13(), g4_2.n23(), g4_2.n33(), 
            g4.n14(), g4.n24(), g4.n34(), g4.n44(), 
            g4.n13(), g4.n23(), g4.n33());
          // accU4 should have etamax columns

          if (allowU4 == false) continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
            g4_1.n13(), g4_1.n23(), g4_1.n33(), g4_1.n12(), g4_1.n22(), 
            g4_2.n13(), g4_2.n23(), g4_2.n33(), g4_2.n12(), g4_2.n22(), 
            g4.n13(), g4.n23(), g4.n33(), g4.n12(), g4.n22());

          if (allowU3_4 == false) continue;

          double cgc4 = cgcs.compute(jj4_1, jj4_2, jj4, mm4_1, mm4_2, mm4);

          Eigen::RowVectorXd cgcU4(etamax);
          cgcU4.setZero();

          for (size_t e = 0; e < etamax; e++)
            for (size_t r = 0; r < accU4.rows(); r++)
              cgcU4(e) += accU4(r, e) * accU3_4(r) * cgc4;

          if (cgcU4.norm() <= 1e-15) continue;

          // uncouple Tad
          U3xU4::Tad_gelfand Tad(n, sps3, sps4, Omega_ads, g3_1, g4_1);
          auto [coeffs_ads, adsArr_] = Tad.expand(dbs);

          // uncouple Tat
          U3xU4::Tat_gelfand Tat(n, sps3, sps3_conj, sps4, sps4_conj, Omega_ats,
                                 g3_2, g4_2);
          auto [coeffs_ats, atsArr_] = Tat.expand(dbs);

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
  std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
             std::vector<U3xU4::ads_phys>,    // ad's
             std::vector<U3xU4::ats_phys>>    // a's
  SingleShellTensor_phys::expand(sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    std::vector<Eigen::RowVectorXd> coeffs;
    std::vector<U3xU4::ads_phys> adsArr;
    std::vector<U3xU4::ats_phys> atsArr;

    auto n = shell();
    auto rhomax = rho0max();
    auto etamax = eta0max();

    // only ads
    if (ats == 0)
    {
      U3xU4::Tad_phys Tad(n, Omega_ads, LL, SS, TT, MMT, JJ, MMJ, labels);
      std::tie(coeffs, adsArr) = Tad.expand(dbs);
    }

    // only ats
    else if (ads == 0)
    {
      U3xU4::Tat_phys Tat(n, Omega_ats, LL, SS, TT, MMT, JJ, MMJ, labels);
      std::tie(coeffs, atsArr) = Tat.expand(dbs);
    }

    else
    {
      std::vector<u3::physicalvector> pv3s1, pv3s2;
      std::vector<u4::physicalvector> pv4s1, pv4s2;

      if (ads > 1)
      {
        pv3s1 = u3::physicalBasis(Omega_ads.back().u3());
        pv4s1 = u4::physicalBasis(Omega_ads.back().u4());
      }
      else
      {
        pv3s1 = u3::physicalBasis(u3::irrep(n, 0));
        pv4s1 = u4::physicalBasis(u4::irrep(1, 0, 0));
      }

      if (ats > 1)
      {
        pv3s2 = u3::physicalBasis(Omega_ats.back().u3());
        pv4s2 = u4::physicalBasis(Omega_ats.back().u4());
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
        auto kappa1 = qcx_utils::ito8(pv31.kappa), 
             LL1 = qcx_utils::ito8(2 * pv31.L);

        for (const auto& pv32 : pv3s2)
        {
          int lam2 = pv32.lam(), mu2 = pv32.mu();
          auto kappa2 = qcx_utils::ito8(pv32.kappa), 
               LL2 = qcx_utils::ito8(2 * pv32.L);

          if (std::abs(LL1 - LL2) > LL || (LL1 + LL2) < LL) continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(lam1, mu1, LL1 / 2, 
            lam2, mu2, LL2 / 2, lam(), mu(), LL / 2);
          if (allowU3 == false) continue;

          for (const auto& pv41 : pv4s1)
          {
            int P11 = pv41.P1(), P21 = pv41.P2(), P31 = pv41.P3(); 
            auto SS1 = qcx_utils::ito8(pv41.SS), 
                 TT1 = qcx_utils::ito8(pv41.TT);

            for (const auto& pv42 : pv4s2)
            {
              int P12 = pv42.P1(), P22 = pv42.P2(), P32 = pv42.P3(); 
              auto SS2 = qcx_utils::ito8(pv42.SS), 
                   TT2 = qcx_utils::ito8(pv42.TT);

              if (std::abs(SS1 - SS2) > SS || (SS1 + SS2) < SS ||
                  std::abs(TT1 - TT2) > TT || (TT1 + TT2) < TT)
                continue;

              auto [allowU4, cgcU4] = wigU4SU2SU2db.get(
                                  P11, P21, P31, SS1, TT1, 
                                  P12, P22, P32, SS2, TT2, 
                                  P1(), P2(), P3(), SS, TT);
              if (allowU4 == false) continue;

              for (int8_t JJ1 = LL1 + SS1; JJ1 >= std::abs(LL1 - SS1); JJ1 -= 2)
              {
                for (int8_t JJ2 = LL2 + SS2; JJ2 >= std::abs(LL2 - SS2); JJ2 -= 2)
                {
                  if (std::abs(JJ1 - JJ2) > JJ || (JJ1 + JJ2) < JJ) continue;

                  double U9j = racah9.compute(LL1, LL2, LL, 
                                              SS1, SS2, SS, 
                                              JJ1, JJ2, JJ);

                  for (int8_t MMJ1 = JJ1; MMJ1 >= -JJ1; MMJ1 -= 2)
                  {
                    int8_t MMJ2 = MMJ - MMJ1;
                    if (std::abs(MMJ2) > JJ2) continue;

                    double cgcJ = cgcs.compute(JJ1, JJ2, JJ, MMJ1, MMJ2, MMJ);

                    for (int8_t MMT1 = TT1; MMT1 >= -TT1; MMT1 -= 2)
                    {
                      int8_t MMT2 = MMT - MMT1;
                      if (std::abs(MMT2) > TT2) continue;

                      double cgcT = cgcs.compute(TT1, TT2, TT, MMT1, MMT2, MMT);

                      U3xU4::Tad_phys Tad(n, Omega_ads, LL1, SS1, TT1, MMT1, JJ1,
                                          MMJ1, labels);
                      auto [coeffs1, adArr] = Tad.expand(dbs);

                      U3xU4::Tat_phys Tat(n, Omega_ats, LL2, SS2, TT2, MMT2, JJ2,
                                          MMJ2, labels);
                      auto [coeffs2, atArr] = Tat.expand(dbs);

                      for (int i = 0; i < coeffs1.size(); i++)
                      {
                        for (int j = 0; j < coeffs2.size(); j++)
                        {
                          Eigen::RowVectorXd x(rhomax * etamax * kappamax *
                                               zetamax);
                          x.setZero();
                          for (int rho = 0; rho < rhomax; rho++)
                            for (int eta = 0; eta < etamax; eta++)
                              for (int k = 0; k < kappamax; k++)
                                for (int z = 0; z < zetamax; z++)
                                {
                                  int index =
                                      ((rho * etamax + eta) * kappamax + k) *
                                          zetamax +
                                      z;
                                  for (int k1 = 0; k1 < kappa1; k1++)
                                    for (int k2 = 0; k2 < kappa2; k2++)
                                      x(index) +=
                                          cgcU3((k1 * kappa2 + k2) * kappamax +
                                                    k,
                                                rho) *
                                          cgcU4(z, eta) * U9j * cgcJ * cgcT *
                                          coeffs1[i](k1) * coeffs2[j](k2);
                                }

                          if (x.norm() < 1e-12) continue;

                          coeffs.push_back(std::move(x));
                          adsArr.push_back(adArr[i]);
                          atsArr.push_back(atArr[j]);
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

    return std::make_tuple(coeffs, adsArr, atsArr);
  }

} // namespace U3xU4
