#include <cassert>
#include <stdexcept>

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

    int n13b = sps3_conj[c].n13(), n22b = sps3_conj[c].n22(),
        n11b = sps3_conj[c].n11(), n33p = sps4_conj[r].n33(),
        n22p = sps4_conj[r].n22(), n11p = sps4_conj[r].n11();

    u3::gelfand g3(n13b, 0, 0, n13b - n22b, 0, n13b - n11b);
    c = u3::findGelfand(g3, sps3);

    u4::gelfand g4(1, 0, 0, 0, 1 - n33p, 0, 0, 1 - n22p, 0, 1 - n11p);
    r = u4::findGelfand(g4, sps4);

    int phase = n22b + n11b + n33p + n22p + n11p;

    tensor = false;

    return (phase % 2 == 0) ? 1 : -1;
  }

  /////////////// PRODUCT OF CREATION OPERATORS/TENSORS /////////////////

  // expand coupled product of creation tensors (ad's)
  // in terms of creation operators (no phase needed)
  std::pair<std::vector<double>, std::vector<U3xU4::ads_gelfand>>
  Tad_gelfand::expand(u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
                      u4::WigU4U3_db& wigU4U3db) const
  {
    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps4: Gelfand array for [1,0,0,0]
    */

    std::vector<double> coeffs;
    std::vector<U3xU4::ads_gelfand> tensorsArr;

    std::size_t num = prod_characters.size() + 1;

    // one ad
    if (num == 1)
    {
      coeffs.push_back(1.0);
      auto c = u3::findGelfand(g3, sps3), r = u4::findGelfand(g4, sps4);
      U3xU4::ad_gelfand ad(c, r, sps3, sps4);
      U3xU4::ads_gelfand Tad(1, ad);
      tensorsArr.push_back(Tad);
    }

    // two ad's
    else if (num == 2)
    {
      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3, sps3, sps3);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4, sps4, sps4);

      int jj3 = g3.Lambdax2(), mm3 = g3.MLambdax2(), jj4 = g4.Lambdax2(),
          mm4 = g4.MLambdax2();

      for (std::size_t i = 0; i < uncouple_g3.size(); i++)
      {
        int g3_1 = uncouple_g3[i].first, g3_2 = uncouple_g3[i].second;
        int jj3_1 = sps3[g3_1].Lambdax2(), mm3_1 = sps3[g3_1].MLambdax2(),
            jj3_2 = sps3[g3_2].Lambdax2(), mm3_2 = sps3[g3_2].MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
            sps3[g3_1].n13(), sps3[g3_1].n23(), sps3[g3_1].n33(),
            sps3[g3_1].n12(), sps3[g3_1].n22(), sps3[g3_2].n13(),
            sps3[g3_2].n23(), sps3[g3_2].n33(), sps3[g3_2].n12(),
            sps3[g3_2].n22(), g3.n13(), g3.n23(), g3.n33(), g3.n12(),
            g3.n22()); // accU3 should have only one element

        if (allowU3 == false)
          continue;

        double cgc3 = cgcs.compute(jj3_1, jj3_2, jj3, mm3_1, mm3_2, mm3);

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15)
          continue;

        for (std::size_t j = 0; j < uncouple_g4.size(); j++)
        {
          int g4_1 = uncouple_g4[j].first, g4_2 = uncouple_g4[j].second;
          int jj4_1 = sps4[g4_1].Lambdax2(), mm4_1 = sps4[g4_1].MLambdax2(),
              jj4_2 = sps4[g4_2].Lambdax2(), mm4_2 = sps4[g4_2].MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
              sps4[g4_1].n14(), sps4[g4_1].n24(), sps4[g4_1].n34(),
              sps4[g4_1].n44(), sps4[g4_1].n13(), sps4[g4_1].n23(),
              sps4[g4_1].n33(), sps4[g4_2].n14(), sps4[g4_2].n24(),
              sps4[g4_2].n34(), sps4[g4_2].n44(), sps4[g4_2].n13(),
              sps4[g4_2].n23(), sps4[g4_2].n33(), g4.n14(), g4.n24(), g4.n34(),
              g4.n44(), g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false)
            continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
              sps4[g4_1].n13(), sps4[g4_1].n23(), sps4[g4_1].n33(),
              sps4[g4_1].n12(), sps4[g4_1].n22(), sps4[g4_2].n13(),
              sps4[g4_2].n23(), sps4[g4_2].n33(), sps4[g4_2].n12(),
              sps4[g4_2].n22(), g4.n13(), g4.n23(), g4.n33(), g4.n12(),
              g4.n22());

          if (allowU3_4 == false)
            continue;

          double cgc4 = cgcs.compute(jj4_1, jj4_2, jj4, mm4_1, mm4_2, mm4);

          double cgcU4 = 0.0;
          for (std::size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15)
            continue;

          U3xU4::ad_gelfand ad1(g3_1, g4_1, sps3, sps4),
                            ad2(g3_2, g4_2, sps3, sps4);
          U3xU4::ads_gelfand adad;
          adad.push_back(ad1);
          adad.push_back(ad2);

          coeffs.push_back(cgcU3 * cgcU4);
          tensorsArr.push_back(adad);
        }
      }
    }

    // three ad's
    else if (num == 3)
    {
      std::vector<u3::gelfand> g3Arr0 =
          u3::gelfandBasis(prod_characters[0].u3);
      std::vector<u4::gelfand> g4Arr0 =
          u4::gelfandBasis(prod_characters[0].u4);

      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3, g3Arr0, sps3);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4, g4Arr0, sps4);

      int jj3 = g3.Lambdax2(), mm3 = g3.MLambdax2(), jj4 = g4.Lambdax2(),
          mm4 = g4.MLambdax2();

      for (std::size_t i = 0; i < uncouple_g3.size(); i++)
      {
        int g3_0 = uncouple_g3[i].first, g3_3 = uncouple_g3[i].second;
        int jj3_0 = g3Arr0[g3_0].Lambdax2(), mm3_0 = g3Arr0[g3_0].MLambdax2(),
            jj3_3 = sps3[g3_3].Lambdax2(), mm3_3 = sps3[g3_3].MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
            g3Arr0[g3_0].n13(), g3Arr0[g3_0].n23(), g3Arr0[g3_0].n33(),
            g3Arr0[g3_0].n12(), g3Arr0[g3_0].n22(), sps3[g3_3].n13(),
            sps3[g3_3].n23(), sps3[g3_3].n33(), sps3[g3_3].n12(),
            sps3[g3_3].n22(), g3.n13(), g3.n23(), g3.n33(), g3.n12(),
            g3.n22()); // accU3 should have only one element

        if (allowU3 == false)
          continue;

        double cgc3 = cgcs.compute(jj3_0, jj3_3, jj3, mm3_0, mm3_3, mm3);

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15)
          continue;

        for (std::size_t j = 0; j < uncouple_g4.size(); j++)
        {
          int g4_0 = uncouple_g4[j].first, g4_3 = uncouple_g4[j].second;
          int jj4_0 = g4Arr0[g4_0].Lambdax2(), mm4_0 = g4Arr0[g4_0].MLambdax2(),
              jj4_3 = sps4[g4_3].Lambdax2(), mm4_3 = sps4[g4_3].MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
              g4Arr0[g4_0].n14(), g4Arr0[g4_0].n24(), g4Arr0[g4_0].n34(),
              g4Arr0[g4_0].n44(), g4Arr0[g4_0].n13(), g4Arr0[g4_0].n23(),
              g4Arr0[g4_0].n33(), sps4[g4_3].n14(), sps4[g4_3].n24(),
              sps4[g4_3].n34(), sps4[g4_3].n44(), sps4[g4_3].n13(),
              sps4[g4_3].n23(), sps4[g4_3].n33(), g4.n14(), g4.n24(), g4.n34(),
              g4.n44(), g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false)
            continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
              g4Arr0[g4_0].n13(), g4Arr0[g4_0].n23(), g4Arr0[g4_0].n33(),
              g4Arr0[g4_0].n12(), g4Arr0[g4_0].n22(), sps4[g4_3].n13(),
              sps4[g4_3].n23(), sps4[g4_3].n33(), sps4[g4_3].n12(),
              sps4[g4_3].n22(), g4.n13(), g4.n23(), g4.n33(), g4.n12(),
              g4.n22());

          if (allowU3_4 == false)
            continue;

          double cgc4 = cgcs.compute(jj4_0, jj4_3, jj4, mm4_0, mm4_3, mm4);

          double cgcU4 = 0.0;
          for (std::size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15)
            continue;

          U3xU4::ad_gelfand ad3(g3_3, g4_3, sps3, sps4);

          std::vector<U3xU4::irrep> prod_0;
          prod_0.push_back(prod_characters[0]);

          U3xU4::Tad_gelfand Tad(n, sps3, sps4, prod_0, g3Arr0[g3_0],
                                 g4Arr0[g4_0]);

          auto [coeffs_0, tensors_0] = Tad.expand(cgcs, wigU3U2db, wigU4U3db);

          for (std::size_t k = 0; k < coeffs_0.size(); k++)
          {
            coeffs.push_back(coeffs_0[k] * cgcU3 * cgcU4);
            U3xU4::ads_gelfand tensorsArr_temp = tensors_0[k];
            tensorsArr_temp.push_back(ad3);
            tensorsArr.push_back(tensorsArr_temp);
          }
        }
      }
    }

    return std::make_pair(coeffs, tensorsArr);
  }

  // Expand in terms of uncoupled products of ad's
  std::pair<std::vector<Eigen::RowVectorXd>, // each element has size == kappa
              std::vector<U3xU4::ads_phys>>
  Tad_phys::expand(u34utils::SO3_CGC& cgcs, 
                   u34utils::SO3_Racah9& racah9,
                   u3::WigU3SO3_db& wigU3SO3db, 
                   u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
  {
    std::vector<Eigen::RowVectorXd> coeffsArr;
    std::vector<U3xU4::ads_phys> tensorsArr;

    std::size_t num = prod_characters.size() + 1;

    // one ad
    if (num == 1)
    {
      Eigen::RowVectorXd c(1);
      c(0) = 1.0;
      coeffsArr.push_back(c);
      U3xU4::ad_phys ad(n, LL, JJ, MMJ, MMT);
      U3xU4::ads_phys Tad(1, ad);
      tensorsArr.push_back(Tad);
    }

    // two ad's
    else if (num == 2)
    {
      std::vector<u3::physicalvector> pvs = u3::physicalBasis(u3::irrep(n,0));
      int lam = prod_characters.back().u3.lam(),
          mu = prod_characters.back().u3.mu(),
          P1 = prod_characters.back().u4.P1(),
          P2 = prod_characters.back().u4.P2(),
          P3 = prod_characters.back().u4.P3();

      for (int pva = 0; pva < pvs.size(); pva++)
      {
        int lla = 2 * pvs[pva].L;

        for (int pvb = 0; pvb < pvs.size(); pvb++)
        {
          int llb = 2 * pvs[pvb].L;

          if (std::abs(lla - llb) > LL || (lla + llb) < LL)
            continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(n, 0, lla/2,
                                                 n, 0, llb/2,
                                                 lam, mu, LL/2);

          auto [allowU4, cgcU4] = wigU4SU2SU2db.get(1, 0, 0, 1, 1,
                                                    1, 0, 0, 1, 1,
                                                    P1, P2, P3, SS, TT);

          for (int jja = lla + 1; jja >= std::abs(lla - 1); jja-=2)
          {
            for (int jjb = llb + 1; jjb >=  std::abs(llb - 1); jjb-=2)
            {
              if (std::abs(jja - jjb) > JJ || (jja + jjb) < JJ)
                continue;

              double U9j = racah9.compute(lla, llb, LL, 1, 1, SS, jja, jjb, JJ);

              for (int mmja = jja; mmja >= -jja; mmja-=2)
              {
                int mmjb = MMJ - mmja;
                for (int mmta = 1; mmta >= -1; mmta-=2)
                {
                  int mmtb = MMT - mmta;

                  double cgcJ = cgcs.compute(jja, jjb, JJ, mmja, mmjb, MMJ),
                         cgcT = cgcs.compute(1, 1, TT, mmta, mmtb, MMT),
                         factor = cgcU4(0,0) * U9j * cgcJ * cgcT;

                  Eigen::RowVectorXd coeffs(kappamax);
                  coeffs.setZero();
                  for (int k = 0; k < kappamax; k++)
                    coeffs(k) += cgcU3(k,0) * factor;

                  if (coeffs.norm() < 1e-12)
                    continue;

                  coeffsArr.push_back(coeffs);

                  U3xU4::ad_phys ad1(n, lla, jja, mmja, mmta),
                                 ad2(n, llb, jjb, mmjb, mmtb);
                  U3xU4::ads_phys Tad = {ad1, ad2};
                  tensorsArr.push_back(Tad);
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
        pv3s = u3::physicalBasis(u3::irrep(n,0)),
        pv3s0 = u3::physicalBasis(prod_characters[0].u3);
      std::vector<u4::physicalvector>
        pv4s0 = u4::physicalBasis(prod_characters[0].u4);

      int lam = prod_characters.back().u3.lam(),
          mu = prod_characters.back().u3.mu(),
          P1 = prod_characters.back().u4.P1(),
          P2 = prod_characters.back().u4.P2(),
          P3 = prod_characters.back().u4.P3(),
          lam0 = prod_characters[0].u3.lam(),
          mu0 = prod_characters[0].u3.mu(),
          P10 = prod_characters[0].u4.P1(),
          P20 = prod_characters[0].u4.P2(),
          P30 = prod_characters[0].u4.P3();

      for (int pvc = 0; pvc < pv3s.size(); pvc++)
      {
        int llc = 2 * pv3s[pvc].L; 

        for (int pv30 = 0; pv30 < pv3s0.size(); pv30++)
        {
          int LL0 = 2 * pv3s0[pv30].L;

          if (std::abs(LL0 - llc) > LL || (LL0 + llc) < LL)
            continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(lam0, mu0, LL0/2,
                                                 n, 0, llc/2,
                                                 lam, mu, LL/2);

          for (int pv40 = 0; pv40 < pv4s0.size(); pv40++)
          {
            int SS0 = pv4s0[pv40].SS, TT0 = pv4s0[pv40].TT;

            if (std::abs(SS0 - 1) > SS || (SS0 + 1) < SS ||
                std::abs(TT0 - 1) > TT || (TT0 + 1) < TT)
              continue;

            auto [allowU4, cgcU4] = wigU4SU2SU2db.get(P10, P20, P30, SS0, TT0,
                                                      1, 0, 0, 1, 1,
                                                      P1, P2, P3, SS, TT);
            for (int jjc = llc + 1; jjc >= std::abs(llc - 1); jjc-=2)
            {
              for (int JJ0 = LL0 + SS0; JJ0 >= std::abs(LL0 - SS0); JJ0-=2)
              {
                if (std::abs(JJ0 - jjc) > JJ || (JJ0 + jjc) < JJ)
                  continue;  

                double U9j = racah9.compute(LL0, llc, LL, SS0, 1, SS, JJ0, jjc, JJ);

                for (int mmjc = jjc; mmjc >= -jjc; mmjc-=2)
                {
                  int MMJ0 = MMJ - mmjc;
                  for (int mmtc = 1; mmtc >= -1; mmtc-=2)
                  {
                    int MMT0 = MMT - mmtc;

                    double cgcJ = cgcs.compute(JJ0, jjc, JJ, MMJ0, mmjc, MMJ),
                           cgcT = cgcs.compute(TT0, 1, TT, MMT0, mmtc, MMT),
                           factor = cgcU4(0,0) * U9j * cgcJ * cgcT;

                    std::vector<U3xU4::irrep> prod_ads0(1, prod_characters[0]);
                    U3xU4::Tad_phys Tad0(n,prod_ads0,LL0,SS0,TT0,MMT0,JJ0,MMJ0);

                    auto [coeffsArr0, adArr0] = Tad0.expand(
                      cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

                    for (int i = 0; i < coeffsArr0.size(); i++)
                    {
                      Eigen::RowVectorXd coeffs(kappamax);
                      coeffs.setZero();
                      for (int k = 0; k < kappamax; k++)
                        for (int k0 = 0; k0 < Tad0.kappamax; k0++)
                          coeffs(k) += cgcU3(k0 * kappamax + k,0) * factor * coeffsArr0[i](k0);

                      if (coeffs.norm() < 1e-12)
                        continue;

                      coeffsArr.push_back(coeffs);

                      U3xU4::ads_phys adad = adArr0[i];
                      U3xU4::ad_phys ad3(n, llc, jjc, mmjc, mmtc);
                      adad.push_back(ad3);
                      tensorsArr.push_back(adad);
                    }
                  }
                }
              }
            }

          }
        }
      }
    } 

    return std::make_pair(coeffsArr, tensorsArr);
  }  

  /////////////// PRODUCT OF ANNIHILATION OPERATORS/TENSORS ///////////////

  // expand coupled product of annihilation tensors (at's)
  // in terms of annihilation operators (a's) via a phase
  std::pair<std::vector<double>, std::vector<U3xU4::ats_gelfand>>
  Tat_gelfand::expand(u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
                      u4::WigU4U3_db& wigU4U3db) const
  {
    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps3_conj: Gelfand array for [n,n,0]

    sps4: Gelfand array for [1,0,0,0]
    sps4_conj: Gelfand array for [1,1,1,0]
    */

    std::vector<double> coeffs;
    std::vector<std::vector<U3xU4::at_gelfand>> tensorsArr;

    std::size_t num = prod_characters.size() + 1;

    // one at
    if (num == 1)
    {
      auto c = u3::findGelfand(g3, sps3_conj),
           r = u4::findGelfand(g4, sps4_conj);
      U3xU4::at_gelfand at(c, r, sps3, sps3_conj, sps4, sps4_conj);
      auto phase = at.turnToAnnihilation();

      assert(!at.isTensor());

      U3xU4::ats_gelfand Tat(1, at);
      coeffs.push_back(phase);
      tensorsArr.push_back(Tat);
    }

    // two at's
    else if (num == 2)
    {
      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3, sps3_conj, sps3_conj);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4, sps4_conj, sps4_conj);

      int jj3 = g3.Lambdax2(), mm3 = g3.MLambdax2(), jj4 = g4.Lambdax2(),
          mm4 = g4.MLambdax2();

      for (std::size_t i = 0; i < uncouple_g3.size(); i++)
      {
        int g3_1 = uncouple_g3[i].first, g3_2 = uncouple_g3[i].second;
        int jj3_1 = sps3_conj[g3_1].Lambdax2(),
            mm3_1 = sps3_conj[g3_1].MLambdax2(),
            jj3_2 = sps3_conj[g3_2].Lambdax2(),
            mm3_2 = sps3_conj[g3_2].MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
            sps3_conj[g3_1].n13(), sps3_conj[g3_1].n23(), sps3_conj[g3_1].n33(),
            sps3_conj[g3_1].n12(), sps3_conj[g3_1].n22(), sps3_conj[g3_2].n13(),
            sps3_conj[g3_2].n23(), sps3_conj[g3_2].n33(), sps3_conj[g3_2].n12(),
            sps3_conj[g3_2].n22(), g3.n13(), g3.n23(), g3.n33(), g3.n12(),
            g3.n22()); // accU3 should have only one element

        if (allowU3 == false)
          continue;

        double cgc3 = cgcs.compute(jj3_1, jj3_2, jj3, mm3_1, mm3_2, mm3);

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15)
          continue;

        for (std::size_t j = 0; j < uncouple_g4.size(); j++)
        {
          int g4_1 = uncouple_g4[j].first, g4_2 = uncouple_g4[j].second;
          int jj4_1 = sps4_conj[g4_1].Lambdax2(),
              mm4_1 = sps4_conj[g4_1].MLambdax2(),
              jj4_2 = sps4_conj[g4_2].Lambdax2(),
              mm4_2 = sps4_conj[g4_2].MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
              sps4_conj[g4_1].n14(), sps4_conj[g4_1].n24(),
              sps4_conj[g4_1].n34(), sps4_conj[g4_1].n44(),
              sps4_conj[g4_1].n13(), sps4_conj[g4_1].n23(),
              sps4_conj[g4_1].n33(), sps4_conj[g4_2].n14(),
              sps4_conj[g4_2].n24(), sps4_conj[g4_2].n34(),
              sps4_conj[g4_2].n44(), sps4_conj[g4_2].n13(),
              sps4_conj[g4_2].n23(), sps4_conj[g4_2].n33(), g4.n14(), g4.n24(),
              g4.n34(), g4.n44(), g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false)
            continue;

          auto [allowU3_4, accU3_4] =
              wigU3U2db.get(sps4_conj[g4_1].n13(), sps4_conj[g4_1].n23(),
                            sps4_conj[g4_1].n33(), sps4_conj[g4_1].n12(),
                            sps4_conj[g4_1].n22(), sps4_conj[g4_2].n13(),
                            sps4_conj[g4_2].n23(), sps4_conj[g4_2].n33(),
                            sps4_conj[g4_2].n12(), sps4_conj[g4_2].n22(),
                            g4.n13(), g4.n23(), g4.n33(), g4.n12(), g4.n22());

          if (allowU3_4 == false)
            continue;

          double cgc4 = cgcs.compute(jj4_1, jj4_2, jj4, mm4_1, mm4_2, mm4);

          double cgcU4 = 0.0;
          for (std::size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15)
            continue;

          U3xU4::at_gelfand at1(g3_1, g4_1, sps3, sps3_conj, sps4, sps4_conj),
              at2(g3_2, g4_2, sps3, sps3_conj, sps4, sps4_conj);
          auto phase1 = at1.turnToAnnihilation(),
               phase2 = at2.turnToAnnihilation();

          assert(!at1.isTensor());
          assert(!at2.isTensor());

          U3xU4::ats_gelfand atat;
          atat.push_back(at1);
          atat.push_back(at2);

          coeffs.push_back(phase1 * phase2 * cgcU3 * cgcU4);
          tensorsArr.push_back(atat);
        }
      }
    }

    // three at's
    else if (num == 3)
    {
      std::vector<u3::gelfand> g3Arr0 =
          u3::gelfandBasis(prod_characters[0].u3);
      std::vector<u4::gelfand> g4Arr0 =
          u4::gelfandBasis(prod_characters[0].u4);

      std::vector<std::pair<int, int>> uncouple_g3 =
          u3::expand_to_uncoupled(g3, g3Arr0, sps3_conj);

      std::vector<std::pair<int, int>> uncouple_g4 =
          u4::expand_to_uncoupled(g4, g4Arr0, sps4_conj);

      int jj3 = g3.Lambdax2(), mm3 = g3.MLambdax2(), jj4 = g4.Lambdax2(),
          mm4 = g4.MLambdax2();

      for (std::size_t i = 0; i < uncouple_g3.size(); i++)
      {
        int g3_0 = uncouple_g3[i].first, g3_3 = uncouple_g3[i].second;
        int jj3_0 = g3Arr0[g3_0].Lambdax2(), mm3_0 = g3Arr0[g3_0].MLambdax2(),
            jj3_3 = sps3_conj[g3_3].Lambdax2(),
            mm3_3 = sps3_conj[g3_3].MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
            g3Arr0[g3_0].n13(), g3Arr0[g3_0].n23(), g3Arr0[g3_0].n33(),
            g3Arr0[g3_0].n12(), g3Arr0[g3_0].n22(), sps3_conj[g3_3].n13(),
            sps3_conj[g3_3].n23(), sps3_conj[g3_3].n33(), sps3_conj[g3_3].n12(),
            sps3_conj[g3_3].n22(), g3.n13(), g3.n23(), g3.n33(), g3.n12(),
            g3.n22()); // accU3 should have only one element

        if (allowU3 == false)
          continue;

        double cgc3 = cgcs.compute(jj3_0, jj3_3, jj3, mm3_0, mm3_3, mm3);

        double cgcU3 = accU3(0) * cgc3;

        if (std::fabs(cgcU3) <= 1e-15)
          continue;

        for (std::size_t j = 0; j < uncouple_g4.size(); j++)
        {
          int g4_0 = uncouple_g4[j].first, g4_3 = uncouple_g4[j].second;
          int jj4_0 = g4Arr0[g4_0].Lambdax2(), mm4_0 = g4Arr0[g4_0].MLambdax2(),
              jj4_3 = sps4_conj[g4_3].Lambdax2(),
              mm4_3 = sps4_conj[g4_3].MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
              g4Arr0[g4_0].n14(), g4Arr0[g4_0].n24(), g4Arr0[g4_0].n34(),
              g4Arr0[g4_0].n44(), g4Arr0[g4_0].n13(), g4Arr0[g4_0].n23(),
              g4Arr0[g4_0].n33(), sps4_conj[g4_3].n14(), sps4_conj[g4_3].n24(),
              sps4_conj[g4_3].n34(), sps4_conj[g4_3].n44(),
              sps4_conj[g4_3].n13(), sps4_conj[g4_3].n23(),
              sps4_conj[g4_3].n33(), g4.n14(), g4.n24(), g4.n34(), g4.n44(),
              g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false)
            continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
              g4Arr0[g4_0].n13(), g4Arr0[g4_0].n23(), g4Arr0[g4_0].n33(),
              g4Arr0[g4_0].n12(), g4Arr0[g4_0].n22(), sps4_conj[g4_3].n13(),
              sps4_conj[g4_3].n23(), sps4_conj[g4_3].n33(),
              sps4_conj[g4_3].n12(), sps4_conj[g4_3].n22(), g4.n13(), g4.n23(),
              g4.n33(), g4.n12(), g4.n22());

          if (allowU3_4 == false)
            continue;

          double cgc4 = cgcs.compute(jj4_0, jj4_3, jj4, mm4_0, mm4_3, mm4);

          double cgcU4 = 0.0;
          for (std::size_t r = 0; r < accU4.rows(); r++)
            cgcU4 += accU4(r, 0) * accU3_4(r) * cgc4;

          if (std::fabs(cgcU4) <= 1e-15)
            continue;

          U3xU4::at_gelfand at3(g3_3, g4_3, sps3, sps3_conj, sps4, sps4_conj);
          auto phase3 = at3.turnToAnnihilation();

          assert(!at3.isTensor());

          std::vector<U3xU4::irrep> prod_0;
          prod_0.push_back(prod_characters[0]);

          U3xU4::Tat_gelfand Tat(n, sps3, sps3_conj, sps4, sps4_conj, prod_0,
                                 g3Arr0[g3_0], g4Arr0[g4_0]);

          auto [coeffs_0, tensors_0] = Tat.expand(cgcs, wigU3U2db, wigU4U3db);

          for (std::size_t k = 0; k < coeffs_0.size(); k++)
          {
            coeffs.push_back(phase3 * coeffs_0[k] * cgcU3 * cgcU4);
            U3xU4::ats_gelfand tensorsArr_temp = tensors_0[k];
            tensorsArr_temp.push_back(at3);
            tensorsArr.push_back(tensorsArr_temp);
          }
        }
      }
    }

    return std::make_pair(coeffs, tensorsArr);
  }

  // expand in terms of uncoupled products of at's
  std::pair<std::vector<Eigen::RowVectorXd>, // each element has size == kappa
              std::vector<U3xU4::ats_phys>>
  Tat_phys::expand(u34utils::SO3_CGC& cgcs, 
                   u34utils::SO3_Racah9& racah9,
                   u3::WigU3SO3_db& wigU3SO3db, 
                   u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
  {
    std::vector<Eigen::RowVectorXd> coeffsArr;
    std::vector<U3xU4::ats_phys> tensorsArr;

    std::size_t num = prod_characters.size() + 1;

    // one at
    if (num == 1)
    {
      Eigen::RowVectorXd c(1);
      U3xU4::at_phys at(n, LL, JJ, MMJ, MMT);
      c(0) = at.turnToAnnihilation();
      coeffsArr.push_back(c);
      U3xU4::ats_phys Tat(1, at);
      tensorsArr.push_back(Tat);
    }

    // two at's
    else if (num == 2)
    {
      std::vector<u3::physicalvector> pvs = u3::physicalBasis(u3::irrep(0,n));
      int lam = prod_characters.back().u3.lam(),
          mu = prod_characters.back().u3.mu(),
          P1 = prod_characters.back().u4.P1(),
          P2 = prod_characters.back().u4.P2(),
          P3 = prod_characters.back().u4.P3();

      for (int pva = 0; pva < pvs.size(); pva++)
      {
        int lla = 2 * pvs[pva].L;

        for (int pvb = 0; pvb < pvs.size(); pvb++)
        {
          int llb = 2 * pvs[pvb].L;

          if (std::abs(lla - llb) > LL || (lla + llb) < LL)
            continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(0, n, lla/2,
                                                 0, n, llb/2,
                                                 lam, mu, LL/2);

          auto [allowU4, cgcU4] = wigU4SU2SU2db.get(0, 0, 1, 1, 1,
                                                    0, 0, 1, 1, 1,
                                                    P1, P2, P3, SS, TT);

          for (int jja = lla + 1; jja >= std::fabs(lla - 1); jja-=2)
          {
            for (int jjb = llb + 1; jjb >=  std::fabs(llb - 1); jjb-=2)
            {
              if (std::abs(jja - jjb) > JJ || (jja + jjb) < JJ)
                continue;

              double U9j = racah9.compute(lla, llb, LL, 1, 1, SS, jja, jjb, JJ);

              for (int mmja = jja; mmja >= -jja; mmja-=2)
              {
                int mmjb = MMJ - mmja;
                for (int mmta = 1; mmta >= -1; mmta-=2)
                {
                  int mmtb = MMT - mmta;

                  double cgcJ = cgcs.compute(jja, jjb, JJ, mmja, mmjb, MMJ),
                         cgcT = cgcs.compute(1, 1, TT, mmta, mmtb, MMT);

                  U3xU4::at_phys at1(n, lla, jja, mmja, mmta),
                                 at2(n, llb, jjb, mmjb, mmtb);
                  double phase1 = at1.turnToAnnihilation(),
                         phase2 = at2.turnToAnnihilation(),
                         factor = cgcU4(0,0) * U9j * cgcJ * cgcT * phase1 * phase2;

                  Eigen::RowVectorXd coeffs(kappamax);
                  coeffs.setZero();
                  for (int k = 0; k < kappamax; k++)
                    coeffs(k) += cgcU3(k,0) * factor;

                  if (coeffs.norm() < 1e-12)
                    continue;

                  coeffsArr.push_back(coeffs);

                  assert(!at1.isTensor());
                  assert(!at2.isTensor());

                  U3xU4::ats_phys Tat = {at1, at2};
                  tensorsArr.push_back(Tat);
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
        pv3s = u3::physicalBasis(u3::irrep(0,n)),
        pv3s0 = u3::physicalBasis(prod_characters[0].u3);
      std::vector<u4::physicalvector>
        pv4s0 = u4::physicalBasis(prod_characters[0].u4);

      int lam = prod_characters.back().u3.lam(),
          mu = prod_characters.back().u3.mu(),
          P1 = prod_characters.back().u4.P1(),
          P2 = prod_characters.back().u4.P2(),
          P3 = prod_characters.back().u4.P3(),
          lam0 = prod_characters[0].u3.lam(),
          mu0 = prod_characters[0].u3.mu(),
          P10 = prod_characters[0].u4.P1(),
          P20 = prod_characters[0].u4.P2(),
          P30 = prod_characters[0].u4.P3();

      for (int pvc = 0; pvc < pv3s.size(); pvc++)
      {
        int llc = 2 * pv3s[pvc].L; 

        for (int pv30 = 0; pv30 < pv3s0.size(); pv30++)
        {
          int LL0 = 2 * pv3s0[pv30].L;

          if (std::abs(LL0 - llc) > LL || (LL0 + llc) < LL)
            continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(lam0, mu0, LL0/2,
                                                 0, n, llc/2,
                                                 lam, mu, LL/2);

          for (int pv40 = 0; pv40 < pv4s0.size(); pv40++)
          {
            int SS0 = pv4s0[pv40].SS, TT0 = pv4s0[pv40].TT;

            if (std::abs(SS0 - 1) > SS || (SS0 + 1) < SS ||
                std::abs(TT0 - 1) > TT || (TT0 + 1) < TT)
              continue;

            auto [allowU4, cgcU4] = wigU4SU2SU2db.get(P10, P20, P30, SS0, TT0,
                                                      0, 0, 1, 1, 1,
                                                      P1, P2, P3, SS, TT);
            for (int jjc = llc + 1; jjc >= std::abs(llc - 1); jjc-=2)
            {
              for (int JJ0 = LL0 + SS0; JJ0 >= std::abs(LL0 - SS0); JJ0-=2)
              {
                if (std::abs(JJ0 - jjc) > JJ || (JJ0 + jjc) < JJ)
                  continue;  

                double U9j = racah9.compute(LL0, llc, LL, SS0, 1, SS, JJ0, jjc, JJ);

                for (int mmjc = jjc; mmjc >= -jjc; mmjc-=2)
                {
                  int MMJ0 = MMJ - mmjc;
                  for (int mmtc = 1; mmtc >= -1; mmtc-=2)
                  {
                    int MMT0 = MMT - mmtc;

                    double cgcJ = cgcs.compute(JJ0, jjc, JJ, MMJ0, mmjc, MMJ),
                           cgcT = cgcs.compute(TT0, 1, TT, MMT0, mmtc, MMT),
                           factor = cgcU4(0,0) * U9j * cgcJ * cgcT;

                    std::vector<U3xU4::irrep> prod_ats0(1, prod_characters[0]);
                    U3xU4::Tat_phys Tat0(n,prod_ats0,LL0,SS0,TT0,MMT0,JJ0,MMJ0);

                    auto [coeffsArr0, aArr0] = Tat0.expand(
                      cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

                    for (int i = 0; i < coeffsArr0.size(); i++)
                    {
                      U3xU4::at_phys at3(n, llc, jjc, mmjc, mmtc);
                      double phase3 = at3.turnToAnnihilation();

                      Eigen::RowVectorXd coeffs(kappamax);
                      coeffs.setZero();
                      for (int k = 0; k < kappamax; k++)
                        for (int k0 = 0; k0 < Tat0.kappamax; k0++)
                          coeffs(k) += cgcU3(k0 * kappamax + k,0) * factor 
                                    * coeffsArr0[i](k0) * phase3;

                      if (coeffs.norm() < 1e-12)
                        continue;

                      coeffsArr.push_back(coeffs);

                      U3xU4::ats_phys atat = aArr0[i];
                      atat.push_back(at3);
                      tensorsArr.push_back(atat);
                    }
                  }
                }
              }
            }

          }
        }
      }
    } 

    return std::make_pair(coeffsArr, tensorsArr);
  }

  ////// PRODUCT OF CREATION AND ANNIHILATION OPERATORS/TENSORS ///////

  // expand coupled product of creation and annihilation tensors
  // in terms of creation and annihilation operators (ad's and a's)
  std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
             std::vector<U3xU4::ads_gelfand>, // ad's
             std::vector<U3xU4::ats_gelfand>> // a's
  Tadat_gelfand::expand(u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
                        u4::WigU4U3_db& wigU4U3db) const
  {
    std::vector<Eigen::RowVectorXd> coeffs;
    std::vector<U3xU4::ads_gelfand> adsArr;
    std::vector<U3xU4::ats_gelfand> atsArr;

    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps3_conj: Gelfand array for [n,n,0]

    sps4: Gelfand array for [1,0,0,0]
    sps4_conj: Gelfand array for [1,1,1,0]
    */

    // only ads
    if (ats == 0)
    {
      U3xU4::Tad_gelfand Tad(n, sps3, sps4, prod_ads, g3, g4);

      auto [coeffs_, adsArr_] = Tad.expand(cgcs, wigU3U2db, wigU4U3db);

      for (std::size_t i = 0; i < coeffs_.size(); i++)
      {
        Eigen::RowVectorXd x(1);
        x(0) = coeffs_[i];
        coeffs.push_back(x);
        adsArr.push_back(adsArr_[i]);
      }
    }

    // only ats
    else if (ads == 0)
    {
      U3xU4::Tat_gelfand Tat(n, sps3, sps3_conj, sps4, sps4_conj, prod_ats, g3,
                             g4);

      auto [coeffs_, atsArr_] = Tat.expand(cgcs, wigU3U2db, wigU4U3db);

      for (std::size_t i = 0; i < coeffs_.size(); i++)
      {
        Eigen::RowVectorXd x(1);
        x(0) = coeffs_[i];
        coeffs.push_back(x);
        atsArr.push_back(atsArr_[i]);
      }
    }

    else
    {
      std::vector<u3::gelfand> g3Arr1, g3Arr2;
      std::vector<u4::gelfand> g4Arr1, g4Arr2;

      if (ads > 1)
      {
        g3Arr1 = u3::gelfandBasis(prod_ads.back().u3);
        g4Arr1 = u4::gelfandBasis(prod_ads.back().u4);
      }
      else
      {
        g3Arr1 = sps3;
        g4Arr1 = sps4;
      }

      if (ats > 1)
      {
        g3Arr2 = u3::gelfandBasis(prod_ats.back().u3);
        g4Arr2 = u4::gelfandBasis(prod_ats.back().u4);
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

      for (std::size_t i = 0; i < uncouple_g3.size(); i++)
      {
        int g3_1 = uncouple_g3[i].first, g3_2 = uncouple_g3[i].second;
        int jj3_1 = g3Arr1[g3_1].Lambdax2(), mm3_1 = g3Arr1[g3_1].MLambdax2(),
            jj3_2 = g3Arr2[g3_2].Lambdax2(), mm3_2 = g3Arr2[g3_2].MLambdax2();

        auto [allowU3, accU3] = wigU3U2db.get(
            g3Arr1[g3_1].n13(), g3Arr1[g3_1].n23(), g3Arr1[g3_1].n33(),
            g3Arr1[g3_1].n12(), g3Arr1[g3_1].n22(), g3Arr2[g3_2].n13(),
            g3Arr2[g3_2].n23(), g3Arr2[g3_2].n33(), g3Arr2[g3_2].n12(),
            g3Arr2[g3_2].n22(), g3.n13(), g3.n23(), g3.n33(), g3.n12(),
            g3.n22()); // accU3 should have only rhomax elements

        if (allowU3 == false)
          continue;

        double cgc3 = cgcs.compute(jj3_1, jj3_2, jj3, mm3_1, mm3_2, mm3);

        auto cgcU3 = accU3 * cgc3;

        if (cgcU3.norm() <= 1e-15)
          continue;

        for (std::size_t j = 0; j < uncouple_g4.size(); j++)
        {
          int g4_1 = uncouple_g4[j].first, g4_2 = uncouple_g4[j].second;
          int jj4_1 = g4Arr1[g4_1].Lambdax2(), mm4_1 = g4Arr1[g4_1].MLambdax2(),
              jj4_2 = g4Arr2[g4_2].Lambdax2(), mm4_2 = g4Arr2[g4_2].MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
              g4Arr1[g4_1].n14(), g4Arr1[g4_1].n24(), g4Arr1[g4_1].n34(),
              g4Arr1[g4_1].n44(), g4Arr1[g4_1].n13(), g4Arr1[g4_1].n23(),
              g4Arr1[g4_1].n33(), g4Arr2[g4_2].n14(), g4Arr2[g4_2].n24(),
              g4Arr2[g4_2].n34(), g4Arr2[g4_2].n44(), g4Arr2[g4_2].n13(),
              g4Arr2[g4_2].n23(), g4Arr2[g4_2].n33(), g4.n14(), g4.n24(),
              g4.n34(), g4.n44(), g4.n13(), g4.n23(), g4.n33());
          // accU4 should have etamax columns

          if (allowU4 == false)
            continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
              g4Arr1[g4_1].n13(), g4Arr1[g4_1].n23(), g4Arr1[g4_1].n33(),
              g4Arr1[g4_1].n12(), g4Arr1[g4_1].n22(), g4Arr2[g4_2].n13(),
              g4Arr2[g4_2].n23(), g4Arr2[g4_2].n33(), g4Arr2[g4_2].n12(),
              g4Arr2[g4_2].n22(), g4.n13(), g4.n23(), g4.n33(), g4.n12(),
              g4.n22());

          if (allowU3_4 == false)
            continue;

          double cgc4 = cgcs.compute(jj4_1, jj4_2, jj4, mm4_1, mm4_2, mm4);

          Eigen::RowVectorXd cgcU4(etamax);
          cgcU4.setZero();

          for (std::size_t e = 0; e < etamax; e++)
            for (std::size_t r = 0; r < accU4.rows(); r++)
              cgcU4(e) += accU4(r, e) * accU3_4(r) * cgc4;

          if (cgcU4.norm() <= 1e-15)
            continue;

          // uncouple Tad
          U3xU4::Tad_gelfand Tad(n, sps3, sps4, prod_ads, g3Arr1[g3_1],
                                 g4Arr1[g4_1]);
          auto [coeffs_ads, adsArr_] = Tad.expand(cgcs, wigU3U2db, wigU4U3db);

          // uncouple Tat
          U3xU4::Tat_gelfand Tat(n, sps3, sps3_conj, sps4, sps4_conj, prod_ats,
                                 g3Arr2[g3_2], g4Arr2[g4_2]);
          auto [coeffs_ats, atsArr_] = Tat.expand(cgcs, wigU3U2db, wigU4U3db);

          for (std::size_t k = 0; k < coeffs_ads.size(); k++)
          {
            for (std::size_t l = 0; l < coeffs_ats.size(); l++)
            {
              Eigen::RowVectorXd coeffs_(rhomax * etamax);
              coeffs_.setZero();

              for (std::size_t r = 0; r < rhomax; r++)
                for (std::size_t e = 0; e < etamax; e++)
                {
                  coeffs_(r * etamax + e) +=
                      cgcU3(r) * cgcU4(e) * coeffs_ads[k] * coeffs_ats[l];
                }

              coeffs.push_back(coeffs_);
              adsArr.push_back(adsArr_[k]);
              atsArr.push_back(atsArr_[l]);
            }
          }
        }
      }
    }

    return std::make_tuple(coeffs, adsArr, atsArr);
  }

  // expand coupled product of creation and annihilation tensors
  // in terms of creation and annihilation operators (ad's and a's)
  std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
               std::vector<U3xU4::ads_phys>, // ad's
               std::vector<U3xU4::ats_phys>> // a's
  Tadat_phys::expand(u34utils::SO3_CGC& cgcs, u34utils::SO3_Racah9& racah9,
           u3::WigU3SO3_db& wigU3SO3db, u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
  {
    std::vector<Eigen::RowVectorXd> coeffs;
    std::vector<U3xU4::ads_phys> adsArr;
    std::vector<U3xU4::ats_phys> atsArr;

    // only ads
    if (ats == 0)
    {
      U3xU4::Tad_phys Tad(n, prod_ads, LL, SS, TT, MMT, JJ, MMJ);
      std::tie(coeffs,adsArr) = Tad.expand(cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);
    }

    // only ats
    else if (ads == 0)
    {
      U3xU4::Tat_phys Tat(n, prod_ats, LL, SS, TT, MMT, JJ, MMJ);
      std::tie(coeffs,atsArr) = Tat.expand(cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);
    }

    else
    {
      std::vector<u3::physicalvector> pv3s1, pv3s2;
      std::vector<u4::physicalvector> pv4s1, pv4s2;

      if (ads > 1)
      {
        pv3s1 = u3::physicalBasis(prod_ads.back().u3);
        pv4s1 = u4::physicalBasis(prod_ads.back().u4);
      }
      else
      {
        pv3s1 = u3::physicalBasis(u3::irrep(n,0));
        pv4s1 = u4::physicalBasis(u4::irrep(1,0,0));
      }

      if (ats > 1)
      {
        pv3s2 = u3::physicalBasis(prod_ats.back().u3);
        pv4s2 = u4::physicalBasis(prod_ats.back().u4);
      }
      else
      {
        pv3s2 = u3::physicalBasis(u3::irrep(0,n));
        pv4s2 = u4::physicalBasis(u4::irrep(0,0,1));
      }

      // uncouple the final tensor to Tad and Tat
      for (int pv31 = 0; pv31 < pv3s1.size(); pv31++)
      {
        int lam1 = pv3s1[pv31].toprow.lam(), 
            mu1 = pv3s1[pv31].toprow.mu(),
            kappa1 = pv3s1[pv31].kappa, 
            LL1 = 2 * pv3s1[pv31].L;

        for (int pv32 = 0; pv32 < pv3s2.size(); pv32++)
        {
          int lam2 = pv3s2[pv32].toprow.lam(), 
              mu2 = pv3s2[pv32].toprow.mu(),
              kappa2 = pv3s2[pv32].kappa,
              LL2 = 2 * pv3s2[pv32].L;

          if (std::abs(LL1 - LL2) > LL || (LL1 + LL2) < LL)
            continue;

          auto [allowU3, cgcU3] = wigU3SO3db.get(lam1, mu1, LL1/2,
                                                 lam2, mu2, LL2/2,
                                                 T.u3.lam(), T.u3.mu(), LL/2);

          for (int pv41 = 0; pv41 < pv4s1.size(); pv41++)
          {
            int P11 = pv4s1[pv41].toprow.P1(), 
                P21 = pv4s1[pv41].toprow.P2(),
                P31 = pv4s1[pv41].toprow.P3(),
                SS1 = pv4s1[pv41].SS,
                TT1 = pv4s1[pv41].TT;

            for (int pv42 = 0; pv42 < pv4s2.size(); pv42++)
            {
              int P12 = pv4s2[pv42].toprow.P1(), 
                  P22 = pv4s2[pv42].toprow.P2(),
                  P32 = pv4s2[pv42].toprow.P3(),
                  SS2 = pv4s2[pv42].SS,
                  TT2 = pv4s2[pv42].TT;

              if (std::abs(SS1 - SS2) > SS || (SS1 + SS2) < SS ||
                  std::abs(TT1 - TT2) > TT || (TT1 + TT2) < TT)
                continue;

              auto [allowU4, cgcU4] = wigU4SU2SU2db.get(P11, P21, P31, SS1, TT1,
                                                        P12, P22, P32, SS2, TT2,
                                                        T.u4.P1(), T.u4.P2(), T.u4.P3(), SS, TT);

              for (int JJ1 = LL1 + SS1; JJ1 >= std::abs(LL1 - SS1); JJ1-=2)
              {
                for (int JJ2 = LL2 + SS2; JJ2 >= std::abs(LL2 - SS2); JJ2-=2)
                {
                  if (std::abs(JJ1 - JJ2) > JJ || (JJ1 + JJ2) < JJ)
                    continue; 
                  double U9j = racah9.compute(LL1, LL2, LL,
                                              SS1, SS2, SS,
                                              JJ1, JJ2, JJ);

                  for (int MMJ1 = JJ1; MMJ1 >= -JJ1; MMJ1-=2)
                  {
                    int MMJ2 = MMJ - MMJ1;
                    double cgcJ = cgcs.compute(JJ1, JJ2, JJ, MMJ1, MMJ2, MMJ);

                    for (int MMT1 = TT1; MMT1 >= -TT1; MMT1-=2)
                    {
                      int MMT2 = MMT - MMT1;
                      double cgcT = cgcs.compute(TT1, TT2, TT, MMT1, MMT2, MMT);

                      U3xU4::Tad_phys Tad(n,prod_ads,LL1,SS1,TT1,MMT1,JJ1,MMJ1);
                      auto [coeffs1, adArr] = Tad.expand(cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

                      U3xU4::Tat_phys Tat(n,prod_ats,LL2,SS2,TT2,MMT2,JJ2,MMJ2);
                      auto [coeffs2, atArr] = Tat.expand(cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

                      for (int i = 0; i < coeffs1.size(); i++)
                      {
                        for (int j = 0; j < coeffs2.size(); j++)
                        {
                          Eigen::RowVectorXd x(rhomax * etamax * kappamax * zetamax);
                          x.setZero();
                          for (int rho = 0; rho < rhomax; rho++)
                            for (int eta = 0; eta < etamax; eta++)
                              for (int k = 0; k < kappamax; k++)
                                for (int z = 0; z < zetamax; z++)
                                {
                                  int index = ((rho * etamax + eta) * kappamax + k) * zetamax + z;
                                  for (int k1 = 0; k1 < kappa1; k1++)
                                    for (int k2 = 0; k2 < kappa2; k2++)
                                      x(index) += cgcU3((k1 * kappa2 + k2) * kappamax + k,rho)
                                                * cgcU4(z,eta) * U9j * cgcJ * cgcT
                                                * coeffs1[i](k1) * coeffs2[j](k2);
                                }

                          if (x.norm() < 1e-12)
                            continue;

                          coeffs.push_back(x);
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