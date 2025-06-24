#include <cassert>

#include <U3xU4SingleShellTensorsDecomposition.h>

namespace U3xU4
{
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

  // expand coupled product of creation tensors (ad's)
  // in terms of creation operators (no phase needed)
  std::pair<std::vector<double>, std::vector<std::vector<U3xU4::ad_gelfand>>>
  Tad_gelfand::expand(u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
                      u4::WigU4U3_db& wigU4U3db) const
  {
    /* Reminder
    sps3: Gelfand array for [n,0,0]
    sps4: Gelfand array for [1,0,0,0]
    */

    std::vector<double> coeffs;
    std::vector<std::vector<U3xU4::ad_gelfand>> tensorsArr;

    std::size_t num = prod_characters.size() + 1;

    // one ad
    if (num == 1)
    {
      coeffs.push_back(1.0);
      auto c = u3::findGelfand(g3, sps3), r = u4::findGelfand(g4, sps4);
      U3xU4::ad_gelfand ad(c, r, sps3, sps4);
      std::vector<U3xU4::ad_gelfand> Tad(1, ad);
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
          std::vector<U3xU4::ad_gelfand> adad;
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
          u3::gelfandBasis(prod_characters[0].first);
      std::vector<u4::gelfand> g4Arr0 =
          u4::gelfandBasis(prod_characters[0].second);

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

          std::vector<std::pair<u3::irrep, u4::irrep>> prod_0;
          prod_0.push_back(prod_characters[0]);

          U3xU4::Tad_gelfand Tad(n, sps3, sps4, prod_0, g3Arr0[g3_0], g4Arr0[g4_0]);

          auto [coeffs_0, tensors_0] = Tad.expand(cgcs, wigU3U2db, wigU4U3db);

          for (std::size_t k = 0; k < coeffs_0.size(); k++)
          {
            coeffs.push_back(coeffs_0[k] * cgcU3 * cgcU4);
            std::vector<U3xU4::ad_gelfand> tensorsArr_temp = tensors_0[k];
            tensorsArr_temp.push_back(ad3);
            tensorsArr.push_back(tensorsArr_temp);
          }
        }
      }
    }

    return std::make_pair(coeffs, tensorsArr);
  }

  // expand coupled product of annihilation tensors (at's)
  // in terms of annihilation operators (a's) via a phase
  std::pair<std::vector<double>, std::vector<std::vector<U3xU4::at_gelfand>>>
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
      auto c = u3::findGelfand(g3, sps3_conj), r = u4::findGelfand(g4, sps4_conj);
      U3xU4::at_gelfand at(c, r, sps3, sps3_conj, sps4, sps4_conj);
      auto phase = at.turnToAnnihilation();

      assert(!at.isTensor());

      std::vector<U3xU4::at_gelfand> Tat(1, at);
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
        int jj3_1 = sps3_conj[g3_1].Lambdax2(), mm3_1 = sps3_conj[g3_1].MLambdax2(),
            jj3_2 = sps3_conj[g3_2].Lambdax2(), mm3_2 = sps3_conj[g3_2].MLambdax2();

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
          int jj4_1 = sps4_conj[g4_1].Lambdax2(), mm4_1 = sps4_conj[g4_1].MLambdax2(),
              jj4_2 = sps4_conj[g4_2].Lambdax2(), mm4_2 = sps4_conj[g4_2].MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
              sps4_conj[g4_1].n14(), sps4_conj[g4_1].n24(), sps4_conj[g4_1].n34(),
              sps4_conj[g4_1].n44(), sps4_conj[g4_1].n13(), sps4_conj[g4_1].n23(),
              sps4_conj[g4_1].n33(), sps4_conj[g4_2].n14(), sps4_conj[g4_2].n24(),
              sps4_conj[g4_2].n34(), sps4_conj[g4_2].n44(), sps4_conj[g4_2].n13(),
              sps4_conj[g4_2].n23(), sps4_conj[g4_2].n33(), g4.n14(), g4.n24(), 
              g4.n34(), g4.n44(), g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false)
          	continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
              sps4_conj[g4_1].n13(), sps4_conj[g4_1].n23(), sps4_conj[g4_1].n33(),
              sps4_conj[g4_1].n12(), sps4_conj[g4_1].n22(), sps4_conj[g4_2].n13(),
              sps4_conj[g4_2].n23(), sps4_conj[g4_2].n33(), sps4_conj[g4_2].n12(),
              sps4_conj[g4_2].n22(), g4.n13(), g4.n23(), g4.n33(), g4.n12(), g4.n22());

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

          std::vector<U3xU4::at_gelfand> atat;
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
          u3::gelfandBasis(prod_characters[0].first);
      std::vector<u4::gelfand> g4Arr0 =
          u4::gelfandBasis(prod_characters[0].second);

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
            jj3_3 = sps3_conj[g3_3].Lambdax2(), mm3_3 = sps3_conj[g3_3].MLambdax2();

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
              jj4_3 = sps4_conj[g4_3].Lambdax2(), mm4_3 = sps4_conj[g4_3].MLambdax2();

          auto [allowU4, accU4] = wigU4U3db.get(
              g4Arr0[g4_0].n14(), g4Arr0[g4_0].n24(), g4Arr0[g4_0].n34(),
              g4Arr0[g4_0].n44(), g4Arr0[g4_0].n13(), g4Arr0[g4_0].n23(),
              g4Arr0[g4_0].n33(), sps4_conj[g4_3].n14(), sps4_conj[g4_3].n24(),
              sps4_conj[g4_3].n34(), sps4_conj[g4_3].n44(), sps4_conj[g4_3].n13(),
              sps4_conj[g4_3].n23(), sps4_conj[g4_3].n33(), g4.n14(), g4.n24(), 
              g4.n34(), g4.n44(), g4.n13(), g4.n23(), g4.n33());
          // accU4 should have only one column

          if (allowU4 == false)
          	continue;

          auto [allowU3_4, accU3_4] = wigU3U2db.get(
              g4Arr0[g4_0].n13(), g4Arr0[g4_0].n23(), g4Arr0[g4_0].n33(),
              g4Arr0[g4_0].n12(), g4Arr0[g4_0].n22(), sps4_conj[g4_3].n13(),
              sps4_conj[g4_3].n23(), sps4_conj[g4_3].n33(), sps4_conj[g4_3].n12(),
              sps4_conj[g4_3].n22(), g4.n13(), g4.n23(), g4.n33(), g4.n12(),
              g4.n22());

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

          std::vector<std::pair<u3::irrep, u4::irrep>> prod_0;
          prod_0.push_back(prod_characters[0]);

          U3xU4::Tat_gelfand Tat(n, sps3, sps3_conj, sps4, sps4_conj,
                                 prod_0, g3Arr0[g3_0], g4Arr0[g4_0]);

          auto [coeffs_0, tensors_0] = Tat.expand(cgcs, wigU3U2db, wigU4U3db);

          for (std::size_t k = 0; k < coeffs_0.size(); k++)
          {
            coeffs.push_back(phase3 * coeffs_0[k] * cgcU3 * cgcU4);
            std::vector<U3xU4::at_gelfand> tensorsArr_temp = tensors_0[k];
            tensorsArr_temp.push_back(at3);
            tensorsArr.push_back(tensorsArr_temp);
          }
        }
      }
    }

    return std::make_pair(coeffs, tensorsArr);
  }

  // expand coupled product of creation and annihilation tensors
  // in terms of creation and annihilation operators (ad's and a's)
  std::tuple<std::vector<Eigen::RowVectorXd>,             // coeffs
             std::vector<std::vector<U3xU4::ad_gelfand>>, // ad's
             std::vector<std::vector<U3xU4::at_gelfand>>> // a's
  Tadat_gelfand::expand(u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
                        u4::WigU4U3_db& wigU4U3db) 
  {
    std::vector<Eigen::RowVectorXd> coeffs;
    std::vector<std::vector<U3xU4::ad_gelfand>> adsArr;
    std::vector<std::vector<U3xU4::at_gelfand>> atsArr;

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
      U3xU4::Tat_gelfand Tat(n, sps3, sps3_conj, sps4, sps4_conj, prod_ats, g3, g4);

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
        g3Arr1 = u3::gelfandBasis(prod_ads.back().first);
        g4Arr1 = u4::gelfandBasis(prod_ads.back().second);
      }
      else 
      {
        g3Arr1 = sps3;
        g4Arr1 = sps4;
      }

      if (ats > 1)
      {
        g3Arr2 = u3::gelfandBasis(prod_ats.back().first);
        g4Arr2 = u4::gelfandBasis(prod_ats.back().second);
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
          U3xU4::Tad_gelfand Tad(n, sps3, sps4, prod_ads, g3Arr1[g3_1], g4Arr1[g4_1]);
          auto [coeffs_ads, adsArr_] = Tad.expand(cgcs, wigU3U2db, wigU4U3db);

          // uncouple Tat
          U3xU4::Tat_gelfand Tat(n, sps3, sps3_conj, sps4, sps4_conj, prod_ats, g3Arr2[g3_2], g4Arr2[g4_2]);
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

} // namespace U3xU4