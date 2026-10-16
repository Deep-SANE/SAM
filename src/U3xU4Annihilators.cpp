#include <cassert>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <readwrite.h>

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
    c_ = utils::itou8(u3::findGelfand(g3, sps3_));

    u4::gelfand g4(1, 0, 0, 0, 1 - n33p, 0, 0, 1 - n22p, 0, 1 - n11p);
    r_ = utils::itou8(u4::findGelfand(g4, sps4_));

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

  std::ostream& operator<<(std::ostream& s, const at_gelfand& x)
  {
    if (x.tensor)
      return s << "at(" << x.sps3_conj_[x.c_] 
               << " ; " << x.sps4_conj_[x.r_] << ")";
    else
      return s << "a(" << x.sps3_[x.c_] << " ; " << x.sps4_[x.r_] << ")";
  }

  std::ostream& operator<<(std::ostream& s, const at_phys& x)
  {
    auto [n, ll, jj, mmj, mmt] = x.getLabels();
    if (x.tensor)
      return s << "at(" << static_cast<int>(n) << "," 
               << static_cast<int>(ll) << "," 
               << static_cast<int>(jj) << "," 
               << static_cast<int>(mmj) << ","
               << static_cast<int>(mmt) << ")";
    else
      return s << "a(" << static_cast<int>(n) << ","
               << static_cast<int>(ll) << ","
               << static_cast<int>(jj) << ","
               << static_cast<int>(mmj) << ","
               << static_cast<int>(mmt) << ")";
  }
  // expand coupled product of annihilation tensors (at's)
  // in terms of annihilation operators (a's) via a phase
  std::pair<std::vector<double>, std::vector<U3xU4::ats_gelfand>>
  Tat_gelfand::expand() const
  {
    auto& cgcs = utils::cgcs();
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
    auto& cgcs = utils::cgcs();
    auto& racah9 = utils::fanos();
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

} // namespace U3xU4
