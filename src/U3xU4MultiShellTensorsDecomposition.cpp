#include <cassert>

#include <U3xU4MultiShellTensors.h>
#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  //////////////////// TWO-BODY MULTI-SHELL TENSORS ////////////////////

  // Expand into uncoupled products of creation operators (ad)
  // and annihilation operators (a)
  // coeffs have rhomax x etamax x kappamax x zetamax columns, index of each
  // column is
  // ((rho * etamax + eta) * kappamax + kappa) * zetamax + zeta
  std::pair<std::vector<Eigen::RowVectorXd>,  // coeffs
            std::vector<U3xU4::ads_ats_phys>> // list of ad's and a's
  TwoBodyMultiShellTensor::expand(u34utils::SO3_CGC& cgcs,
                                  u34utils::SO3_Racah9& racah9,
                                  u3::WigU3SO3_db& wigU3SO3db,
                                  u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
  {
    std::vector<Eigen::RowVectorXd> coeffsArr;
    std::vector<U3xU4::ads_ats_phys> opsArr;
    // recall: ads_ats_phys =
    // std::vector<std::variant<U3xU4::ad_phys,U3xU4::at_phys>>;

    if (coupled_irreps.size() == 0)
    {
      // Delegate to single-shell expansion
      const U3xU4::Tadat& tadat = single_shell_tensors[0];
      U3xU4::Tadat_phys tadat_phys(tadat.n, tadat.ads, tadat.ats,
                                   tadat.prod_ads, tadat.prod_ats, tadat.T,
                                   rhomax, etamax, LL, SS, TT, MMT, JJ, MMJ, labels);

      auto [coeffs, ads, ats] =
          tadat_phys.expand(cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

      for (int i = 0; i < coeffs.size(); i++)
      {
        U3xU4::ads_ats_phys ops;
        if (!ads.empty())
          for (const auto& ad : ads[i])
            ops.emplace_back(ad);
        if (!ats.empty())
          for (const auto& a : ats[i])
            ops.emplace_back(a);

        coeffsArr.push_back(std::move(coeffs[i]));
        opsArr.push_back(std::move(ops));
      }
    }
    else if (coupled_irreps.size() == 1)
    {

      U3xU4::irrep T = coupled_irreps[0];

      // There are two single-shell tensors in this case
      U3xU4::Tadat tadat1 = single_shell_tensors[0],
                   tadat2 = single_shell_tensors[1];

      assert(tadat1.rho0max == 1 && tadat1.eta0max == 1);
      assert(tadat2.rho0max == 1 && tadat2.eta0max == 1);

      std::vector<u3::physicalvector> pv3s1 = u3::physicalBasis(tadat1.T.u3),
                                      pv3s2 = u3::physicalBasis(tadat2.T.u3);

      std::vector<u4::physicalvector> pv4s1 = u4::physicalBasis(tadat1.T.u4),
                                      pv4s2 = u4::physicalBasis(tadat2.T.u4);

      for (int pv31 = 0; pv31 < pv3s1.size(); pv31++)
      {
        int lam1 = pv3s1[pv31].toprow.lam(), mu1 = pv3s1[pv31].toprow.mu(),
            kappa1 = pv3s1[pv31].kappa, LL1 = 2 * pv3s1[pv31].L;

        for (int pv32 = 0; pv32 < pv3s2.size(); pv32++)
        {
          int lam2 = pv3s2[pv32].toprow.lam(), mu2 = pv3s2[pv32].toprow.mu(),
              kappa2 = pv3s2[pv32].kappa, LL2 = 2 * pv3s2[pv32].L;

          if (std::abs(LL1 - LL2) > LL || (LL1 + LL2) < LL)
            continue;

          auto [allowU3, cgcU3] =
              wigU3SO3db.get(lam1, mu1, LL1 / 2, lam2, mu2, LL2 / 2, T.u3.lam(),
                             T.u3.mu(), LL / 2);

          for (int pv41 = 0; pv41 < pv4s1.size(); pv41++)
          {
            int P11 = pv4s1[pv41].toprow.P1(), P21 = pv4s1[pv41].toprow.P2(),
                P31 = pv4s1[pv41].toprow.P3(), SS1 = pv4s1[pv41].SS,
                TT1 = pv4s1[pv41].TT, zeta1 = pv4s1[pv41].zeta;

            for (int pv42 = 0; pv42 < pv4s2.size(); pv42++)
            {
              int P12 = pv4s2[pv42].toprow.P1(), P22 = pv4s2[pv42].toprow.P2(),
                  P32 = pv4s2[pv42].toprow.P3(), SS2 = pv4s2[pv42].SS,
                  TT2 = pv4s2[pv42].TT, zeta2 = pv4s2[pv42].zeta;

              if (std::abs(SS1 - SS2) > SS || (SS1 + SS2) < SS ||
                  std::abs(TT1 - TT2) > TT || (TT1 + TT2) < TT)
                continue;

              auto [allowU4, cgcU4] = wigU4SU2SU2db.get(
                  P11, P21, P31, SS1, TT1, P12, P22, P32, SS2, TT2, T.u4.P1(),
                  T.u4.P2(), T.u4.P3(), SS, TT);

              for (int JJ1 = LL1 + SS1; JJ1 >= std::abs(LL1 - SS1); JJ1 -= 2)
              {
                for (int JJ2 = LL2 + SS2; JJ2 >= std::abs(LL2 - SS2); JJ2 -= 2)
                {
                  if (std::abs(JJ1 - JJ2) > JJ || (JJ1 + JJ2) < JJ)
                    continue;
                  double U9j =
                      racah9.compute(LL1, LL2, LL, SS1, SS2, SS, JJ1, JJ2, JJ);

                  for (int MMJ1 = JJ1; MMJ1 >= -JJ1; MMJ1 -= 2)
                  {
                    int MMJ2 = MMJ - MMJ1;
                    if (std::abs(MMJ2) > JJ2)
                      continue;

                    double cgcJ = cgcs.compute(JJ1, JJ2, JJ, MMJ1, MMJ2, MMJ);

                    for (int MMT1 = TT1; MMT1 >= -TT1; MMT1 -= 2)
                    {
                      int MMT2 = MMT - MMT1;
                      if (std::abs(MMT2) > TT2)
                        continue;

                      double cgcT = cgcs.compute(TT1, TT2, TT, MMT1, MMT2, MMT);

                      U3xU4::Tadat_phys tadat_phys1(
                          tadat1.n, tadat1.ads, tadat1.ats, tadat1.prod_ads,
                          tadat1.prod_ats, tadat1.T, tadat1.rho0max,
                          tadat1.eta0max, LL1, SS1, TT1, MMT1, JJ1, MMJ1, labels);
                      auto [coeffsArr1, adsArr1, asArr1] = tadat_phys1.expand(
                          cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

                      U3xU4::Tadat_phys tadat_phys2(
                          tadat2.n, tadat2.ads, tadat2.ats, tadat2.prod_ads,
                          tadat2.prod_ats, tadat2.T, tadat2.rho0max,
                          tadat2.eta0max, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels);
                      auto [coeffsArr2, adsArr2, asArr2] = tadat_phys2.expand(
                          cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

                      for (int i = 0; i < coeffsArr1.size(); i++)
                      {
                        for (int j = 0; j < coeffsArr2.size(); j++)
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
                                      for (int z1 = 0; z1 < zeta1; z1++)
                                        for (int z2 = 0; z2 < zeta2; z2++)
                                        {
                                          x(index) +=
                                              cgcU3((k1 * kappa2 + k2) *
                                                            kappamax +
                                                        k,
                                                    rho) *
                                              cgcU4((z1 * zeta2 + z2) *
                                                            zetamax +
                                                        z,
                                                    eta) *
                                              U9j * cgcJ * cgcT *
                                              coeffsArr1[i](k1 * zeta1 + z1) *
                                              coeffsArr2[j](k2 * zeta2 + z2);
                                        }
                                }

                          if (x.norm() < 1e-12)
                            continue;

                          U3xU4::ads_ats_phys ops;
                          if (!adsArr1.empty())
                            for (const auto& ad1 : adsArr1[i])
                              ops.emplace_back(ad1);
                          if (!asArr1.empty())
                            for (const auto& a1 : asArr1[i])
                              ops.emplace_back(a1);
                          if (!adsArr2.empty())
                            for (const auto& ad2 : adsArr2[j])
                              ops.emplace_back(ad2);
                          if (!asArr2.empty())
                            for (const auto& a2 : asArr2[j])
                              ops.emplace_back(a2);

                          coeffsArr.push_back(std::move(x));
                          opsArr.push_back(std::move(ops));
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
    else if (coupled_irreps.size() >= 2)
    {
      // There are three or more single-shell tensors in this case

      // Extract the last coupled irrep
      U3xU4::irrep T = coupled_irreps.back();
      // Take out the rest
      std::vector<U3xU4::irrep> coupled_irreps_new = coupled_irreps;
      coupled_irreps_new.pop_back();

      // Extract the last single-shell tensor
      U3xU4::Tadat tadat = single_shell_tensors.back();
      assert(tadat.rho0max == 1 && tadat.eta0max == 1);
      // Take out the rest
      std::vector<U3xU4::Tadat> single_shell_tensors_new = single_shell_tensors;
      single_shell_tensors_new.pop_back();
      assert(single_shell_tensors_new.back().rho0max == 1 &&
             single_shell_tensors_new.back().eta0max == 1);

      std::vector<u3::physicalvector> pv3s1 = u3::physicalBasis(
                                          coupled_irreps_new.back().u3),
                                      pv3s2 = u3::physicalBasis(tadat.T.u3);

      std::vector<u4::physicalvector> pv4s1 = u4::physicalBasis(
                                          coupled_irreps_new.back().u4),
                                      pv4s2 = u4::physicalBasis(tadat.T.u4);

      for (int pv31 = 0; pv31 < pv3s1.size(); pv31++)
      {
        int lam1 = pv3s1[pv31].toprow.lam(), mu1 = pv3s1[pv31].toprow.mu(),
            kappa1 = pv3s1[pv31].kappa, LL1 = 2 * pv3s1[pv31].L;

        for (int pv32 = 0; pv32 < pv3s2.size(); pv32++)
        {
          int lam2 = pv3s2[pv32].toprow.lam(), mu2 = pv3s2[pv32].toprow.mu(),
              kappa2 = pv3s2[pv32].kappa, LL2 = 2 * pv3s2[pv32].L;

          if (std::abs(LL1 - LL2) > LL || (LL1 + LL2) < LL)
            continue;

          auto [allowU3, cgcU3] =
              wigU3SO3db.get(lam1, mu1, LL1 / 2, lam2, mu2, LL2 / 2, T.u3.lam(),
                             T.u3.mu(), LL / 2);

          for (int pv41 = 0; pv41 < pv4s1.size(); pv41++)
          {
            int P11 = pv4s1[pv41].toprow.P1(), P21 = pv4s1[pv41].toprow.P2(),
                P31 = pv4s1[pv41].toprow.P3(), SS1 = pv4s1[pv41].SS,
                TT1 = pv4s1[pv41].TT, zeta1 = pv4s1[pv41].zeta;

            for (int pv42 = 0; pv42 < pv4s2.size(); pv42++)
            {
              int P12 = pv4s2[pv42].toprow.P1(), P22 = pv4s2[pv42].toprow.P2(),
                  P32 = pv4s2[pv42].toprow.P3(), SS2 = pv4s2[pv42].SS,
                  TT2 = pv4s2[pv42].TT, zeta2 = pv4s2[pv42].zeta;

              if (std::abs(SS1 - SS2) > SS || (SS1 + SS2) < SS ||
                  std::abs(TT1 - TT2) > TT || (TT1 + TT2) < TT)
                continue;

              auto [allowU4, cgcU4] = wigU4SU2SU2db.get(
                  P11, P21, P31, SS1, TT1, P12, P22, P32, SS2, TT2, T.u4.P1(),
                  T.u4.P2(), T.u4.P3(), SS, TT);

              for (int JJ1 = LL1 + SS1; JJ1 >= std::abs(LL1 - SS1); JJ1 -= 2)
              {
                for (int JJ2 = LL2 + SS2; JJ2 >= std::abs(LL2 - SS2); JJ2 -= 2)
                {
                  if (std::abs(JJ1 - JJ2) > JJ || (JJ1 + JJ2) < JJ)
                    continue;
                  double U9j =
                      racah9.compute(LL1, LL2, LL, SS1, SS2, SS, JJ1, JJ2, JJ);

                  for (int MMJ1 = JJ1; MMJ1 >= -JJ1; MMJ1 -= 2)
                  {
                    int MMJ2 = MMJ - MMJ1;
                    if (std::abs(MMJ2) > JJ2)
                      continue;

                    double cgcJ = cgcs.compute(JJ1, JJ2, JJ, MMJ1, MMJ2, MMJ);

                    for (int MMT1 = TT1; MMT1 >= -TT1; MMT1 -= 2)
                    {
                      int MMT2 = MMT - MMT1;
                      if (std::abs(MMT2) > TT2)
                        continue;

                      double cgcT = cgcs.compute(TT1, TT2, TT, MMT1, MMT2, MMT);

                      // Even though the extracted tensor is no longer a
                      // two-body multi-shell tensor, we can still utilize its
                      // structure to uncouple everything. This extracted tensor
                      // is supposed to have multiplicities rhomax == etamax ==
                      // 1
                      U3xU4::TwoBodyMultiShellTensor new_multi_shell_tensor(
                          single_shell_tensors_new, coupled_irreps_new, 1, 1,
                          kappa1, LL1, zeta1, SS1, TT1, MMT1, JJ1, MMJ1, {}, labels);

                      auto [coeffsArr1, opsArr1] =
                          new_multi_shell_tensor.expand(
                              cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

                      U3xU4::Tadat_phys tadat_phys(
                          tadat.n, tadat.ads, tadat.ats, tadat.prod_ads,
                          tadat.prod_ats, tadat.T, tadat.rho0max, tadat.eta0max,
                          LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels);
                      auto [coeffsArr2, adsArr2, asArr2] = tadat_phys.expand(
                          cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

                      for (int i = 0; i < coeffsArr1.size(); i++)
                      {
                        for (int j = 0; j < coeffsArr2.size(); j++)
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
                                      for (int z1 = 0; z1 < zeta1; z1++)
                                        for (int z2 = 0; z2 < zeta2; z2++)
                                          x(index) +=
                                              cgcU3((k1 * kappa2 + k2) *
                                                            kappamax +
                                                        k,
                                                    rho) *
                                              cgcU4((z1 * zeta2 + z2) *
                                                            zetamax +
                                                        z,
                                                    eta) *
                                              U9j * cgcJ * cgcT *
                                              coeffsArr1[i](k1 * zeta1 + z1) *
                                              coeffsArr2[j](k2 * zeta2 + z2);
                                }

                          if (x.norm() < 1e-12)
                            continue;

                          assert(!opsArr1[i].empty());

                          U3xU4::ads_ats_phys ops = opsArr1[i];
                          if (!adsArr2.empty())
                            for (const auto& ad2 : adsArr2[j])
                              ops.emplace_back(ad2);
                          if (!asArr2.empty())
                            for (const auto& a2 : asArr2[j])
                              ops.emplace_back(a2);

                          coeffsArr.push_back(std::move(x));
                          opsArr.push_back(std::move(ops));
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

    return std::make_pair(coeffsArr, opsArr);
  }
} // namespace U3xU4
