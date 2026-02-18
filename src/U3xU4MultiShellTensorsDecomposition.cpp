#include <cassert>

#include <qcx_ND.h>
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

#include <U3SO3xU3SO3.h>
#include <U4SU2SU2xU4SU2SU2.h>

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
  TwoBodyMultiShellTensor::expand() const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::racahs9();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    std::vector<Eigen::RowVectorXd> coeffsArr;
    std::vector<U3xU4::ads_ats_phys> opsArr;
    // recall: ads_ats_phys =
    // std::vector<std::variant<U3xU4::ad_phys,U3xU4::at_phys>>;

    if (Omegas_.size() == 0)
    {
      // Delegate to single-shell expansion
      const U3xU4::SingleShellTensor& tadat = Gammas_[0];
      U3xU4::SingleShellTensor_phys tadat_phys(tadat.shell(), tadat.num_ads(), 
        tadat.num_ats(), tadat.Omega_ads(), tadat.Omega_ats(), tadat.T(), 
        rhomax(), etamax(), LL(), SS(), TT(), MMT(), JJ(), MMJ(), labels_);

      auto [coeffs, ads, ats] = tadat_phys.expand();

      for (size_t i = 0; i < coeffs.size(); i++)
      {
        U3xU4::ads_ats_phys ops;
        if (!ads.empty())
          for (const auto& ad : ads[i]) ops.emplace_back(ad);
        if (!ats.empty())
          for (const auto& a : ats[i]) ops.emplace_back(a);

        coeffsArr.push_back(std::move(coeffs[i]));
        opsArr.push_back(std::move(ops));
      }
    }
    else if (Omegas_.size() == 1)
    {
      const U3xU4::irrep& T = Omegas_[0];

      // There are two single-shell tensors in this case
      const U3xU4::SingleShellTensor& tadat1 = Gammas_[0], tadat2 = Gammas_[1];

      assert(tadat1.rho0max() == 1 && tadat1.eta0max() == 1);
      assert(tadat2.rho0max() == 1 && tadat2.eta0max() == 1);

      u3::physicalvectors pv3s1 = u3::physicalBasis(tadat1.u3()),
                          pv3s2 = u3::physicalBasis(tadat2.u3());

      u4::physicalvectors pv4s1 = u4::physicalBasis(tadat1.u4()),
                          pv4s2 = u4::physicalBasis(tadat2.u4());

      for (const auto& pv31 : pv3s1)
      {
        int lam1 = pv31.lam(), mu1 = pv31.mu(),
            kappa1 = pv31.kappa(), LL1 = 2 * pv31.L();

        for (const auto& pv32 : pv3s2)
        {
          int lam2 = pv32.lam(), mu2 = pv32.mu(),
              kappa2 = pv32.kappa(), LL2 = 2 * pv32.L();

          // if (abs(LL1 - LL2) > LL_ || (LL1 + LL2) < LL_) continue;

          const auto cgcU3 = wigU3SO3db.get_U3SO3(lam1, mu1, LL1 / 2, 
            lam2, mu2, LL2 / 2, T.lam(), T.mu(), LL() / 2);

          if (cgcU3.empty()) continue;

          for (const auto& pv41 : pv4s1)
          {
            int P11 = pv41.P1(), P21 = pv41.P2(), P31 = pv41.P3(), 
                zeta1 = pv41.zeta(), SS1 = pv41.SS(), TT1 = pv41.TT();

            for (const auto& pv42 : pv4s2)
            {
              int P12 = pv42.P1(), P22 = pv42.P2(), P32 = pv42.P3(), 
                  zeta2 = pv42.zeta(), SS2 = pv42.SS(), TT2 = pv42.TT();

              // if (abs(SS1 - SS2) > SS_ || (SS1 + SS2) < SS_ ||
              //     abs(TT1 - TT2) > TT_ || (TT1 + TT2) < TT_)
              //   continue;

              const auto cgcU4 = wigU4SU2SU2db.get_U4SU2SU2(
                  P11, P21, P31, SS1, TT1, P12, P22, P32, SS2, TT2,
                  T.P1(), T.P2(), T.P3(), SS(), TT());

              if (cgcU4.empty()) continue;

              for (int JJ1 = LL1 + SS1; JJ1 >= abs(LL1 - SS1); JJ1 -= 2)
              {
                for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                {
                  if (abs(JJ1 - JJ2) > JJ_ || (JJ1 + JJ2) < JJ_) continue;
                  
                  double U9j = racah9.get({LL1, LL2, LL(),
                                              SS1, SS2, SS(),
                                              JJ1, JJ2, JJ()});

                  for (int MMJ1 = JJ1; MMJ1 >= -JJ1; MMJ1 -= 2)
                  {
                    int MMJ2 = MMJ() - MMJ1;
                    if (abs(MMJ2) > JJ2) continue;

                    double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                    for (int MMT1 = TT1; MMT1 >= -TT1; MMT1 -= 2)
                    {
                      int MMT2 = MMT() - MMT1;
                      if (abs(MMT2) > TT2) continue;

                      double cgcT = cgcs.get({TT1, TT2, TT(), MMT1, MMT2, MMT()});

                      U3xU4::SingleShellTensor_phys tadat_phys1(
                        tadat1.shell(), tadat1.num_ads(), tadat1.num_ats(),
                        tadat1.Omega_ads(), tadat1.Omega_ats(), tadat1.T(),
                        tadat1.rho0max(), tadat1.eta0max(), LL1, SS1, TT1,
                        MMT1, JJ1, MMJ1, labels_);
                      auto [coeffsArr1, adsArr1, asArr1] = tadat_phys1.expand();

                      U3xU4::SingleShellTensor_phys tadat_phys2(
                          tadat2.shell(), tadat2.num_ads(), tadat2.num_ats(),
                          tadat2.Omega_ads(), tadat2.Omega_ats(), tadat2.T(),
                          tadat2.rho0max(), tadat2.eta0max(), LL2, SS2, TT2,
                          MMT2, JJ2, MMJ2, labels_);
                      auto [coeffsArr2, adsArr2, asArr2] = tadat_phys2.expand();

                      for (size_t i = 0; i < coeffsArr1.size(); i++)
                      {
                        for (size_t j = 0; j < coeffsArr2.size(); j++)
                        {
                          Eigen::RowVectorXd x(rhomax() * etamax() * 
                                               kappamax() * zetamax());
                          x.setZero();
                          for (int rho = 0; rho < rhomax(); rho++)
                            for (int eta = 0; eta < etamax(); eta++)
                              for (int k = 0; k < kappamax(); k++)
                                for (int z = 0; z < zetamax(); z++)
                                {
                                  int index =
                                      ((rho * etamax() + eta) * kappamax() + k) *
                                          zetamax() + z;

                                  for (int k1 = 0; k1 < kappa1; k1++)
                                    for (int k2 = 0; k2 < kappa2; k2++)
                                      for (int z1 = 0; z1 < zeta1; z1++)
                                        for (int z2 = 0; z2 < zeta2; z2++)
                                        {
                                          x(index) +=
                                              cgcU3(k1, k2, k, rho) *
                                              cgcU4(z1, z2, z, eta) *
                                              U9j * cgcJ * cgcT *
                                              coeffsArr1[i](k1 * zeta1 + z1) *
                                              coeffsArr2[j](k2 * zeta2 + z2);
                                        }
                                }

                          if (x.norm() < 1e-12) continue;

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
    else if (Omegas_.size() >= 2)
    {
      // There are three or more single-shell tensors in this case

      // Extract the last coupled irrep
      const U3xU4::irrep& T = Omegas_.back();
      // Take out the rest
      U3xU4::irreps Omegas_new = Omegas_;
      Omegas_new.pop_back();

      // Extract the last single-shell tensor
      const U3xU4::SingleShellTensor& tadat = Gammas_.back();
      assert(tadat.rho0max() == 1 && tadat.eta0max() == 1);
      // Take out the rest
      U3xU4::SingleShellTensors Gammas_new = Gammas_;
      Gammas_new.pop_back();
      assert(Gammas_new.back().rho0max() == 1 &&
             Gammas_new.back().eta0max() == 1);

      u3::physicalvectors pv3s1 = u3::physicalBasis(Omegas_new.back().u3()),
                          pv3s2 = u3::physicalBasis(tadat.u3());

      u4::physicalvectors pv4s1 = u4::physicalBasis(Omegas_new.back().u4()),
                          pv4s2 = u4::physicalBasis(tadat.u4());

      for (const auto& pv31 : pv3s1)
      {
        int lam1 = pv31.lam(), mu1 = pv31.mu(),
            kappa1 = pv31.kappa(), LL1 = 2 * pv31.L();

        for (const auto& pv32 : pv3s2)
        {
          int lam2 = pv32.lam(), mu2 = pv32.mu(),
              kappa2 = pv32.kappa(), LL2 = 2 * pv32.L();

          // if (abs(LL1 - LL2) > LL_ || (LL1 + LL2) < LL_) continue;

          const auto cgcU3 = wigU3SO3db.get_U3SO3(lam1, mu1, LL1 / 2, 
                      lam2, mu2, LL2 / 2, T.lam(), T.mu(), LL() / 2);

          if (cgcU3.empty()) continue;

          for (const auto& pv41 : pv4s1)
          {
            int P11 = pv41.P1(), P21 = pv41.P2(), P31 = pv41.P3(),
                zeta1 = pv41.zeta(), SS1 = pv41.SS(), TT1 = pv41.TT();

            for (const auto& pv42 : pv4s2)
            {
              int P12 = pv42.P1(), P22 = pv42.P2(), P32 = pv42.P3(), 
                  zeta2 = pv42.zeta(), SS2 = pv42.SS(), TT2 = pv42.TT();

              // if (abs(SS1 - SS2) > SS_ || (SS1 + SS2) < SS_ ||
              //     abs(TT1 - TT2) > TT_ || (TT1 + TT2) < TT_)
              //   continue;

              const auto cgcU4 = wigU4SU2SU2db.get_U4SU2SU2(
                  P11, P21, P31, SS1, TT1, P12, P22, P32, SS2, TT2,
                  T.P1(), T.P2(), T.P3(), SS(), TT());

              if (cgcU4.empty()) continue;

              for (int JJ1 = LL1 + SS1; JJ1 >= abs(LL1 - SS1); JJ1 -= 2)
              {
                for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                {
                  if (abs(JJ1 - JJ2) > JJ_ || (JJ1 + JJ2) < JJ_) continue;

                  double U9j = racah9.get({LL1, LL2, LL(),
                                              SS1, SS2, SS(),
                                              JJ1, JJ2, JJ()});

                  for (int MMJ1 = JJ1; MMJ1 >= -JJ1; MMJ1 -= 2)
                  {
                    int MMJ2 = MMJ() - MMJ1;
                    if (abs(MMJ2) > JJ2) continue;

                    double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                    for (int MMT1 = TT1; MMT1 >= -TT1; MMT1 -= 2)
                    {
                      int MMT2 = MMT() - MMT1;
                      if (abs(MMT2) > TT2) continue;

                      double cgcT = cgcs.get({TT1, TT2, TT(), MMT1, MMT2, MMT()});

                      // Even though the extracted tensor is no longer a
                      // two-body multi-shell tensor, we can still utilize its
                      // structure to uncouple everything. This extracted tensor
                      // is supposed to have multiplicities rhomax == etamax ==
                      // 1
                      U3xU4::TwoBodyMultiShellTensor new_multi_shell_tensor(
                          Gammas_new, Omegas_new, 1, 1,
                          kappa1, LL1, zeta1, SS1, TT1, MMT1, JJ1, MMJ1, {},
                          labels_);

                      auto [coeffsArr1, opsArr1] =
                          new_multi_shell_tensor.expand();

                      U3xU4::SingleShellTensor_phys tadat_phys(
                          tadat.shell(), tadat.num_ads(), tadat.num_ats(),
                          tadat.Omega_ads(), tadat.Omega_ats(), tadat.T(), 
                          tadat.rho0max(), tadat.eta0max(), LL2, SS2, TT2, MMT2,
                          JJ2, MMJ2, labels_);
                      auto [coeffsArr2, adsArr2, asArr2] =
                          tadat_phys.expand();

                      for (size_t i = 0; i < coeffsArr1.size(); i++)
                      {
                        for (size_t j = 0; j < coeffsArr2.size(); j++)
                        {
                          Eigen::RowVectorXd x(rhomax() * etamax() * 
                                               kappamax() * zetamax());
                          x.setZero();
                          for (int rho = 0; rho < rhomax(); rho++)
                            for (int eta = 0; eta < etamax(); eta++)
                              for (int k = 0; k < kappamax(); k++)
                                for (int z = 0; z < zetamax(); z++)
                                {
                                  int index =
                                      ((rho * etamax() + eta) * kappamax() + k) *
                                          zetamax() + z;
                                  for (int k1 = 0; k1 < kappa1; k1++)
                                    for (int k2 = 0; k2 < kappa2; k2++)
                                      for (int z1 = 0; z1 < zeta1; z1++)
                                        for (int z2 = 0; z2 < zeta2; z2++)
                                          x(index) +=
                                              cgcU3(k1, k2, k, rho) *
                                              cgcU4(z1, z2, z, eta) *
                                              U9j * cgcJ * cgcT *
                                              coeffsArr1[i](k1 * zeta1 + z1) *
                                              coeffsArr2[j](k2 * zeta2 + z2);
                                }

                          if (x.norm() < 1e-12) continue;

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

    return std::make_pair(std::move(coeffsArr), std::move(opsArr));
  }

  //////////////////// GENERIC MULTI-SHELL TENSORS ////////////////////

  // Expand into uncoupled products of creation operators (ad)
  // and annihilation operators (a)
  std::pair<std::vector<Eigen::RowVectorXd>,  // coeffs
            std::vector<U3xU4::ads_ats_phys>> // list of ad's and a's
  MultiShellTensor_phys::expand(int LL, int SS, int TT) const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::racahs9();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    std::vector<Eigen::RowVectorXd> coeffsArr;
    std::vector<U3xU4::ads_ats_phys> opsArr;
    int estimate = 1;
    for (const auto& G : Gammas())
    {
      int n = G.shell(), shell_dim = 2 * (n + 1) * (n + 2);
      for (int i = 0; i < G.num_ads(); i++) estimate *= shell_dim;
      for (int i = 0; i < G.num_ats(); i++) estimate *= shell_dim;
    }
    coeffsArr.reserve(estimate);
    opsArr.reserve(estimate);

    if (Omegas().size() == 0)
    {
      // Delegate to single-shell expansion
      const U3xU4::SingleShellTensor& t = Gammas()[0];
      U3xU4::SingleShellTensor_phys t_phys(t, LL, SS, TT, MMT(), JJ(), MMJ(), labels_);

      auto [coeffs, ads, ats] = t_phys.expand();

      for (size_t i = 0; i < coeffs.size(); i++)
      {
        U3xU4::ads_ats_phys ops;
        if (!ads.empty())
          for (const auto& ad : ads[i]) ops.emplace_back(ad);
        if (!ats.empty())
          for (const auto& a : ats[i]) ops.emplace_back(a);

        opsArr.push_back(std::move(ops));
      }

      coeffsArr = std::move(coeffs);
    }
    
    else if (Omegas().size() == 1)
    {
      // Final coupling
      const U3xU4::OmegaElement& T = Omegas()[0];
      int kappamax = u3::Kappa(T.u3(), LL / 2);
      int zetamax = u4::Zeta(T.u4(), SS, TT);

      assert(kappamax > 0 && zetamax > 0);

      // There are two single-shell tensors in this case
      const U3xU4::SingleShellTensor& t1 = Gammas()[0];
      int alphamax1 = t1.rho0max() * t1.eta0max();
      const U3xU4::SingleShellTensor& t2 = Gammas()[1];
      int alphamax2 = t2.rho0max() * t2.eta0max();

      // Multiplicities
      qcx_utils::NDIndex6 multis;
      multis.add_dimensions(alphamax1, alphamax2, T.rhomax(), 
                            T.etamax(), kappamax, zetamax);
      size_t n_multis = multis.cardinality();

      // Construct physical bases
      u3::physicalvectors pv3s1 = u3::physicalBasis(t1.u3()),
                          pv3s2 = u3::physicalBasis(t2.u3());

      u4::physicalvectors pv4s1 = u4::physicalBasis(t1.u4()),
                          pv4s2 = u4::physicalBasis(t2.u4());      

      for (const auto& pv31 : pv3s1)
      {
        int lam1 = pv31.lam(), mu1 = pv31.mu(),
            kappa1 = pv31.kappa(), LL1 = 2 * pv31.L();

        for (const auto& pv32 : pv3s2)
        {
          int lam2 = pv32.lam(), mu2 = pv32.mu(),
              kappa2 = pv32.kappa(), LL2 = 2 * pv32.L();

          // if (abs(LL1 - LL2) > LL || (LL1 + LL2) < LL) continue;

          const auto cgcU3 = wigU3SO3db.get_U3SO3(lam1, mu1, LL1 / 2, 
            lam2, mu2, LL2 / 2, T.lam(), T.mu(), LL / 2);

          if (cgcU3.empty()) continue;

          for (const auto& pv41 : pv4s1)
          {
            int P11 = pv41.P1(), P21 = pv41.P2(), P31 = pv41.P3(), 
                zeta1 = pv41.zeta(), SS1 = pv41.SS(), TT1 = pv41.TT();

            for (const auto& pv42 : pv4s2)
            {
              int P12 = pv42.P1(), P22 = pv42.P2(), P32 = pv42.P3(), 
                  zeta2 = pv42.zeta(), SS2 = pv42.SS(), TT2 = pv42.TT();

              // if (abs(SS1 - SS2) > SS_ || (SS1 + SS2) < SS_ ||
              //     abs(TT1 - TT2) > TT_ || (TT1 + TT2) < TT_)
              //   continue;

              const auto cgcU4 = wigU4SU2SU2db.get_U4SU2SU2(
                  P11, P21, P31, SS1, TT1, P12, P22, P32, SS2, TT2,
                  T.P1(), T.P2(), T.P3(), SS, TT);

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
                    double cgcT = cgcs.get({TT1, TT2, TT, MMT1, MMT2, MMT()});
                    
                    U3xU4::SingleShellTensor_phys 
                      t1_phys(t1, LL1, SS1, TT1, MMT1, JJ1, MMJ1, labels_);

                    std::vector<Eigen::RowVectorXd> coeffsArr1;
                    std::vector<U3xU4::ads_phys> adsArr1;
                    std::vector<U3xU4::ats_phys> asArr1;

                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ()) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;
                  
                      double U9j = racah9.get({LL1, LL2, LL,
                                              SS1, SS2, SS,
                                              JJ1, JJ2, JJ()});
                      
                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (coeffsArr1.empty())
                      {
                        // std::cout << "Start expanding t1: " << t1_phys << std::endl;
                        std::tie(coeffsArr1, adsArr1, asArr1) = t1_phys.expand();
                        // std::cout << "Finish expanding t1" << std::endl;
                      }

                      U3xU4::SingleShellTensor_phys 
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);   

                      // std::cout << "Start expanding t2: " << t2_phys << std::endl;
                      auto [coeffsArr2, adsArr2, asArr2] = t2_phys.expand();
                      // std::cout << "Finish expanding t2" << std::endl;

                      for (size_t i = 0; i < coeffsArr1.size(); i++)
                      {
                        for (size_t j = 0; j < coeffsArr2.size(); j++)
                        {
                          Eigen::RowVectorXd x(n_multis);
                          x.setZero();
                                                    
                          for (size_t k = 0; k < n_multis; k++)
                          {
                            std::array<size_t, 6> multi;
                            multis.to_multi(k, multi);

                            auto alpha1 = multi[0];
                            auto alpha2 = multi[1];
                            auto rho = multi[2];
                            auto eta = multi[3];
                            auto kappa = multi[4];
                            auto zeta = multi[5];

                            for (int k1 = 0; k1 < kappa1; k1++)
                              for (int k2 = 0; k2 < kappa2; k2++)
                                for (int z1 = 0; z1 < zeta1; z1++)
                                  for (int z2 = 0; z2 < zeta2; z2++)
                                  {
                                    x(k) +=
                                      cgcU3(k1, k2, kappa, rho) *
                                      cgcU4(z1, z2, zeta, eta) *
                                      cgcU2 *
                                      coeffsArr1[i]((alpha1 * kappa1 + k1) * zeta1 + z1) *
                                      coeffsArr2[j]((alpha2 * kappa2 + k2) * zeta2 + z2);
                                  }
                          }

                          if (x.norm() < 1e-12) continue;

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
    
    else // (Omegas_.size() >= 2)
    {
      // Extract the final coupling
      const U3xU4::OmegaElement& T = Omegas().back();
      int kappamax = u3::Kappa(T.u3(), LL / 2);
      int zetamax = u4::Zeta(T.u4(), SS, TT);

      assert(kappamax > 0 && zetamax > 0);

      // Extract the last single-shell tensor
      const U3xU4::SingleShellTensor& t2 = Gammas().back();
      int alphamax2 = t2.rho0max() * t2.eta0max();

      // Prepare for new multi-shell tensors
      U3xU4::SingleShellTensors Gammas_new = Gammas(); // create a copy
      Gammas_new.pop_back();
      U3xU4::OmegaElements Omegas_new = Omegas(); // create a copy
      Omegas_new.pop_back();      

      int alphamax1 = 1;
      for (const auto& G : Gammas_new) alphamax1 *= G.rho0max() * G.eta0max();
      for (const auto& O : Omegas_new) alphamax1 *= O.rhomax() * O.etamax();

      // Multiplicities
      qcx_utils::NDIndex6 multis;
      multis.add_dimensions(alphamax1, alphamax2, T.rhomax(), 
                            T.etamax(), kappamax, zetamax);
      size_t n_multis = multis.cardinality();

      // Construct physical bases
      u3::physicalvectors pv3s1 = u3::physicalBasis(Omegas_new.back().u3()),
                          pv3s2 = u3::physicalBasis(t2.u3());

      u4::physicalvectors pv4s1 = u4::physicalBasis(Omegas_new.back().u4()),
                          pv4s2 = u4::physicalBasis(t2.u4());

      for (const auto& pv31 : pv3s1)
      {
        int lam1 = pv31.lam(), mu1 = pv31.mu(),
            kappa1 = pv31.kappa(), LL1 = 2 * pv31.L();

        for (const auto& pv32 : pv3s2)
        {
          int lam2 = pv32.lam(), mu2 = pv32.mu(),
              kappa2 = pv32.kappa(), LL2 = 2 * pv32.L();

          const auto cgcU3 = wigU3SO3db.get_U3SO3(lam1, mu1, LL1 / 2, 
                      lam2, mu2, LL2 / 2, T.lam(), T.mu(), LL / 2);

          if (cgcU3.empty()) continue;

          for (const auto& pv41 : pv4s1)
          {
            int P11 = pv41.P1(), P21 = pv41.P2(), P31 = pv41.P3(),
                zeta1 = pv41.zeta(), SS1 = pv41.SS(), TT1 = pv41.TT();

            for (const auto& pv42 : pv4s2)
            {
              int P12 = pv42.P1(), P22 = pv42.P2(), P32 = pv42.P3(), 
                  zeta2 = pv42.zeta(), SS2 = pv42.SS(), TT2 = pv42.TT();

              const auto cgcU4 = wigU4SU2SU2db.get_U4SU2SU2(
                  P11, P21, P31, SS1, TT1, P12, P22, P32, SS2, TT2,
                  T.P1(), T.P2(), T.P3(), SS, TT);

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
                    double cgcT = cgcs.get({TT1, TT2, TT, MMT1, MMT2, MMT()});

                    U3xU4::MultiShellTensor_phys 
                      t1_phys(Gammas_new, Omegas_new, MMT1, JJ1, MMJ1, labels_);

                    std::vector<Eigen::RowVectorXd> coeffsArr1;
                    std::vector<U3xU4::ads_ats_phys> opsArr1;
                    
                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ_) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;
                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double U9j = racah9.get({LL1, LL2, LL,
                                                SS1, SS2, SS,
                                                JJ1, JJ2, JJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (coeffsArr1.empty())
                      {
                        // std::cout << "Start expanding t1: " << t1_phys << std::endl;
                        std::tie(coeffsArr1, opsArr1) = t1_phys.expand(LL1, SS1, TT1);
                        // std::cout << "Finish expanding t1" << std::endl;       
                      }

                      U3xU4::SingleShellTensor_phys 
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);

                      // std::cout << "Start expanding t2: " << t2_phys << std::endl;
                      auto [coeffsArr2, adsArr2, asArr2] = t2_phys.expand();
                      // std::cout << "Finish expanding t2" << std::endl;

                      for (size_t i = 0; i < coeffsArr1.size(); i++)
                      {
                        for (size_t j = 0; j < coeffsArr2.size(); j++)
                        {
                          Eigen::RowVectorXd x(n_multis);
                          x.setZero();
                                                    
                          for (size_t k = 0; k < n_multis; k++)
                          {
                            std::array<size_t, 6> multi;
                            multis.to_multi(k, multi);

                            auto alpha1 = multi[0];
                            auto alpha2 = multi[1];
                            auto rho = multi[2];
                            auto eta = multi[3];
                            auto kappa = multi[4];
                            auto zeta = multi[5];

                            for (int k1 = 0; k1 < kappa1; k1++)
                              for (int k2 = 0; k2 < kappa2; k2++)
                                for (int z1 = 0; z1 < zeta1; z1++)
                                  for (int z2 = 0; z2 < zeta2; z2++)
                                  {
                                    x(k) +=
                                      cgcU3(k1, k2, kappa, rho) *
                                      cgcU4(z1, z2, zeta, eta) *
                                      cgcU2 *
                                      coeffsArr1[i]((alpha1 * kappa1 + k1) * zeta1 + z1) *
                                      coeffsArr2[j]((alpha2 * kappa2 + k2) * zeta2 + z2);
                                  }

                          }

                          if (x.norm() < 1e-12) continue;

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
    
    return std::make_pair(std::move(coeffsArr), std::move(opsArr));
  }

  // Get two-body strength directly during expand()

  // Restores the original ordering using inverse permutation
  static std::array<int, 4>
  inverse_permutation(const std::array<int, 4>& sorted,
                      const std::vector<int>& inverse)
  {
    std::array<int, 4> restored;
    for (int i = 0; i < 4; ++i)
      restored[i] = sorted[inverse[i]];
    return restored;
  }

  std::vector<double> MultiShellTensor_phys::getStrengths(
    int LL, int SS, int TT,
    const HO::TwBME_scalar& TBME,
    const std::vector<std::vector<int>>& inverse,
    const std::vector<int>& phase) const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::racahs9();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    if (Omegas().size() == 0)
    {
      // Delegate to single-shell expansion
      const U3xU4::SingleShellTensor& t = Gammas()[0];
      U3xU4::SingleShellTensor_phys 
        t_phys(t, LL, SS, TT, MMT(), JJ(), MMJ(), labels_);

      return t_phys.getStrengths(TBME);
    }
    
    else if (Omegas().size() == 1)
    {
      // Final coupling
      const U3xU4::OmegaElement& T = Omegas()[0];
      int kappamax = u3::Kappa(T.u3(), LL / 2);
      int zetamax = u4::Zeta(T.u4(), SS, TT);

      assert(kappamax > 0 && zetamax > 0);

      std::vector<double> strengths(alphamax() * kappamax * zetamax, 0.0);

      // There are two single-shell tensors in this case
      const U3xU4::SingleShellTensor& t1 = Gammas()[0];
      int alphamax1 = t1.rho0max() * t1.eta0max();
      const U3xU4::SingleShellTensor& t2 = Gammas()[1];
      int alphamax2 = t2.rho0max() * t2.eta0max();

      // Multiplicities
      qcx_utils::NDIndex6 multis;
      multis.add_dimensions(alphamax1, alphamax2, T.rhomax(), 
                            T.etamax(), kappamax, zetamax);
      size_t n_multis = multis.cardinality();

      // Construct physical bases
      u3::physicalvectors pv3s1 = u3::physicalBasis(t1.u3()),
                          pv3s2 = u3::physicalBasis(t2.u3());

      u4::physicalvectors pv4s1 = u4::physicalBasis(t1.u4()),
                          pv4s2 = u4::physicalBasis(t2.u4());      

      for (const auto& pv31 : pv3s1)
      {
        int lam1 = pv31.lam(), mu1 = pv31.mu(),
            kappa1 = pv31.kappa(), LL1 = 2 * pv31.L();

        for (const auto& pv32 : pv3s2)
        {
          int lam2 = pv32.lam(), mu2 = pv32.mu(),
              kappa2 = pv32.kappa(), LL2 = 2 * pv32.L();

          const auto cgcU3 = wigU3SO3db.get_U3SO3(lam1, mu1, LL1 / 2, 
            lam2, mu2, LL2 / 2, T.lam(), T.mu(), LL / 2);

          if (cgcU3.empty()) continue;

          for (const auto& pv41 : pv4s1)
          {
            int P11 = pv41.P1(), P21 = pv41.P2(), P31 = pv41.P3(), 
                zeta1 = pv41.zeta(), SS1 = pv41.SS(), TT1 = pv41.TT();

            for (const auto& pv42 : pv4s2)
            {
              int P12 = pv42.P1(), P22 = pv42.P2(), P32 = pv42.P3(), 
                  zeta2 = pv42.zeta(), SS2 = pv42.SS(), TT2 = pv42.TT();

              const auto cgcU4 = wigU4SU2SU2db.get_U4SU2SU2(
                  P11, P21, P31, SS1, TT1, P12, P22, P32, SS2, TT2,
                  T.P1(), T.P2(), T.P3(), SS, TT);

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
                    double cgcT = cgcs.get({TT1, TT2, TT, MMT1, MMT2, MMT()});
                    
                    U3xU4::SingleShellTensor_phys 
                      t1_phys(t1, LL1, SS1, TT1, MMT1, JJ1, MMJ1, labels_);

                    std::vector<Eigen::RowVectorXd> coeffsArr1;
                    std::vector<U3xU4::ads_phys> adsArr1;
                    std::vector<U3xU4::ats_phys> asArr1;

                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ()) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;
                  
                      double U9j = racah9.get({LL1, LL2, LL,
                                              SS1, SS2, SS,
                                              JJ1, JJ2, JJ()});
                      
                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (coeffsArr1.empty())
                      {
                        // std::cout << "Start expanding t1: " << t1_phys << std::endl;
                        std::tie(coeffsArr1, adsArr1, asArr1) = t1_phys.expand();
                        // std::cout << "Finish expanding t1" << std::endl;
                      }

                      U3xU4::SingleShellTensor_phys 
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);   

                      // std::cout << "Start expanding t2: " << t2_phys << std::endl;
                      auto [coeffsArr2, adsArr2, asArr2] = t2_phys.expand();
                      // std::cout << "Finish expanding t2" << std::endl;

                      for (size_t i = 0; i < coeffsArr1.size(); i++)
                      {
                        for (size_t j = 0; j < coeffsArr2.size(); j++)
                        {
                          // Extract quantum numbers
                          std::array<int, 4> ns, lls, jjs, mmjs, mmts;
                          int o = 0;
                          if (!adsArr1.empty())
                            for (const auto& ad1 : adsArr1[i])
                            {
                              std::tie(ns[o], lls[o], jjs[o], mmjs[o], mmts[o])
                                = ad1.getLabels();
                              o++;
                            }
                          if (!asArr1.empty())
                            for (const auto& a1 : asArr1[i])
                            {
                              std::tie(ns[o], lls[o], jjs[o], mmjs[o], mmts[o])
                                = a1.getLabels();
                              o++;
                            }
                          if (!adsArr2.empty())
                            for (const auto& ad2 : adsArr2[j])
                            {
                              std::tie(ns[o], lls[o], jjs[o], mmjs[o], mmts[o])
                                = ad2.getLabels();
                              o++;
                            }
                          if (!asArr2.empty())
                            for (const auto& a2 : asArr2[j])
                            {
                              std::tie(ns[o], lls[o], jjs[o], mmjs[o], mmts[o])
                                = a2.getLabels();
                              o++;
                            }
                          assert(o == 4);

                          // Get matrix elements
                          double me = 0.0;
                          for (int k = 0; k < inverse.size(); k++)
                          {
                            std::array<int, 4> 
                              ns_orig = inverse_permutation(ns, inverse[k]),
                              lls_orig = inverse_permutation(lls, inverse[k]),
                              jjs_orig = inverse_permutation(jjs, inverse[k]),
                              mmjs_orig = inverse_permutation(mmjs, inverse[k]),
                              mmts_orig = inverse_permutation(mmts, inverse[k]);

                            // me needs computing with normal ordering
                            me += phase[k] * TBME.getME(
                              ns_orig[0], lls_orig[0], jjs_orig[0], mmjs_orig[0], mmts_orig[0],
                              ns_orig[1], lls_orig[1], jjs_orig[1], mmjs_orig[1], mmts_orig[1],
                              ns_orig[3], lls_orig[3], jjs_orig[3], mmjs_orig[3], mmts_orig[3],
                              ns_orig[2], lls_orig[2], jjs_orig[2], mmjs_orig[2], mmts_orig[2]) / 4.0;
                          }

                          if (std::fabs(me) < 1e-10) continue;

                          for (size_t k = 0; k < n_multis; k++)
                          {
                            std::array<size_t, 6> multi;
                            multis.to_multi(k, multi);

                            auto alpha1 = multi[0];
                            auto alpha2 = multi[1];
                            auto rho = multi[2];
                            auto eta = multi[3];
                            auto kappa = multi[4];
                            auto zeta = multi[5];

                            for (int k1 = 0; k1 < kappa1; k1++)
                              for (int k2 = 0; k2 < kappa2; k2++)
                                for (int z1 = 0; z1 < zeta1; z1++)
                                  for (int z2 = 0; z2 < zeta2; z2++)
                                  {
                                    strengths[k] += me *
                                      cgcU3(k1, k2, kappa, rho) *
                                      cgcU4(z1, z2, zeta, eta) *
                                      cgcU2 *
                                      coeffsArr1[i]((alpha1 * kappa1 + k1) * zeta1 + z1) *
                                      coeffsArr2[j]((alpha2 * kappa2 + k2) * zeta2 + z2);
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
      }

      return strengths;
    }
    
    else // (Omegas_.size() >= 2)
    {
      // Extract the final coupling
      const U3xU4::OmegaElement& T = Omegas().back();
      int kappamax = u3::Kappa(T.u3(), LL / 2);
      int zetamax = u4::Zeta(T.u4(), SS, TT);

      assert(kappamax > 0 && zetamax > 0);

      std::vector<double> strengths(alphamax() * kappamax * zetamax, 0.0);

      // Extract the last single-shell tensor
      const U3xU4::SingleShellTensor& t2 = Gammas().back();
      int alphamax2 = t2.rho0max() * t2.eta0max();

      // Prepare for new multi-shell tensors
      U3xU4::SingleShellTensors Gammas_new = Gammas(); // create a copy
      Gammas_new.pop_back();
      U3xU4::OmegaElements Omegas_new = Omegas(); // create a copy
      Omegas_new.pop_back();      

      int alphamax1 = 1;
      for (const auto& G : Gammas_new) alphamax1 *= G.rho0max() * G.eta0max();
      for (const auto& O : Omegas_new) alphamax1 *= O.rhomax() * O.etamax();

      // Multiplicities
      qcx_utils::NDIndex6 multis;
      multis.add_dimensions(alphamax1, alphamax2, T.rhomax(), 
                            T.etamax(), kappamax, zetamax);
      size_t n_multis = multis.cardinality();

      // Construct physical bases
      u3::physicalvectors pv3s1 = u3::physicalBasis(Omegas_new.back().u3()),
                          pv3s2 = u3::physicalBasis(t2.u3());

      u4::physicalvectors pv4s1 = u4::physicalBasis(Omegas_new.back().u4()),
                          pv4s2 = u4::physicalBasis(t2.u4());

      for (const auto& pv31 : pv3s1)
      {
        int lam1 = pv31.lam(), mu1 = pv31.mu(),
            kappa1 = pv31.kappa(), LL1 = 2 * pv31.L();

        for (const auto& pv32 : pv3s2)
        {
          int lam2 = pv32.lam(), mu2 = pv32.mu(),
              kappa2 = pv32.kappa(), LL2 = 2 * pv32.L();

          const auto cgcU3 = wigU3SO3db.get_U3SO3(lam1, mu1, LL1 / 2, 
                      lam2, mu2, LL2 / 2, T.lam(), T.mu(), LL / 2);

          if (cgcU3.empty()) continue;

          for (const auto& pv41 : pv4s1)
          {
            int P11 = pv41.P1(), P21 = pv41.P2(), P31 = pv41.P3(),
                zeta1 = pv41.zeta(), SS1 = pv41.SS(), TT1 = pv41.TT();

            for (const auto& pv42 : pv4s2)
            {
              int P12 = pv42.P1(), P22 = pv42.P2(), P32 = pv42.P3(), 
                  zeta2 = pv42.zeta(), SS2 = pv42.SS(), TT2 = pv42.TT();

              const auto cgcU4 = wigU4SU2SU2db.get_U4SU2SU2(
                  P11, P21, P31, SS1, TT1, P12, P22, P32, SS2, TT2,
                  T.P1(), T.P2(), T.P3(), SS, TT);

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
                    double cgcT = cgcs.get({TT1, TT2, TT, MMT1, MMT2, MMT()});

                    U3xU4::MultiShellTensor_phys 
                      t1_phys(Gammas_new, Omegas_new, MMT1, JJ1, MMJ1, labels_);

                    std::vector<Eigen::RowVectorXd> coeffsArr1;
                    std::vector<U3xU4::ads_ats_phys> opsArr1;
                    
                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ_) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;
                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double U9j = racah9.get({LL1, LL2, LL,
                                                SS1, SS2, SS,
                                                JJ1, JJ2, JJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (coeffsArr1.empty())
                      {
                        // std::cout << "Start expanding t1: " << t1_phys << std::endl;
                        std::tie(coeffsArr1, opsArr1) = t1_phys.expand(LL1, SS1, TT1);
                        // std::cout << "Finish expanding t1" << std::endl;       
                      }

                      U3xU4::SingleShellTensor_phys 
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);

                      // std::cout << "Start expanding t2: " << t2_phys << std::endl;
                      auto [coeffsArr2, adsArr2, asArr2] = t2_phys.expand();
                      // std::cout << "Finish expanding t2" << std::endl;

                      for (size_t i = 0; i < coeffsArr1.size(); i++)
                      {
                        for (size_t j = 0; j < coeffsArr2.size(); j++)
                        {
                          // Extract quantum numbers
                          std::array<int, 4> ns, lls, jjs, mmjs, mmts;
                          int o = 0;
                          for (const auto& op1 : opsArr1[i])
                          {
                            std::visit([&](auto&& op)
                            {
                              std::tie(ns[o], lls[o], jjs[o], mmjs[o], mmts[o]) = op.getLabels();
                            }, op1);
                            o++;
                          }
                          if (!adsArr2.empty())
                            for (const auto& ad2 : adsArr2[j])
                            {
                              std::tie(ns[o], lls[o], jjs[o], mmjs[o], mmts[o])
                                = ad2.getLabels();
                              o++;
                            }
                          if (!asArr2.empty())
                            for (const auto& a2 : asArr2[j])
                            {
                              std::tie(ns[o], lls[o], jjs[o], mmjs[o], mmts[o])
                                = a2.getLabels();
                              o++;
                            }
                          assert(o == 4);

                          // Get matrix elements
                          double me = 0.0;
                          for (int k = 0; k < inverse.size(); k++)
                          {
                            std::array<int, 4> 
                              ns_orig = inverse_permutation(ns, inverse[k]),
                              lls_orig = inverse_permutation(lls, inverse[k]),
                              jjs_orig = inverse_permutation(jjs, inverse[k]),
                              mmjs_orig = inverse_permutation(mmjs, inverse[k]),
                              mmts_orig = inverse_permutation(mmts, inverse[k]);

                            // me needs computing with normal ordering
                            me += phase[k] * TBME.getME(
                              ns_orig[0], lls_orig[0], jjs_orig[0], mmjs_orig[0], mmts_orig[0],
                              ns_orig[1], lls_orig[1], jjs_orig[1], mmjs_orig[1], mmts_orig[1],
                              ns_orig[3], lls_orig[3], jjs_orig[3], mmjs_orig[3], mmts_orig[3],
                              ns_orig[2], lls_orig[2], jjs_orig[2], mmjs_orig[2], mmts_orig[2]) / 4.0;
                          }

                          if (std::fabs(me) < 1e-10) continue;

                          for (size_t k = 0; k < n_multis; k++)
                          {
                            std::array<size_t, 6> multi;
                            multis.to_multi(k, multi);

                            auto alpha1 = multi[0];
                            auto alpha2 = multi[1];
                            auto rho = multi[2];
                            auto eta = multi[3];
                            auto kappa = multi[4];
                            auto zeta = multi[5];

                            for (int k1 = 0; k1 < kappa1; k1++)
                              for (int k2 = 0; k2 < kappa2; k2++)
                                for (int z1 = 0; z1 < zeta1; z1++)
                                  for (int z2 = 0; z2 < zeta2; z2++)
                                  {
                                    strengths[k] += me *
                                      cgcU3(k1, k2, kappa, rho) *
                                      cgcU4(z1, z2, zeta, eta) *
                                      cgcU2 *
                                      coeffsArr1[i]((alpha1 * kappa1 + k1) * zeta1 + z1) *
                                      coeffsArr2[j]((alpha2 * kappa2 + k2) * zeta2 + z2);
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
      }

      return strengths;
    }
  }

} // namespace U3xU4
