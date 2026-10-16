#include <cassert>

#include <ND.h>
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Fano.h>

#include <U3SO3xU3SO3.h>
#include <U4SU2SU2xU4SU2SU2.h>

#include <U3xU4MultiShellTensors.h>
#include <U3xU4SingleShellTensors.h>

#ifdef TIME_GETSTRENGTHS
#include <cstdio>
#include <omp.h>
namespace
{
  struct GSStat { double me = 0, acc = 0; unsigned long long getme = 0, terms = 0; };
  GSStat g_gs[256];   // per-thread slots, indexed by omp_get_thread_num()
}
#define GS_ME_BEGIN  double _gs_t0 = omp_get_wtime()
#define GS_ME_END(N) do { _gs.me += omp_get_wtime() - _gs_t0; \
                          _gs.getme += (N); _gs.terms++; } while (0)
#define GS_ACC_BEGIN double _gs_t2 = omp_get_wtime()
#define GS_ACC_END   _gs.acc += omp_get_wtime() - _gs_t2
#else
#define GS_ME_BEGIN
#define GS_ME_END(N)
#define GS_ACC_BEGIN
#define GS_ACC_END
#endif

namespace U3xU4
{
  void report_getStrengths_timing()
  {
#ifdef TIME_GETSTRENGTHS
    double me = 0, acc = 0; unsigned long long gm = 0, tm = 0;
    for (auto& s : g_gs) { me += s.me; acc += s.acc; gm += s.getme; tm += s.terms; }
    std::printf("[getStrengths] me-phase=%.2fs  accum-phase=%.2fs  "
                "getME_calls=%llu  terms=%llu  (%.2f getME/term)\n",
                me, acc, gm, tm, tm ? double(gm) / double(tm) : 0.0);
    for (auto& s : g_gs) s = GSStat{};
#endif
  }

  //////////////////// TWO-BODY MULTI-SHELL TENSORS ////////////////////

  // Expand into uncoupled products of creation operators (ad)
  // and annihilation operators (a)
  // coeffs have rhomax x etamax x kappamax x zetamax columns, index of each
  // column is
  // ((rho * etamax + eta) * kappamax + kappa) * zetamax + zeta
  Expansion TwoBodyMultiShellTensor::expand(SSTDecomp_phys_db& db) const
  {
    auto& cgcs = utils::cgcs();
    auto& racah9 = utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    Expansion result;
    result.labels = &labels_;
    std::vector<double> scratch;   // reused per-row accumulator (one allocation)

    if (Omegas_.size() == 0)
    {
      // Delegate to single-shell expansion
      const U3xU4::SingleShellTensor& tadat = Gammas_[0];
      U3xU4::SingleShellTensor_phys tadat_phys(tadat.shell(), tadat.num_ads(),
        tadat.num_ats(), tadat.Omega_ads(), tadat.Omega_ats(), tadat.T(),
        rhomax(), etamax(), LL(), SS(), TT(), MMT(), JJ(), MMJ(), labels_);

      auto e = db.get_view(tadat_phys);
      result.coeffs.ncols = e->ncols();
      result.nops = e->nops;

      for (size_t i = 0; i < e->rows(); i++)
      {
        result.coeffs.push_row(e->coeff_row(i));   // row already contiguous
        result.append_ops_from(*e, i);
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
                      auto view1 = db.get_view(tadat_phys1);

                      U3xU4::SingleShellTensor_phys tadat_phys2(
                          tadat2.shell(), tadat2.num_ads(), tadat2.num_ats(),
                          tadat2.Omega_ads(), tadat2.Omega_ats(), tadat2.T(),
                          tadat2.rho0max(), tadat2.eta0max(), LL2, SS2, TT2,
                          MMT2, JJ2, MMJ2, labels_);
                      auto view2 = db.get_view(tadat_phys2);

                      const std::size_t ncols =
                          rhomax() * etamax() * kappamax() * zetamax();
                      result.coeffs.ncols = ncols;
                      result.nops = view1->nops + view2->nops;

                      for (size_t i = 0; i < view1->rows(); i++)
                      {
                        const double* c1 = view1->coeff_row(i);
                        for (size_t j = 0; j < view2->rows(); j++)
                        {
                          const double* c2 = view2->coeff_row(j);
                          scratch.assign(ncols, 0.0);
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
                                          scratch[index] +=
                                              cgcU3(k1, k2, k, rho) *
                                              cgcU4(z1, z2, z, eta) *
                                              U9j * cgcJ * cgcT *
                                              c1[k1 * zeta1 + z1] *
                                              c2[k2 * zeta2 + z2];
                                        }
                                }

                          double s2 = 0.0;
                          for (double v : scratch) s2 += v * v;
                          if (s2 < 1e-24) continue;

                          result.coeffs.push_row(scratch.data());
                          result.append_ops_from(*view1, i);
                          result.append_ops_from(*view2, j);
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

                      Expansion e1 = new_multi_shell_tensor.expand(db);

                      U3xU4::SingleShellTensor_phys tadat_phys(
                          tadat.shell(), tadat.num_ads(), tadat.num_ats(),
                          tadat.Omega_ads(), tadat.Omega_ats(), tadat.T(),
                          tadat.rho0max(), tadat.eta0max(), LL2, SS2, TT2, MMT2,
                          JJ2, MMJ2, labels_);
                      auto view2 = db.get_view(tadat_phys);

                      const std::size_t ncols =
                          rhomax() * etamax() * kappamax() * zetamax();
                      result.coeffs.ncols = ncols;
                      result.nops = e1.nops + view2->nops;

                      for (size_t i = 0; i < e1.rows(); i++)
                      {
                        const double* c1 = e1.coeff_row(i);
                        for (size_t j = 0; j < view2->rows(); j++)
                        {
                          const double* c2 = view2->coeff_row(j);
                          scratch.assign(ncols, 0.0);
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
                                          scratch[index] +=
                                              cgcU3(k1, k2, k, rho) *
                                              cgcU4(z1, z2, z, eta) *
                                              U9j * cgcJ * cgcT *
                                              c1[k1 * zeta1 + z1] *
                                              c2[k2 * zeta2 + z2];
                                }

                          double s2 = 0.0;
                          for (double v : scratch) s2 += v * v;
                          if (s2 < 1e-24) continue;

                          result.coeffs.push_row(scratch.data());
                          result.append_ops_from(e1, i);
                          result.append_ops_from(*view2, j);
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

  //////////////////// GENERIC MULTI-SHELL TENSORS ////////////////////

  // Expand into uncoupled products of creation operators (ad)
  // and annihilation operators (a)
  Expansion MultiShellTensor_phys::expand(int LL, int SS, int TT,
                                          SSTDecomp_phys_db& db) const
  {
    auto& cgcs = utils::cgcs();
    auto& racah9 = utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    Expansion result;
    result.labels = &labels_;
    std::vector<double> scratch;   // reused per-row accumulator (one allocation)
    int estimate = 1;
    for (const auto& G : Gammas())
    {
      int n = G.shell(), shell_dim = 2 * (n + 1) * (n + 2);
      for (int i = 0; i < G.num_ads(); i++) estimate *= shell_dim;
      for (int i = 0; i < G.num_ats(); i++) estimate *= shell_dim;
    }

    if (Omegas().size() == 0)
    {
      // Delegate to single-shell expansion
      const U3xU4::SingleShellTensor& t = Gammas()[0];
      U3xU4::SingleShellTensor_phys t_phys(t, LL, SS, TT, MMT(), JJ(), MMJ(), labels_);

      auto e = db.get_view(t_phys);
      result.coeffs.ncols = e->ncols();
      result.nops = e->nops;
      result.coeffs.reserve_rows(estimate);

      for (size_t i = 0; i < e->rows(); i++)
      {
        result.coeffs.push_row(e->coeff_row(i));   // row already contiguous
        result.append_ops_from(*e, i);
      }
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
      utils::NDIndex6 multis;
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

                    SSTDecomp_phys_db::ExpansionRef view1;
                    bool got1 = false;

                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ()) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;

                      double U9j = racah9.get({LL1, LL2, LL,
                                              SS1, SS2, SS,
                                              JJ1, JJ2, JJ()});

                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (!got1)
                      {
                        view1 = db.get_view(t1_phys);
                        got1 = true;
                      }

                      U3xU4::SingleShellTensor_phys
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);

                      auto view2 = db.get_view(t2_phys);

                      result.coeffs.ncols = n_multis;
                      result.nops = view1->nops + view2->nops;

                      for (size_t i = 0; i < view1->rows(); i++)
                      {
                        const double* c1 = view1->coeff_row(i);
                        for (size_t j = 0; j < view2->rows(); j++)
                        {
                          const double* c2 = view2->coeff_row(j);
                          scratch.assign(n_multis, 0.0);

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
                                    scratch[k] +=
                                      cgcU3(k1, k2, kappa, rho) *
                                      cgcU4(z1, z2, zeta, eta) *
                                      cgcU2 *
                                      c1[(alpha1 * kappa1 + k1) * zeta1 + z1] *
                                      c2[(alpha2 * kappa2 + k2) * zeta2 + z2];
                                  }
                          }

                          double s2 = 0.0;
                          for (double v : scratch) s2 += v * v;
                          if (s2 < 1e-24) continue;

                          result.coeffs.push_row(scratch.data());
                          result.append_ops_from(*view1, i);
                          result.append_ops_from(*view2, j);
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
      utils::NDIndex6 multis;
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

                    Expansion e1;
                    
                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ_) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;
                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double U9j = racah9.get({LL1, LL2, LL,
                                                SS1, SS2, SS,
                                                JJ1, JJ2, JJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (e1.empty())
                        e1 = t1_phys.expand(LL1, SS1, TT1, db);

                      U3xU4::SingleShellTensor_phys 
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);

                      // std::cout << "Start expanding t2: " << t2_phys << std::endl;
                      auto view2 = db.get_view(t2_phys);

                      result.coeffs.ncols = n_multis;
                      result.nops = e1.nops + view2->nops;

                      for (size_t i = 0; i < e1.rows(); i++)
                      {
                        const double* c1 = e1.coeff_row(i);
                        for (size_t j = 0; j < view2->rows(); j++)
                        {
                          const double* c2 = view2->coeff_row(j);
                          scratch.assign(n_multis, 0.0);

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
                                    scratch[k] +=
                                      cgcU3(k1, k2, kappa, rho) *
                                      cgcU4(z1, z2, zeta, eta) *
                                      cgcU2 *
                                      c1[(alpha1 * kappa1 + k1) * zeta1 + z1] *
                                      c2[(alpha2 * kappa2 + k2) * zeta2 + z2];
                                  }

                          }

                          double s2 = 0.0;
                          for (double v : scratch) s2 += v * v;
                          if (s2 < 1e-24) continue;

                          result.coeffs.push_row(scratch.data());
                          result.append_ops_from(e1, i);
                          result.append_ops_from(*view2, j);
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
    const std::vector<int>& phase,
    SSTDecomp_phys_db& db) const
  {
    auto& cgcs = utils::cgcs();
    auto& racah9 = utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    // Resolve this thread's m-scheme cache ONCE here, not on every getME call
    // (billions of them) -- a thread_local access is emulated-TLS on macOS.
    auto& mecache = HO::twbme_mscheme_cache();

    // Small per-tensor cache in front of mecache: absorbs the ~9x intra-tensor
    // quad reuse in L2/L3 instead of probing the 2.9GB global table (DRAM) each
    // time. thread_local so its buffer is reused across tensors; cleared per call
    // so it only holds THIS tensor's quads.
    thread_local HO::TwBME_mscheme_cache_t local_cache;
    local_cache.clear();

#ifdef MEASURE_MSCHEME_REUSE
    HO::MSchemeReuseScope _reuse_scope;  // brackets this tensor's probes
#endif

#ifdef TIME_GETSTRENGTHS
    auto& _gs = g_gs[omp_get_thread_num()];
#endif

    if (Omegas().size() == 0)
    {
      // Delegate to single-shell expansion
      const U3xU4::SingleShellTensor& t = Gammas()[0];
      U3xU4::SingleShellTensor_phys
        t_phys(t, LL, SS, TT, MMT(), JJ(), MMJ(), labels_);

      return t_phys.getStrengths(TBME, db);
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
      utils::NDIndex6 multis;
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

                    SSTDecomp_phys_db::ExpansionRef view1;
                    bool got1 = false;

                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ()) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;

                      double U9j = racah9.get({LL1, LL2, LL,
                                              SS1, SS2, SS,
                                              JJ1, JJ2, JJ()});

                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (!got1)
                      {
                        view1 = db.get_view(t1_phys);
                        got1 = true;
                      }

                      U3xU4::SingleShellTensor_phys 
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);   

                      auto view2 = db.get_view(t2_phys);

                      for (size_t i = 0; i < view1->rows(); i++)
                      {
                        const double* c1 = view1->coeff_row(i);
                        for (size_t j = 0; j < view2->rows(); j++)
                        {
                          const double* c2 = view2->coeff_row(j);

                          // Extract quantum numbers straight from the table
                          std::array<int, 4> ns, lls, jjs, mmjs, mmts;
                          int o = 0;
                          o = view1->fill_labels(i, o, ns, lls, jjs, mmjs, mmts);
                          o = view2->fill_labels(j, o, ns, lls, jjs, mmjs, mmts);
                          assert(o == 4);

                          // Get matrix elements
                          GS_ME_BEGIN;
                          double me = 0.0;
                          for (int k = 0; k < inverse.size(); k++)
                          {
                            // inverse_permutation is just restored[i]=sorted[p[i]];
                            // index the label arrays directly (no array rebuilds).
                            const auto& p = inverse[k];
                            const int i0 = p[0], i1 = p[1], i2 = p[2], i3 = p[3];

                            // me needs computing with normal ordering (slots 0,1,3,2)
                            me += phase[k] * TBME.getME(local_cache, mecache,
                              ns[i0], lls[i0], jjs[i0], mmjs[i0], mmts[i0],
                              ns[i1], lls[i1], jjs[i1], mmjs[i1], mmts[i1],
                              ns[i3], lls[i3], jjs[i3], mmjs[i3], mmts[i3],
                              ns[i2], lls[i2], jjs[i2], mmjs[i2], mmts[i2]) / 4.0;
                          }
                          GS_ME_END(inverse.size());

                          if (std::fabs(me) < 1e-10) continue;

                          GS_ACC_BEGIN;
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
                                      c1[(alpha1 * kappa1 + k1) * zeta1 + z1] *
                                      c2[(alpha2 * kappa2 + k2) * zeta2 + z2];
                                  }
                          }
                          GS_ACC_END;
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
      utils::NDIndex6 multis;
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

                    Expansion e1;
                    
                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ_) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;
                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double U9j = racah9.get({LL1, LL2, LL,
                                                SS1, SS2, SS,
                                                JJ1, JJ2, JJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (e1.empty())
                        e1 = t1_phys.expand(LL1, SS1, TT1, db);

                      U3xU4::SingleShellTensor_phys 
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);

                      auto view2 = db.get_view(t2_phys);

                      for (size_t i = 0; i < e1.rows(); i++)
                      {
                        const double* c1 = e1.coeff_row(i);
                        for (size_t j = 0; j < view2->rows(); j++)
                        {
                          const double* c2 = view2->coeff_row(j);

                          // Extract quantum numbers straight from the table
                          std::array<int, 4> ns, lls, jjs, mmjs, mmts;
                          int o = 0;
                          o = e1.fill_labels(i, o, ns, lls, jjs, mmjs, mmts);
                          o = view2->fill_labels(j, o, ns, lls, jjs, mmjs, mmts);
                          assert(o == 4);

                          // Get matrix elements
                          GS_ME_BEGIN;
                          double me = 0.0;
                          for (int k = 0; k < inverse.size(); k++)
                          {
                            // inverse_permutation is just restored[i]=sorted[p[i]];
                            // index the label arrays directly (no array rebuilds).
                            const auto& p = inverse[k];
                            const int i0 = p[0], i1 = p[1], i2 = p[2], i3 = p[3];

                            // me needs computing with normal ordering (slots 0,1,3,2)
                            me += phase[k] * TBME.getME(local_cache, mecache,
                              ns[i0], lls[i0], jjs[i0], mmjs[i0], mmts[i0],
                              ns[i1], lls[i1], jjs[i1], mmjs[i1], mmts[i1],
                              ns[i3], lls[i3], jjs[i3], mmjs[i3], mmts[i3],
                              ns[i2], lls[i2], jjs[i2], mmjs[i2], mmts[i2]) / 4.0;
                          }
                          GS_ME_END(inverse.size());

                          if (std::fabs(me) < 1e-10) continue;

                          GS_ACC_BEGIN;
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
                                      c1[(alpha1 * kappa1 + k1) * zeta1 + z1] *
                                      c2[(alpha2 * kappa2 + k2) * zeta2 + z2];
                                  }
                          }
                          GS_ACC_END;
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

  std::vector<double> MultiShellTensor_phys::getStrengths(
    int LL, int SS, int TT,
    const HO::TwBME_general& TBME,
    const std::vector<std::vector<int>>& inverse,
    const std::vector<int>& phase,
    SSTDecomp_phys_db& db) const
  {
    auto& cgcs = utils::cgcs();
    auto& racah9 = utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    // Resolve this thread's m-scheme cache ONCE here, not on every getME call
    // (billions of them) -- a thread_local access is emulated-TLS on macOS.
    auto& mecache = HO::twbme_mscheme_cache();

    // Small per-tensor cache in front of mecache: absorbs the ~9x intra-tensor
    // quad reuse in L2/L3 instead of probing the 2.9GB global table (DRAM) each
    // time. thread_local so its buffer is reused across tensors; cleared per call
    // so it only holds THIS tensor's quads.
    thread_local HO::TwBME_mscheme_cache_t local_cache;
    local_cache.clear();

#ifdef MEASURE_MSCHEME_REUSE
    HO::MSchemeReuseScope _reuse_scope;  // brackets this tensor's probes
#endif

#ifdef TIME_GETSTRENGTHS
    auto& _gs = g_gs[omp_get_thread_num()];
#endif

    if (Omegas().size() == 0)
    {
      // Delegate to single-shell expansion
      const U3xU4::SingleShellTensor& t = Gammas()[0];
      U3xU4::SingleShellTensor_phys
        t_phys(t, LL, SS, TT, MMT(), JJ(), MMJ(), labels_);

      return t_phys.getStrengths(TBME, db);
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
      utils::NDIndex6 multis;
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

                    SSTDecomp_phys_db::ExpansionRef view1;
                    bool got1 = false;

                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ()) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;

                      double U9j = racah9.get({LL1, LL2, LL,
                                              SS1, SS2, SS,
                                              JJ1, JJ2, JJ()});

                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (!got1)
                      {
                        view1 = db.get_view(t1_phys);
                        got1 = true;
                      }

                      U3xU4::SingleShellTensor_phys 
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);   

                      auto view2 = db.get_view(t2_phys);

                      for (size_t i = 0; i < view1->rows(); i++)
                      {
                        const double* c1 = view1->coeff_row(i);
                        for (size_t j = 0; j < view2->rows(); j++)
                        {
                          const double* c2 = view2->coeff_row(j);

                          // Extract quantum numbers straight from the table
                          std::array<int, 4> ns, lls, jjs, mmjs, mmts;
                          int o = 0;
                          o = view1->fill_labels(i, o, ns, lls, jjs, mmjs, mmts);
                          o = view2->fill_labels(j, o, ns, lls, jjs, mmjs, mmts);
                          assert(o == 4);

                          // Get matrix elements
                          GS_ME_BEGIN;
                          double me = 0.0;
                          for (int k = 0; k < inverse.size(); k++)
                          {
                            // inverse_permutation is just restored[i]=sorted[p[i]];
                            // index the label arrays directly (no array rebuilds).
                            const auto& p = inverse[k];
                            const int i0 = p[0], i1 = p[1], i2 = p[2], i3 = p[3];

                            // me needs computing with normal ordering (slots 0,1,3,2)
                            me += phase[k] * TBME.getME(local_cache, mecache,
                              ns[i0], lls[i0], jjs[i0], mmjs[i0], mmts[i0],
                              ns[i1], lls[i1], jjs[i1], mmjs[i1], mmts[i1],
                              ns[i3], lls[i3], jjs[i3], mmjs[i3], mmts[i3],
                              ns[i2], lls[i2], jjs[i2], mmjs[i2], mmts[i2]) / 4.0;
                          }
                          GS_ME_END(inverse.size());

                          if (std::fabs(me) < 1e-10) continue;

                          GS_ACC_BEGIN;
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
                                      c1[(alpha1 * kappa1 + k1) * zeta1 + z1] *
                                      c2[(alpha2 * kappa2 + k2) * zeta2 + z2];
                                  }
                          }
                          GS_ACC_END;
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
      utils::NDIndex6 multis;
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

                    Expansion e1;
                    
                    for (int JJ2 = LL2 + SS2; JJ2 >= abs(LL2 - SS2); JJ2 -= 2)
                    {
                      if (u2::mult(JJ1, JJ2, JJ_) == 0) continue;
                      if (abs(MMJ2) > JJ2) continue;
                      double cgcJ = cgcs.get({JJ1, JJ2, JJ(), MMJ1, MMJ2, MMJ()});

                      double U9j = racah9.get({LL1, LL2, LL,
                                                SS1, SS2, SS,
                                                JJ1, JJ2, JJ()});

                      double cgcU2 = U9j * cgcJ * cgcT;

                      if (e1.empty())
                        e1 = t1_phys.expand(LL1, SS1, TT1, db);

                      U3xU4::SingleShellTensor_phys 
                        t2_phys(t2, LL2, SS2, TT2, MMT2, JJ2, MMJ2, labels_);

                      auto view2 = db.get_view(t2_phys);

                      for (size_t i = 0; i < e1.rows(); i++)
                      {
                        const double* c1 = e1.coeff_row(i);
                        for (size_t j = 0; j < view2->rows(); j++)
                        {
                          const double* c2 = view2->coeff_row(j);

                          // Extract quantum numbers straight from the table
                          std::array<int, 4> ns, lls, jjs, mmjs, mmts;
                          int o = 0;
                          o = e1.fill_labels(i, o, ns, lls, jjs, mmjs, mmts);
                          o = view2->fill_labels(j, o, ns, lls, jjs, mmjs, mmts);
                          assert(o == 4);

                          // Get matrix elements
                          GS_ME_BEGIN;
                          double me = 0.0;
                          for (int k = 0; k < inverse.size(); k++)
                          {
                            // inverse_permutation is just restored[i]=sorted[p[i]];
                            // index the label arrays directly (no array rebuilds).
                            const auto& p = inverse[k];
                            const int i0 = p[0], i1 = p[1], i2 = p[2], i3 = p[3];

                            // me needs computing with normal ordering (slots 0,1,3,2)
                            me += phase[k] * TBME.getME(local_cache, mecache,
                              ns[i0], lls[i0], jjs[i0], mmjs[i0], mmts[i0],
                              ns[i1], lls[i1], jjs[i1], mmjs[i1], mmts[i1],
                              ns[i3], lls[i3], jjs[i3], mmjs[i3], mmts[i3],
                              ns[i2], lls[i2], jjs[i2], mmjs[i2], mmts[i2]) / 4.0;
                          }
                          GS_ME_END(inverse.size());

                          if (std::fabs(me) < 1e-10) continue;

                          GS_ACC_BEGIN;
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
                                      c1[(alpha1 * kappa1 + k1) * zeta1 + z1] *
                                      c2[(alpha2 * kappa2 + k2) * zeta2 + z2];
                                  }
                          }
                          GS_ACC_END;
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
