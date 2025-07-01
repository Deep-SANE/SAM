#include <chrono>
#include <fstream>
#include <iostream>

#include <U3basis.h>

#include <U4basis.h>

#include <U3xU4SingleShellTensors.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  u34utils::SO3_CGC cgcs;
  u3::WigU3U2_db wigU3U2db(cgcs);
  u4::WigU4U3_db wigU4U3db(cgcs, wigU3U2db);

  U3xU4::ListIrreps prod_ads, prod_ats;

  /* Example 1:
    < alpha_i (4,3,1) (3,1,0,0) |||
                 (3,1,0)       (3,3,0)            (3,2,0)   (4,1,0)
      { [[ad x ad]         x ad]        x [at x at]        }
                 (1,1,0,0)     (2,1,0,0)          (1,1,0,0) (1,0,0,0)
                                        ||| alpha_f (5,3,2) (3,1,1,0) >
  */
  /*
  std::size_t n = 2, ads = 3, ats = 2;
  u3::irrep i3(4,3,1), t3_1_0(3,1,0), t3_1(3,3,0), t3_2(3,2,0), t3(4,1,0),
  f3(5,3,2); u4::irrep i4(3,1,0,0), t4_1_0(1,1,0,0), t4_1(2,1,0,0),
  t4_2(1,1,0,0), t4(1,0,0,0), f4(3,1,1,0);

  prod_ads.emplace_back(t3_1_0,t4_1_0);
  prod_ads.emplace_back(t3_1,t4_1);
  prod_ats.emplace_back(t3_2,t4_2);

  std::size_t rho0max = u3::Rho(t3_1,t3_2,t3), eta0max = u4::Eta(t4_1,t4_2,t4);
  */

  /* Example 2:
    < alpha_i (14,2,0) (1,1,1,1) |||
                (6,2,0)             (6,4,0)   (4,2,0)
      { [ad x ad]         x [at x at]         }
                (1,1,0,0)           (1,1,0,0) (0,0,0,0)
                ||| alpha_f (14,2,0) (1,1,1,1) >
  */
  /*
  std::size_t n = 4, ads = 2, ats = 2;
  u3::irrep i3(14,2,0), t3_1(6,2,0), t3_2(6,4,0), t3(4,2,0), f3(14,2,0);
  u4::irrep i4(1,1,1,1), t4_1(1,1,0,0), t4_2(1,1,0,0), t4(0,0,0,0), f4(1,1,1,1);

  prod_ads.emplace_back(t3_1,t4_1);
  prod_ats.emplace_back(t3_2,t4_2);

  std::size_t rho0max = u3::Rho(t3_1,t3_2,t3), eta0max = u4::Eta(t4_1,t4_2,t4);
  */

  /* Example 3:
                              (n,0,0)
    < (n,0,0) (1,0,0,0) ||| ad         ||| (0,0,0) (0,0,0,0) >
                              (1,0,0,0)

  */
  /*
  std::size_t ads = 1, ats = 0, n = 4;
  u3::irrep i3(0,0,0), t3(n,0,0), f3(n,0,0);
  u4::irrep i4(0,0,0,0), t4(1,0,0,0), f4(1,0,0,0);

  std::size_t rho0max = 1, eta0max = 1;
  */

  /* Example 4:
                              (n,n,0)
    < (0,0,0) (0,0,0,0) ||| at         ||| (n,0,0) (1,0,0,0) >
                              (1,1,1,0)

  */
  /*
  std::size_t ads = 0, ats = 1, n = 4;
  u3::irrep i3(n,0,0), t3(n,n,0), f3(0,0,0);
  u4::irrep i4(1,0,0,0), t4(1,1,1,0), f4(0,0,0,0);

  std::size_t rho0max = 1, eta0max = 1;
  */

  /* Example 5: scalar
                                    (0,0,0)
    / (n13 n23 n33)     ||| [ad x at]         ||| (n13 n23 n33)     \
    \ (n14 n24 n34 n44) |||         (0,0,0,0) ||| (n14 n24 n34 n44) /
  */

  std::size_t n = 8, ads = 1, ats = 1;
  u3::irrep i3(n, 0, 0), t3(0, 0, 0), f3 = i3;
  u4::irrep i4(1, 0, 0, 0), t4(0, 0, 0, 0), f4 = i4;

  std::size_t rho0max = 1, eta0max = 1;

  ////////////////////////////////////

  U3xU4::Tadat Tadxat(n, ads, ats, prod_ads, prod_ats, t3, rho0max, t4, eta0max);

  std::size_t alpha_i, alpha_f, rhomax, etamax;

  std::vector<Eigen::MatrixXd> RMEs = Tadxat.RME(
      alpha_i, i3, i4, alpha_f, f3, f4, rhomax, etamax, cgcs, wigU3U2db, wigU4U3db);

  std::cout << "< " << alpha_f << f3 << f4 << " ||| ";
  for (int i = 0; i < ads; i++)
    std::cout << "ad ";
  for (int i = 0; i < ats; i++)
    std::cout << "at ";
  std::cout << rho0max << t3 << eta0max << t4 << " ||| " << alpha_i << i3 << i4
            << " >" << rhomax << "," << etamax << std::endl;

  for (std::size_t ai = 0; ai < alpha_i; ai++)
    for (std::size_t af = 0; af < alpha_f; af++)
    {
      std::cout << "ai = " << ai + 1 << ", af = " << af + 1 << std::endl;

      std::cout << "rho\teta\trho0\teta0" << std::endl;

      for (int rho = 0; rho < rhomax; rho++)
        for (int eta = 0; eta < etamax; eta++)
          for (int rho0 = 0; rho0 < rho0max; rho0++)
            for (int eta0 = 0; eta0 < eta0max; eta0++)
              std::cout << rho + 1 << "\t" << eta + 1 << "\t" << rho0 + 1
                        << "\t" << eta0 + 1 << "\t"
                        << RMEs[ai * alpha_f + af](rho * etamax + eta,
                                                   rho0 * eta0max + eta0)
                        << std::endl;
    }

#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}