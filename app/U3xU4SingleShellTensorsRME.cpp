#include <chrono>
#include <fstream>
#include <iostream>

#include <U3xU4Irrep.h>
#include <U3xU4SingleShellTensors.h>

#include <qcx_ND.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  qcx_utils::SO3_CGC cgcs;
  u3::WigU3U2_db wigU3U2db(cgcs);
  u4::WigU4U3_db wigU4U3db(cgcs, wigU3U2db);

  std::vector<U3xU4::irrep> prod_ads, prod_ats;

  /* Example 1:
    < alpha_i (4,3,1) (3,1,0,0) |||
                 (3,1,0)       (3,3,0)            (3,2,0)   (4,1,0)
      { [[ad x ad]         x ad]        x [at x at]        }
                 (1,1,0,0)     (2,1,0,0)          (1,1,0,0) (1,0,0,0)
                                        ||| alpha_f (5,3,2) (3,1,1,0) >
  */
  /*
  std::size_t n = 2, ads = 3, ats = 2;
  U3xU4::irrep i(4,3,1, 3,1,0,0), f(5,3,2, 3,1,1,0),
               T1_0(3,1,0, 1,1,0,0), T1(3,3,0, 2,1,0,0),
               T2(3,2,0, 1,1,0,0), T(4,1,0, 1,0,0,0);

  prod_ads.push_back(T1_0);
  prod_ads.push_back(T1);
  prod_ats.emplace_back(T2);

  std::size_t rho0max = u3::Rho(T1.u3,T2.u3,T.u3),
              eta0max = u4::Eta(T1.u4,T2.u4,T.u4);
  */

  /* Example 2:
    < alpha_i (14,2,0) (1,1,1,1) |||
                (6,2,0)             (6,4,0)   (4,2,0)
      { [ad x ad]         x [at x at]         }
                (1,1,0,0)           (1,1,0,0) (0,0,0,0)
                ||| alpha_f (14,2,0) (1,1,1,1) >
  */

  std::size_t n = 4, ads = 2, ats = 2;
  U3xU4::irrep i(14, 2, 0, 1, 1, 1, 1), f(14, 2, 0, 1, 1, 1, 1),
      T1(6, 2, 0, 1, 1, 0, 0), T2(6, 4, 0, 1, 1, 0, 0), T(4, 2, 0, 0, 0, 0, 0);

  prod_ads.push_back(T1);
  prod_ats.push_back(T2);

  std::size_t rho0max = u3::Rho(T1.u3, T2.u3, T.u3),
              eta0max = u4::Eta(T1.u4, T2.u4, T.u4);

  /* Example 3:
                              (n,0,0)
    < (n,0,0) (1,0,0,0) ||| ad         ||| (0,0,0) (0,0,0,0) >
                              (1,0,0,0)

  */
  /*
  std::size_t ads = 1, ats = 0, n = 4;
  U3xU4::irrep i(0,0,0, 0,0,0,0), T(n,0,0, 1,0,0,0), f(n,0,0, 1,0,0,0);
  std::size_t rho0max = 1, eta0max = 1;
  */

  /* Example 4:
                              (n,n,0)
    < (0,0,0) (0,0,0,0) ||| at         ||| (n,0,0) (1,0,0,0) >
                              (1,1,1,0)

  */
  /*
  std::size_t ads = 0, ats = 1, n = 4;
  U3xU4::irrep i(n,0,0, 1,0,0,0), T(n,n,0, 1,1,1,0), f(0,0,0, 0,0,0,0);
  std::size_t rho0max = 1, eta0max = 1;
  */

  /* Example 5: scalar
                                    (0,0,0)
    / (n13 n23 n33)     ||| [ad x at]         ||| (n13 n23 n33)     \
    \ (n14 n24 n34 n44) |||         (0,0,0,0) ||| (n14 n24 n34 n44) /
  */
  /*
  std::size_t n = 8, ads = 1, ats = 1;
  U3xU4::irrep i(n,0,0, 1,0,0,0), T(0,0,0, 0,0,0,0), f(n,0,0, 1,0,0,0);
  std::size_t rho0max = 1, eta0max = 1;
  */

  ////////////////////////////////////

  U3xU4::Tadat Tadxat(n, ads, ats, prod_ads, prod_ats, T, rho0max, eta0max);

  std::size_t ai_max, af_max, rhomax, etamax;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto RMEs = Tadxat.RME(
      ai_max, i, af_max, f, rhomax, etamax, cgcs, wigU3U2db, wigU4U3db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "< " << af_max << f << " ||| " << Tadxat << " ||| " << ai_max
            << i << " >" << rhomax << "," << etamax << std::endl;

  std::cout << "ai\taf\trho0\teta0\trho\teta" << std::endl;

  qcx_utils::NDIndex multis;
  multis.add_dimension({ai_max, af_max, rho0max, eta0max, rhomax, etamax});

  std::vector<size_t> multi(6);
  for (std::size_t i = 0; i < RMEs.size(); i++)
  {
    multis.to_multi(i, multi);
    std::cout << multi[0] + 1 << "\t" << multi[1] + 1 << "\t"
              << multi[2] + 1 << "\t" << multi[3] + 1 << "\t"
              << multi[4] + 1 << "\t" << multi[5] + 1 << "\t"
              << RMEs[i] << std::endl;
  }

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}
