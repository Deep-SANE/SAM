#include <chrono>
#include <iostream>

#include <OneBodyOperatorJT.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  u34utils::SO3_CGC cgcs;
  u34utils::SO3_Racah9 racah9;
  u3::WigU3U2_db wigU3U2db(cgcs);
  u3::WigU3SO3_db wigU3SO3db(cgcs, wigU3U2db);
  u4::WigU4U3_db wigU4U3db(cgcs, wigU3U2db);
  u4::WigU4SU2SU2_db wigU4SU2SU2db(cgcs, wigU3U2db, wigU4U3db);

  // Example 1: number operator
  
  JT::OneBodyOperator N(0, 0, 0, 0, 1);
  int maxShell = 10;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors =
      N.expand(maxShell, cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  N.is();
  for (auto tensor : tensors) std::cout << tensor << std::endl;
  

  // Example 2: one-body part of the L^2 operator
  /*
  JT::OneBodyOperator oneL2(0, 0, 0, 0, 2);
  int maxShell = 6;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors =
      oneL2.expand(maxShell, cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  oneL2.is();
  for (auto tensor : tensors) std::cout << tensor << std::endl;
  */

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