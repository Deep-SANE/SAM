#include <chrono>
#include <iostream>

#include <Databases.h>
#include <OneBodyOperatorJT.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  // Example 1: number operator
  /*
  int hw = -1, maxShell = 10;
  JT::OneBodyOperator N("NumberOp",hw,maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors = N.expand(cgcs, racah, racah9, wigU3SO3db, wigU4SU2SU2db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  N.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

  // Example 2: one-body part of the L.L operator
  /*
  int hw = -1, maxShell = 6;
  JT::OneBodyOperator LL("L.L",hw,maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors = LL.expand(cgcs, racah, racah9, wigU3SO3db, wigU4SU2SU2db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  LL.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

  // Example 3: one-body part of the relative kinetic energy

  int hw = -1, maxShell = 8;
  JT::OneBodyOperator KE("Kinetic", hw, maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors = KE.expand(dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  KE.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;

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
