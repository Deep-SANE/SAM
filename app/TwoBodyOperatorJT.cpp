#include <chrono>
#include <iostream>

#include <TwoBodyOperatorJT.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  qcx_utils::SO3_CGC cgcs;
  qcx_utils::SO3_Racah racah;
  qcx_utils::SO3_Racah9 racah9;
  u3::WigU3U2_db wigU3U2db(cgcs);
  u3::WigU3SO3_db wigU3SO3db(cgcs, wigU3U2db);
  u4::WigU4U3_db wigU4U3db(cgcs, wigU3U2db);
  u4::WigU4SU2SU2_db wigU4SU2SU2db(cgcs, wigU3U2db, wigU4U3db);

  // Example 1: Two-body part of LL
  /*
  int hw = 20, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator LL("L.L", labels, hw, Nmax, valenceShell, maxShell);

  auto tensors = LL.expand(cgcs, racah, racah9, wigU3SO3db, wigU4SU2SU2db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  LL.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

  // Example 2: Two-body part of relative kinetic energy
  /*
  int hw = 20, valenceShell = 1, Nmax = 2, maxShell = 3;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator KE("Kinetic",labels,hw,Nmax,valenceShell,maxShell);

  auto tensors = KE.expand(cgcs,racah,racah9,wigU3SO3db,wigU4SU2SU2db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  KE.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

  // Example 3: Two-body Hamiltonian (bare) with N2LO_opt
  /*
  int hw = -1, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

    // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator NN("N2LO_opt_bare", labels, hw, Nmax, valenceShell,
  maxShell);

  auto tensors = NN.expand(cgcs, racah, racah9, wigU3SO3db, wigU4SU2SU2db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  NN.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

  // Example 4: Two-body Hamiltonian (bare) with N3LO_EMN500
  int hw = -1, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator NN("N3LO_EMN500_bare", labels, hw, Nmax, valenceShell,
                         maxShell);

  auto tensors = NN.expand(cgcs, racah, racah9, wigU3SO3db, wigU4SU2SU2db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  NN.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;

  std::cout << "There are " << tensors.size() << " tensors" << std::endl;

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
