#include <chrono>
#include <iostream>
#include <tuple>

#include <Databases.h>
#include <TwoBodyOperatorJT.h>

#ifdef USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  int hw = 20, valenceShell = 0, Nmax = 12, maxShell = Nmax;
  HO::Table_nljmjmt labels(maxShell);

  // Example 1: Two-body part of LL
  // std::string name = "L.L";

  // Example 2: Two-body part of LS
  // std::string name = "L.S";

  // Example 3: Two-body part of relative kinetic energy
  std::string name = "Kinetic";  

  // Example 4: Two-body part of the CoM HO operator
  // std::string name = "CoMNumberOp";

  // Example 5: Two-body potential (bare) with N2LO_opt_bare,srg1.0,2.0,4.0
  // std::string name = "N2LO_opt_bare";
  // std::string name = "N2LO_opt_srg1.0";
  // std::string name = "N2LO_opt_srg2.0";
  // std::string name = "N2LO_opt_srg4.0";

  // Example 6: Two-body potential (bare) with N3LO_EMN500_bare,srg1.0,2.0,4.0
  // std::string name = "N3LO_EMN500_bare";
  // std::string name = "N3LO_EMN500_srg1.0";
  // std::string name = "N3LO_EMN500_srg2.0";
  // std::string name = "N3LO_EMN500_srg4.0";

  // Example 7: Two-body potential (bare) with N2LO_sat_bare,srg1.0,2.0,4.0
  // std::string name = "N2LO_sat_bare";
  // std::string name = "N2LO_sat_srg1.0";
  // std::string name = "N2LO_sat_srg2.0";
  // std::string name = "N2LO_sat_srg4.0";
  
  // Example 8: Two-body potential (bare) with N4LO_EMN500_bare,srg1.0,2.0,4.0
  // std::string name = "N4LO_EMN500_bare";
  // std::string name = "N4LO_EMN500_srg1.0";
  // std::string name = "N4LO_EMN500_srg2.0"
  // std::string name = "N4LO_EMN500_srg4.0";

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator op(name, hw, Nmax, valenceShell, maxShell, labels);
  op.is();

  auto tensors = op.expand(dbs);
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "There are " << tensors.size() << " tensors" << std::endl;
  for (const auto& tensor : tensors)
    std::cout << tensor << std::endl;
  
  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

#ifdef USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}
