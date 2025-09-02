#include <chrono>
#include <iostream>

#include <Databases.h>
#include <TwoBodyOperatorJT.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  ///////// GENERAL COMMENTS ///////////
  // Expanding any two-body JT-coupled operator in terms of 
  // two-body U3xU4 multi-shell tensors can be done via three methods:
  // 1) expand(...databases...): calculate the expansion on-the-fly and
  //    then print out on terminal.
  // 2) writeBinary(...databases...): the expansion from 1) will be 
  //    stored in a binary file in SAM/data/TwoBodyTensorExpansion. 
  //    Since this method aims at reusability, one must set valanceShell 
  //    to 0, and it is recommended to set maxShell = Nmax.
  // 3) readBinary(): read back the expansion from binary file generated
  //    in 2). Note that it only reads back relevant tensors allowed by
  //    valanceShell, Nmax, and maxShell. This should take only a fraction
  //    of a second.
  //////////////////////////////////////

  // Example 1: Two-body part of LL
  
  int hw = 20, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator LL("L.L", labels, hw, Nmax, valenceShell, maxShell);

  // expand now
  // auto tensors = LL.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  // LL.writeBinary(dbs);

  // read from binary
  auto tensors = LL.readBinary();
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  LL.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  

  // Example 2: Two-body part of relative kinetic energy
  /*
  int hw = 20, valenceShell = 1, Nmax = 2, maxShell = 3;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator KE("Kinetic",labels,hw,Nmax,valenceShell,maxShell);

  // expand now
  // auto tensors = KE.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  KE.writeBinary(dbs);

  // read from binary
  auto tensors = KE.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  KE.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

  // Example 3: Two-body potential (bare) with N2LO_opt
  /*
  int hw = -1, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator NN("N2LO_opt_bare", labels, hw, Nmax, valenceShell,
  maxShell);

  // expand now
  // auto tensors = NN.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  // NN.writeBinary(dbs);

  // read from binary
  auto tensors = NN.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  NN.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

  // Example 4: Two-body potential (bare) with N3LO_EMN500
  /*
  int hw = -1, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator NN("N3LO_EMN500_bare", labels, hw, Nmax, valenceShell,
                         maxShell);

  // expand now
  // auto tensors = NN.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  // NN.writeBinary(dbs);

  // read from binary
  auto tensors = NN.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  NN.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

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
