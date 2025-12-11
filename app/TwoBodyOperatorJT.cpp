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
  /*
  int hw = 20, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator LL("L.L", labels, hw, Nmax, valenceShell, maxShell);
  LL.is();

  // expand now
  // auto tensors = LL.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  // LL.writeBinary(dbs);

  // read from binary
  auto tensors = LL.readBinary();
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 2: Two-body part of LS
  /*
  int hw = 20, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator LS("L.S", labels, hw, Nmax, valenceShell, maxShell);
  LS.is();

  // expand now
  // auto tensors = LS.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  // LS.writeBinary(dbs);

  // read from binary
  auto tensors = LS.readBinary();
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 3: Two-body part of relative kinetic energy
  /*
  int hw = 20, valenceShell = 0, Nmax = 6, maxShell = 6;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator KE("Kinetic", labels, hw, Nmax, valenceShell, maxShell);
  KE.is();

  // expand now
  // auto tensors = KE.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  KE.writeBinary(dbs);

  // read from binary
  auto tensors = KE.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 4: Two-body part of the CoM HO operator
  
  int hw = 20, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    CoM("CoMNumberOp", labels, hw, Nmax, valenceShell, maxShell);
  CoM.is();

  // expand now
  // auto tensors = CoM.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  // CoM.writeBinary(dbs);

  // read from binary
  auto tensors = CoM.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  

  // Example 5: Two-body potential (bare) with N2LO_opt_bare,srg1.0,2.0,4.0
  /*
  int hw = 20, valenceShell = 0, Nmax = 6, maxShell = 6;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    NN("N2LO_opt_bare", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N2LO_opt_srg1.0", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N2LO_opt_srg2.0", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N2LO_opt_srg4.0", labels, hw, Nmax, valenceShell, maxShell);
  NN.is();

  // expand now
  // auto tensors = NN.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  NN.writeBinary(dbs);

  // read from binary
  auto tensors = NN.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 6: Two-body potential (bare) with N3LO_EMN500_bare,srg1.0,2.0,4.0
  /*
  int hw = 20, valenceShell = 0, Nmax = 6, maxShell = 6;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    NN("N3LO_EMN500_bare", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N3LO_EMN500_srg1.0", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N3LO_EMN500_srg2.0", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N3LO_EMN500_srg4.0", labels, hw, Nmax, valenceShell, maxShell);
  NN.is();

  // expand now
  // auto tensors = NN.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  NN.writeBinary(dbs);

  // read from binary
  auto tensors = NN.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 7: Two-body potential (bare) with N2LO_sat_bare,srg1.0,2.0,4.0
  /*
  int hw = -1, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    NN("N2LO_sat_bare", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N2LO_sat_srg1.0", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N2LO_sat_srg2.0", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N2LO_sat_srg4.0", labels, hw, Nmax, valenceShell, maxShell);
  NN.is();

  // expand now
  // auto tensors = NN.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  NN.writeBinary(dbs);

  // read from binary
  auto tensors = NN.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */
  
  // Example 8: Two-body potential (bare) with N4LO_EMN500_bare,srg1.0,2.0,4.0
  /*
  int hw = -1, valenceShell = 0, Nmax = 6, maxShell = 6;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    NN("N4LO_EMN500_bare", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N4LO_EMN500_srg1.0", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N4LO_EMN500_srg2.0", labels, hw, Nmax, valenceShell, maxShell);
    //NN("N4LO_EMN500_srg4.0", labels, hw, Nmax, valenceShell, maxShell);
  NN.is();

  // expand now
  // auto tensors = NN.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  NN.writeBinary(dbs);

  // read from binary
  auto tensors = NN.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  ////////////// DON'T EDIT BELOW //////////////

  std::cout << "There are " << tensors.size() << " tensors" << std::endl;
  for (const auto& tensor : tensors)
  {
    std::cout << tensor << std::endl;
  }
  
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
