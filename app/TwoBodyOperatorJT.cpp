#include <chrono>
#include <iostream>

#include <Databases.h>
#include <TwoBodyOperatorJT.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

double avg_strength(const std::vector<double>& vs)
{
  double sum = 0.0;
  for (const auto& v : vs) sum += v;
  return sum / vs.size();
}

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
  int hw = 20, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator KE("Kinetic", labels, hw, Nmax, valenceShell, maxShell);
  KE.is();

  // expand now
  // auto tensors = KE.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  // KE.writeBinary(dbs);

  // read from binary
  auto tensors = KE.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 4: Two-body potential (bare) with N2LO_opt
  /*
  int hw = 20, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    NN("N2LO_opt_bare", labels, hw, Nmax, valenceShell, maxShell);
  NN.is();

  // expand now
  // auto tensors = NN.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  // NN.writeBinary(dbs);

  // read from binary
  auto tensors = NN.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 5: Two-body potential (bare) with N3LO_EMN500
  /*
  int hw = -1, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    NN("N3LO_EMN500_bare", labels, hw, Nmax, valenceShell, maxShell);
  NN.is();

  // expand now
  // auto tensors = NN.expand(dbs);

  // write to binary, valenceShell must be 0 for future reuse
  // NN.writeBinary(dbs);

  // read from binary
  auto tensors = NN.readBinary();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

  // Example 6: Two-body part of the CoM HO operator
  
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
  
  
  std::vector<std::vector<double>> strengths_U3, strengths_U4;
  std::vector<u3::irrep> U3s;
  std::vector<u4::irrep> U4s;

  for (auto tensor : tensors)
  {
    // std::cout << tensor << std::endl;

    auto U3 = tensor.getLastIrrep().u3();
    auto U4 = tensor.getLastIrrep().u4();

    auto it3 = std::find(U3s.begin(), U3s.end(), U3);
    if (it3 != U3s.end())
    {
      size_t index = std::distance(U3s.begin(), it3);
      for (const auto c : tensor.coeffs) 
        strengths_U3[index].push_back(c);
    }
    else
    {
      U3s.push_back(U3);
      strengths_U3.push_back(tensor.coeffs);
    }

    auto it4 = std::find(U4s.begin(), U4s.end(), U4);
    if (it4 != U4s.end())
    {
      size_t index = std::distance(U4s.begin(), it4);
      for (const auto c : tensor.coeffs) 
        strengths_U4[index].push_back(c);
    }
    else
    {
      U4s.push_back(U4);
      strengths_U4.push_back(tensor.coeffs);
    }
  }
  
  std::cout << "Average strengths for U(3)" << std::endl;
  for (size_t id = 0; id < U3s.size(); id++)
    std::cout << U3s[id] << ": " << avg_strength(strengths_U3[id]) << std::endl;

  std::cout << "Average strengths for U(4)" << std::endl;
  for (size_t id = 0; id < U4s.size(); id++)
    std::cout << U4s[id] << ": " << avg_strength(strengths_U4[id]) << std::endl;
  

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
