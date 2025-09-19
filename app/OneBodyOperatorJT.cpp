#include <chrono>
#include <iostream>

#include <Databases.h>
#include <OneBodyOperatorJT.h>

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

  // Example 1: number operator
  /*
  int hw = -1, maxShell = 10;
  JT::OneBodyOperator N("NumberOp",hw,maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors = N.expand(dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  N.is();
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;
  */

  // Example 2: one-body part of the L.L operator
  /*
  int hw = -1, maxShell = 4;
  JT::OneBodyOperator LL("L.L",hw,maxShell);
  LL.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors = LL.expand(dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 3: one-body part of the relative kinetic energy
  /*
  int hw = -1, maxShell = 4;
  JT::OneBodyOperator KE("Kinetic", hw, maxShell);
  KE.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors = KE.expand(dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 4: one-body part of the CoM HO Number Operator
  /*
  int hw = -1, maxShell = 4;
  JT::OneBodyOperator CoM("CoMNumberOp", hw, maxShell);
  CoM.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors = CoM.expand(dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  // Example 5: one-body part of the L.S operator
  
  int hw = -1, maxShell = 4;
  JT::OneBodyOperator LS("L.S",hw,maxShell);
  LS.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto tensors = LS.expand(dbs);

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
