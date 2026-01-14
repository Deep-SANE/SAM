#include <chrono>
#include <iostream>

#include <qcx_utils.h>
#include <u3lib.h>
#include <u4lib.h>

#include <OneBodyOperatorJT.h>

double avg_strength(const std::vector<double>& vs)
{
  double sum = 0.0;
  for (const auto& v : vs) sum += v;
  return sum / vs.size();
}

int main(int argc, char* argv[])
{
  qcx_utils::initialize(&argc, &argv);
  u3::initialize();
  u4::initialize();

  int hw = 20, valenceShell = 0, Nmax = 14, maxShell = Nmax;
  HO::Table_nljmjmt labels(maxShell);

  // Example 1: One-body part of LL
  // std::string name = "L.L";

  // Example 2: One-body part of LS
  // std::string name = "L.S";

  // Example 3: One-body part of relative kinetic energy
  // std::string name = "Kinetic";

  // Example 4: One-body part of the CoM HO operator
  // std::string name = "CoMNumberOp";

  // Example 5: Electric quadrupole operator
  // std::string name = "E2";

  // Example 6: Magnetic dipole operator
  // std::string name = "M1";

  // Example 7: Intrinsic mass radius squared
  std::string name = "R.R";

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::OneBodyOperator op(name, hw, Nmax, valenceShell, maxShell, labels);
  op.is();

  auto tensors = op.expand();
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  
  std::cout << "There are " << tensors.size() << " tensors" << std::endl;
  for (auto tensor : tensors)
    std::cout << tensor << std::endl;

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  u4::finalize();
  u3::finalize();
  qcx_utils::finalize();
}
