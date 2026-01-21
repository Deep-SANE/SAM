#include <chrono>
#include <iostream>

#include <qcx_utils.h>
#include <u3lib.h>
#include <u4lib.h>

#include <OneBodyOperatorJT.h>

int main(int argc, char* argv[])
{
  qcx_utils::initialize(&argc, &argv);
  u3::initialize();
  u4::initialize();

  int hw = 20, valenceShell = 0, Nmax = 14, maxShell = Nmax;
  HO::Table_nljmjmt HO_labels(maxShell);
  U3xU4::Table_gelfand gelfand_labels(maxShell);

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

  // Example 7: Intrinsic mass radius
  // std::string name = "Rrel";

  // Example 8: Point-nucleon root-mean-squared radius
  // std::string name = "Radius";

  // Example 9: Point-proton root-mean-squared radius
  // std::string name = "Radius_p";

  // Example 10: Point-neutron root-mean-squared radius
  // std::string name = "Radius_n";

  // Example 11: Fermi for beta decay
  // std::string name = "FermiMinus";
  // std::string name = "FermiPlus";

  // Example 12: Gamow-Teller for beta decay
  // std::string name = "GamowTellerMinus";
  std::string name = "GamowTellerPlus";

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::OneBodyOperator 
    op(name, hw, Nmax, valenceShell, maxShell, HO_labels, gelfand_labels);
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
