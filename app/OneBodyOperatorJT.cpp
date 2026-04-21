#include <chrono>
#include <iostream>

#include <sam.h>

#include <OneBodyOperatorJT.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  int hw = 16, valenceShell = 0, Nmax = 3, maxShell = Nmax;
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
  // std::string name = "GamowTellerPlus";

  // Example 11: Two-body potential with custom name
  std::string name = "HO_Li7_e3_hw16_E39";

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
  for (auto tensor : tensors) std::cout << tensor << std::endl;

  /*
  // This might be helpful for comparison
  auto tensors_new = op.expand_new(); // expand on the fly

  assert(tensors.size() == tensors_new.size());
  for (size_t i = 0; i < tensors.size(); i++)
  {
    std::cout << "Old: " << tensors[i] << std::endl;
    std::cout << "New: " << tensors_new[i] << std::endl;

    assert(tensors[i] == tensors_new[i]);

    assert(tensors[i].LLSSTTstrengths().size() == 
           tensors_new[i].LLSSTTstrengths().size());

    for (size_t j = 0; j < tensors[i].LLSSTTstrengths().size(); j++)
    {
      const auto& [k, LL, z, SS, TT, strengths] = tensors[i].LLSSTTstrengths()[j];
      const auto& [k_, LL_, z_, SS_, TT_, strengths_] = tensors_new[i].LLSSTTstrengths()[j];

      assert(k == k_ && LL == LL_ && z == z_ && SS == SS_ && TT == TT_);

      assert(strengths.size() == strengths_.size());
      for (size_t s = 0; s < strengths.size(); s++)
        assert(std::fabs(strengths[s] - strengths_[s]) < 1e-8);
    }
  }
  */

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
