#include <chrono>
#include <iostream>
#include <tuple>

#include <sam.h>

#include <TwoBodyOperatorJT.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  int hw = 20, valenceShell = 0, Nmax = 8, maxShell = Nmax;
  HO::Table_nljmjmt HO_labels(maxShell);
  U3xU4::Table_gelfand gelfand_labels(maxShell);

  // Example 1: Two-body part of LL
  // std::string name = "L.L";

  // Example 2: Two-body part of LS
  // std::string name = "L.S";

  // Example 3: Two-body part of relative kinetic energy
  // std::string name = "Kinetic";  

  // Example 4: Two-body part of the CoM HO operator
  // std::string name = "CoMNumberOp";

  // Example 5: Intrinsic mass radius squared
  // std::string name = "Rrel";

  // Example 6: Two-body potential (bare) with N2LO_opt_bare,srg1.0,2.0,4.0
  std::string name = "N2LO_opt_bare";
  // std::string name = "N2LO_opt_srg1.0";
  // std::string name = "N2LO_opt_srg2.0";
  // std::string name = "N2LO_opt_srg4.0";

  // Example 7: Two-body potential (bare) with N3LO_EMN500_bare,srg1.0,2.0,4.0
  // std::string name = "N3LO_EMN500_bare";
  // std::string name = "N3LO_EMN500_srg1.0";
  // std::string name = "N3LO_EMN500_srg2.0";
  // std::string name = "N3LO_EMN500_srg4.0";

  // Example 8: Two-body potential (bare) with N2LO_sat_bare,srg1.0,2.0,4.0
  // std::string name = "N2LO_sat_bare";
  // std::string name = "N2LO_sat_srg1.0";
  // std::string name = "N2LO_sat_srg2.0";
  // std::string name = "N2LO_sat_srg4.0";
  
  // Example 9: Two-body potential (bare) with N4LO_EMN500_bare,srg1.0,2.0,4.0
  // std::string name = "N4LO_EMN500_bare";
  // std::string name = "N4LO_EMN500_srg1.0";
  // std::string name = "N4LO_EMN500_srg2.0"
  // std::string name = "N4LO_EMN500_srg4.0";

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    op(name, hw, Nmax, valenceShell, maxShell, HO_labels, gelfand_labels);
  op.is();

  auto tensors = op.expand();
  // auto tensors_new = op.expand_new(); // expand on the fly
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  
  std::cout << "There are " << tensors.size() << " tensors" << std::endl;
  for (const auto& tensor : tensors)
    std::cout << tensor << std::endl;

/*
  // This might be helpful for comparison
  assert(tensors.size() == tensors_new.size());
  for (size_t i = 0; i < tensors.size(); i++)
  {
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
