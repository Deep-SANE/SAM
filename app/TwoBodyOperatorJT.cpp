#include <chrono>
#include <iostream>
#include <tuple>

#include <sam.h>

#include <TwoBodyOperatorJT.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  int hw = 20, valenceShell = 0, Nmax = 10, maxShell = Nmax;
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

  // Example 6: Two-body potential N2LO_opt_bare,srg1.0,2.0,4.0
  // std::string name = "N2LO_opt_bare";
  // std::string name = "N2LO_opt_srg1.0";
  // std::string name = "N2LO_opt_srg2.0";
  // std::string name = "N2LO_opt_srg4.0";

  // Example 7: Two-body potential N3LO_EMN500_bare,srg1.0,2.0,4.0
  // std::string name = "N3LO_EMN500_bare";
  // std::string name = "N3LO_EMN500_srg1.0";
  // std::string name = "N3LO_EMN500_srg2.0";
  // std::string name = "N3LO_EMN500_srg4.0";

  // Example 8: Two-body potential N2LO_sat_bare,srg1.0,2.0,4.0
  // std::string name = "N2LO_sat_bare";
  // std::string name = "N2LO_sat_srg1.0";
  // std::string name = "N2LO_sat_srg2.0";
  // std::string name = "N2LO_sat_srg4.0";

  // Example 9: Two-body potential N4LO_EMN500_bare,srg1.0,2.0,4.0
  // std::string name = "N4LO_EMN500_bare";
  // std::string name = "N4LO_EMN500_srg1.0";
  // std::string name = "N4LO_EMN500_srg2.0"
  // std::string name = "N4LO_EMN500_srg4.0";

  // Example 10: Two-body potential N3LO_EM500_bare,srg1.0,2.0,4.0
  // std::string name = "N3LO_EM500_bare";
  // std::string name = "N3LO_EM500_srg1.0";
  std::string name = "N3LO_EM500_srg2.0";
  // std::string name = "N3LO_EMN500_srg4.0";

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    op(name, hw, Nmax, valenceShell, maxShell, HO_labels, gelfand_labels);
  op.is();

  auto tensors = op.expand();
  // auto tensors = op.expand_new();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "There are " << tensors.size() << " tensors" << std::endl;
  for (const auto& tensor : tensors)  std::cout << tensor << std::endl;

  /*
  // This might be helpful for comparison
  std::string filename = op.dirname() + "/Expansion/" + name + "_hw" 
                         + std::to_string(hw) + "_Nmax6_DL";
  auto tensors_old = op.readBinary(filename);

  // assert(tensors.size() == tensors_old.size());
  for (const auto& tensor : tensors)
  {
    bool found = false;
    for (const auto& tensor_old : tensors_old)
    {
      if (tensor == tensor_old)
      {
        found = true;

        std::cout << "New: " << tensor << std::endl;
        std::cout << "Old: " << tensor_old << std::endl;

        assert(tensor.LLSSTTstrengths().size() == 
               tensor_old.LLSSTTstrengths().size());

        for (size_t j = 0; j < tensor.LLSSTTstrengths().size(); j++)
        {
          const auto& [k, LL, z, SS, TT, strengths] = tensor.LLSSTTstrengths()[j];
          const auto& [k_, LL_, z_, SS_, TT_, strengths_] = tensor_old.LLSSTTstrengths()[j];

          assert(k == k_ && LL == LL_ && z == z_ && SS == SS_ && TT == TT_);

          assert(strengths.size() == strengths_.size());
          for (size_t s = 0; s < strengths.size(); s++)
            assert(std::fabs(strengths[s] - strengths_[s]) < 1e-8);
        }

        break;
      }
    }
    assert(found);
  }
  */

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
