#include <sam.h>

#include <OneBodyOperatorJT.h>
#include <TwoBodyOperatorJT.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  int hw = 16, valenceShell = 0, Nmax = 6, maxShell = Nmax;
  HO::Table_nljmjmt HO_labels(maxShell);
  U3xU4::Table_gelfand gelfand_labels(maxShell);

  // YOU MUST CHOOSE ONLY ONE OF THE FOLLOWING OPTIONS.
  // NOTE THAT THESE OPERATORS MUST HAVE BEEN EXPANDED
  // AND WRITTEN TO BINARY FILES.

  ////////////// ONE-BODY OPERATORS //////////////

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

  // JT::OneBodyOperator 
  //   op(name, hw, Nmax, valenceShell, maxShell, HO_labels, gelfand_labels);

  ////////////// TWO-BODY OPERATORS //////////////

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
  // std::string name = "N2LO_opt_bare";
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

  // Example 10: Two-body GT operator
  std::string name = "GamowTellerMinus_2B_N3LO_EM500_bare";

  JT::TwoBodyOperator 
    op(name, hw, Nmax, valenceShell, maxShell, HO_labels, gelfand_labels);

  ////////////// DON'T EDIT ANYTHING BELOW //////////////
  
  op.is();

  auto tensors_U3_stats = op.U3_tensor_analyzer_write();
  std::cout << "U3 tensor characters:" << std::endl;
  for (const auto& [irr, stats] : tensors_U3_stats)
    std::cout << irr << ":\t" << stats << std::endl;

  auto tensors_U4_stats = op.U4_tensor_analyzer_write();
  std::cout << "U4 tensor characters:" << std::endl;
  for (const auto& [irr, stats] : tensors_U4_stats)
    std::cout << irr << ":\t" << stats << std::endl;

  auto tensors_LL_stats = op.LL_tensor_analyzer_write();
  std::cout << "LL tensor characters:" << std::endl;
  for (const auto& [irr, stats] : tensors_LL_stats)
    std::cout << irr << ":\t" << stats << std::endl;

  auto tensors_SS_stats = op.SS_tensor_analyzer_write();
  std::cout << "SS tensor characters:" << std::endl;
  for (const auto& [irr, stats] : tensors_SS_stats)
    std::cout << irr << ":\t" << stats << std::endl;

  
  sam::finalize();
}
