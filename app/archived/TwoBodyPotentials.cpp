#include <chrono>
#include <iostream>
#include <tuple>

#include <sam.h>

#include <TwoBodyOperatorJT.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  int hw = 20, valenceShell = 0, Nmax = 4, maxShell = Nmax;
  HO::Table_nljmjmt HO_labels(maxShell);
  U3xU4::Table_gelfand gelfand_labels(maxShell);

  // Two-body potential (bare) with N2LO_opt_bare,srg1.0,2.0,4.0
  std::string a = "N2LO_opt_bare";
  std::string b = "N2LO_opt_srg1.0";
  std::string c = "N2LO_opt_srg2.0";
  std::string d = "N2LO_opt_srg4.0";

  // Two-body potential (bare) with N3LO_EMN500_bare,srg1.0,2.0,4.0
  std::string e = "N3LO_EMN500_bare";
  std::string f = "N3LO_EMN500_srg1.0";
  std::string g = "N3LO_EMN500_srg2.0";
  std::string h = "N3LO_EMN500_srg4.0";
  
  // Two-body potential (bare) with N4LO_EMN500_bare,srg1.0,2.0,4.0
  std::string i = "N4LO_EMN500_bare";
  std::string j = "N4LO_EMN500_srg1.0";
  std::string k = "N4LO_EMN500_srg2.0";
  std::string l = "N4LO_EMN500_srg4.0";

  // Two-body potential (bare) with N2LO_sat_bare,srg1.0,2.0,4.0
  std::string m = "N2LO_sat_bare";
  std::string n = "N2LO_sat_srg1.0";
  std::string o = "N2LO_sat_srg2.0";
  std::string p = "N2LO_sat_srg4.0";

  std::vector<std::string> names = {a, e};

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  JT::TwoBodyOperator 
    op(names, hw, Nmax, valenceShell, maxShell, HO_labels, gelfand_labels);
  op.is();

  auto tensors = op.expand_potentials(false);
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  
  for (int id = 0; id < names.size(); id++)
  {
    std::cout << "There are " << tensors[id].size() << " tensors for "
              << names[id] << std::endl;

    for (const auto& tensor : tensors[id])  std::cout << tensor << std::endl;
  }

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
