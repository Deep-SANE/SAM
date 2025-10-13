#include <WavefunctionAnalyzer.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  int Z, N, Nmax, JJ, hw;

  std::cout << "Enter number of protons: ";
  std::cin >> Z;
  std::cout << "Enter number of neutrons: ";
  std::cin >> N;
  std::cout << "Enter Nmax: ";
  std::cin >> Nmax;
  std::cout << "Enter JJ: ";
  std::cin >> JJ;
  std::cout << "Enter hw: ";
  std::cin >> hw;

  U3xU4::ManyNucleonBasis basis(Z, N, Nmax, JJ);
  basis.construct();

  const char* sam_dir = std::getenv("SAM_DIR");
  if (!sam_dir)
    throw std::runtime_error("Environment variable SAM_DIR not set");

  std::string base_dir = std::string(sam_dir) +
                         "/data/OperatorJT/Hamiltonian/Z" + std::to_string(Z) +
                         "_N" + std::to_string(N) + "/JJ" + std::to_string(JJ) +
                         "/Nmax" + std::to_string(Nmax);
  std::string filename =
      base_dir + "/" + "hw" + std::to_string(hw) + "_N2LO_opt_bare";

  //auto [evals, subspaces] = sam::U3xU4_wavefunction_analyzer(basis,filename,1);


  //auto [evals, subspaces] = sam::U3_wavefunction_analyzer(basis,filename,1);


  //auto [evals, subspaces] = sam::U4_wavefunction_analyzer(basis,filename,1);
  auto [evals, subspaces] = sam::U4_wavefunction_analyzer_write(basis,filename,1);

  std::cout << "Eigenvalues: ";
  for (const auto& e : evals)
  	std::cout << e << "\t";
  std::cout << std::endl;

  for (const auto& subspace : subspaces)
  	std::cout << subspace << std::endl;

#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}
