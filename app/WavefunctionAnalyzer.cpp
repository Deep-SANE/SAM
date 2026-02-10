#include <sam.h>

#include <qcx_print.h>

#include <OperatorJT.h>
#include <WavefunctionAnalyzer.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
/*
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
*/

  int N = 3, Z = 3, Nmax = 6, JJ = 2, hw = 20;

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);
  basis.construct();

  int valenceShell = HO::valenceShell(Z + N),
      maxShell = valenceShell + Nmax;

  HO::Table_nljmjmt HO_labels(maxShell);
  U3xU4::Table_gelfand gelfand_labels(maxShell);

  std::string Hamil_name = "Hamiltonian with N2LO_opt_bare";
  JT::Operator Hamil(Hamil_name, N, Z, hw, Nmax, HO_labels, gelfand_labels);

  qcx_utils::print("N={}, Z={}, Nmax={}, JJ={}, hw={} with {}\n",
                    N, Z, Nmax, JJ, hw, Hamil_name);

  // read Hamiltonian matrix
  auto filename = Hamil.filename(basis);
  auto mat = qcx_utils::readBinary(filename);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto [evals, evecs] = qcx_utils::eigensolver_symmetric(mat, 3);
  qcx_utils::print("Eigenvalues: {}\n", evals);

  auto subspaces_u3 = sam::U3_wavefunction_analyzer(basis, evecs, 2);

  auto subspaces_u4 = sam::U4_wavefunction_analyzer(basis, evecs, 2);

  for (const auto& subspace : subspaces_u3)
  	std::cout << subspace << std::endl;

  for (const auto& subspace : subspaces_u4)
    std::cout << subspace << std::endl;

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
