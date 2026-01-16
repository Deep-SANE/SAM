#include <qcx_utils.h>

#include <u3lib.h>
#include <u4lib.h>

#include <OperatorJT.h>
#include <WavefunctionAnalyzer.h>

int main(int argc, char* argv[])
{
  qcx_utils::initialize(&argc, &argv);
  u3::initialize();
  u4::initialize();

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

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);
  basis.construct();

  int valenceShell = HO::valenceShell(Z + N),
      maxShell = valenceShell + Nmax;

  HO::Table_nljmjmt HO_labels(maxShell);
  std::string Hamil_name = "Hamiltonian with N2LO_opt_bare";
  JT::Operator Hamil(Hamil_name, N, Z, hw, Nmax, HO_labels);

  // read Hamiltonian matrix
  auto filename = Hamil.filename(basis);
  auto mat = qcx_utils::readBinary(filename);

  auto [evals, evecs] = qcx_utils::eigensolver_symmetric(mat, mat.rows());

  auto subspaces_u3 = sam::U3_wavefunction_analyzer(basis, evecs, 1);

  auto subspaces_u4 = sam::U4_wavefunction_analyzer(basis, evecs, 1);

  for (const auto& subspace : subspaces_u3)
  	std::cout << subspace << std::endl;

  for (const auto& subspace : subspaces_u4)
    std::cout << subspace << std::endl;

  u4::finalize();
  u3::finalize();
  qcx_utils::finalize();
}
