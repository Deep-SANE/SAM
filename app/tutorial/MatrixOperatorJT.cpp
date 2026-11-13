#include <chrono>
#include <iomanip>
#include <iostream>

#include <eigensolver.h>
#include <readwrite.h>

#include <sam.h>

#include <OperatorJT.h>
#include <WavefunctionAnalyzer.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  /*
  int Z, N, Nmax, JJ, hw, write;

  std::cout << "Enter number of protons: ";
  std::cin >> Z;
  std::cout << "Enter number of neutrons: ";
  std::cin >> N;
  std::cout << "Enter Nmax: ";
  std::cin >> Nmax;
  std::cout << "Enter JJ: ";
  std::cin >> JJ;
  std::cout << "Enter hw (-1 for default): ";
  std::cin >> hw;
  std::cout << "Enter 1 to write matrix to binary file: ";
  std::cin >> write;
  */

  int Z = 3, N = 3, JJ = 2, Nmax = 6, hw = 20, write = 0, analyze = 0;

  // Example 1: LL
  // std::string name = "L.L";

  // Example 2: LS
  // std::string name = "L.S";

  // Example 3: relative kinetic energy
  // std::string name = "Kinetic";  

  // Example 4: CoM HO operator
  // std::string name = "CoMNumberOp";  

  // Example 6: Hamiltonian with N2LO_opt_bare
  std::string name = "Hamiltonian with N2LO_opt_bare";

  // Example 7: Hamiltonian with N3LO_EMN500_bare
  // std::string name = "Hamiltonian with N3LO_EMN500_bare";

  // Example 8: Hamiltonian with N4LO_EMN500_bare
  // std::string name = "Hamiltonian with N4LO_EMN500_bare";

  // Example 9: Hamiltonian with N3LO_EM500_srg2.0.0
  // std::string name = "Hamiltonian with N3LO_EM500_srg2.0";

  // Example 10: NO2B interactions
  // std::string name = "HO_Li7_e10_hw16_E342";

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);
  basis.construct();
  std::cout << "Basis dimension: " << basis.dim() << std::endl;

  HO::Table_nljmjmt HO_labels(basis.maxShell());
  U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

  JT::Operator op(name, N, Z, hw, Nmax, HO_labels, gelfand_labels);
  op.is();

  // Compute matrix
  auto t0 = std::chrono::high_resolution_clock::now();

  auto mat = op.matrix_CSR(basis);

  auto t1 = std::chrono::high_resolution_clock::now();

  // Diagonalize
  auto t2 = std::chrono::high_resolution_clock::now();

  auto [evals, evecs] = utils::eigensolver_symmetric_spectra(mat, 10, 1e-6);

  auto t3 = std::chrono::high_resolution_clock::now();

#ifdef USE_MPI
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (mpi_rank == 0)
#endif
  {
    std::cout << "Eigenvalues are" << std::endl;
    for (auto ev : evals)
    {
      if (std::fabs(ev) < 1e-10) std::cout << 0 << std::endl;
      else std::cout << std::setprecision(8) << ev << std::endl;
    }

    if (write == 1)
    {
      // auto matrix_file = op.matrix_file(basis);
      // utils::writeCSRMatrix(matrix_file, mat);

      auto wfns_file = op.wfns_file(basis);
      utils::writeEigenPairs(wfns_file, evals, evecs);
    }

    // Analyze wfns
    if (analyze == 1)
    {
      std::cout << "Analyze the first two states" << std::endl;

      auto subspaces_u3 = sam::U3_wavefunction_analyzer(basis, evecs, 2);
      for (const auto& subspace : subspaces_u3)
        std::cout << subspace << std::endl;

      auto subspaces_u4 = sam::U4_wavefunction_analyzer(basis, evecs, 2);
      for (const auto& subspace : subspaces_u4)
        std::cout << subspace << std::endl;
    }

    // Calculate the duration
    std::chrono::duration<double> duration1 = t1 - t0, duration2 = t3 - t2;

    // Output the time in seconds
    std::cout << "Elapsed time for matrix elements: "
              << duration1.count() << " seconds or "
              << duration1.count() / 60.0 << " minutes" << std::endl;

    std::cout << "Elapsed time for diagonalizing: "
              << duration2.count() << " seconds or "
              << duration2.count() / 60.0 << " minutes" << std::endl;
  }

  sam::finalize();
}
