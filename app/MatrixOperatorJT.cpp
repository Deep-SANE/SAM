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
      // utils::writeBinary(matrix_file, mat);

      auto wfns_file = op.wfns_file(basis);
      utils::writeEigenBinary(wfns_file, evals, evecs);
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

/*
    // L.L and L.S are analytical
    if (name == "L.L" || name == "L.S")
    {
      auto irreps = basis.irreps();
      auto labels = basis.decomposition();
      std::vector<double> analyticals;

      for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
      {
        auto alphamax = irreps[irrep_id].alphamax();

        for (const auto& [kmax, LL, zmax, SS, TT, offset] : labels[irrep_id])
        {
          for (size_t j = 0; j < alphamax * kmax * zmax; j++)
          {
            if (name == "L.L")
              analyticals.push_back(LL * (LL + 2.0) / 4.0);
            else
            {
              double jj = JJ * (JJ + 2.0) / 4.0, 
                     ll = LL * (LL + 2.0) / 4.0,
                     ss = SS * (SS + 2.0) / 4.0;
              analyticals.push_back((jj - ll - ss) / 2.0);
            }
          }
        }
      }
    
      // std::cout << "Total matrix representation is " << std::endl;
      // std::cout << mat << std::endl;
      for (int k = 0; k < mat.outerSize(); ++k) 
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) 
        {
          if (std::fabs(it.value()) > 1e-10)
          {
            std::cout << "(" << it.row() << "," << it.col() << "): "
                      << it.value() << "\t";
            if (it.row() != it.col())
              std::cout << "Off-diagonal\t";
            if (std::fabs(analyticals[it.row()] - it.value()) > 1e-10)
              std::cout << "Not analytical";
            std::cout << std::endl;
          }
        }
    }
*/
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
