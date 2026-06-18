#include <chrono>
#include <iostream>

#include <eigensolver.h>
#include <qcx_print.h>
#include <qcx_readwrite.h>

#include <sam.h>
#include <OperatorJT.h>
#include <WavefunctionAnalyzer.h>

int main(int argc, char* argv[])
{
  using qcx_utils::print;

  sam::initialize(&argc, &argv);

  int Z = 4, N = 3, JJ = 1, TT = 1, Nmax = 8, hw = 16, write = 1;

  u4::irreps irrs4 = {{1,1,1,0}, {2,1,0,0}, {3,3,1,0}};
  u3::irreps irrs3_4 = {{7,0,0}, {6,1,0}, {5,2,0}, {4,3,0}, {3,1,0}},
             irrs3_6 = {{9,0,0}, {8,1,0}, {7,2,0}, {6,3,0}, {5,4,0}},
             irrs3_8 = {{11,0,0}, {10,1,0}, {9,2,0}, {8,3,0}, {7,4,0}};

  U3xU4::hw_subspace
    hw0(0, {}, {}, {}, {TT}),
    hw2(2, {}, {}, {}, {TT}),
    hw4(4, {irrs3_4}, {irrs4}, {}, {TT}),
    hw6(6, {irrs3_6}, {irrs4}, {}, {TT}),
    hw8(8, {irrs3_8}, {irrs4}, {}, {TT});

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ, {hw0, hw2, hw4, hw6, hw8});
  basis.construct();

  std::string custom_dir = "Truncation_TT1_2to8_25May26";

  // std::string name = "Hamiltonian with N2LO_opt_bare";
  // std::string name = "Hamiltonian with N3LO_EM500_srg2.0";
  std::string name = "HO_Be7_e10_hw16_E342";

#ifdef USE_MPI
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (mpi_rank == 0)
#endif
  {
    print("N={}, Z={}, Nmax={}, JJ={}, TT={}, hw={} with {}\n", 
                                N, Z, Nmax, JJ, TT, hw, name);

    print("Basis dimension: {}\n", basis.dim());
    print("Keep only u4::irreps (>0.1%): {} in N=4,6,8\n", irrs4);
    print("Keep only u3::irreps (>0.1%): {} in N=4\n", irrs3_4);
    print("Keep only u3::irreps (>0.1%): {} in N=6\n", irrs3_6);
    print("Keep only u3::irreps (>0.1%): {} in N=8\n", irrs3_8);
  }

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

  HO::Table_nljmjmt HO_labels(basis.maxShell());
  U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

  JT::Operator op(name, N, Z, hw, Nmax, HO_labels, gelfand_labels);
  op.is();

  // Compute matrix
  auto t0 = std::chrono::high_resolution_clock::now();

  auto mat = op.matrix_CSR(basis);

  auto t1 = std::chrono::high_resolution_clock::now();

/*
  if (write == 1)
  {
    auto matrix_file = op.matrix_file(basis, custom_dir);
    qcx_utils::writeBinary(matrix_file, mat);
  }
*/
  // Diagonalize
  auto t2 = std::chrono::high_resolution_clock::now();

  auto [evals, evecs] = qcx_utils::eigensolver_symmetric_spectra(mat, 5, 1e-10, 100000);

  auto t3 = std::chrono::high_resolution_clock::now();

#ifdef USE_MPI
  if (mpi_rank == 0)
#endif
  {
    print("Eigenvalues are\n");
    for (auto ev : evals) print("{:.6f}\n", ev);

    if (write == 1)
    {
      // auto wfns_file = op.wfns_file(basis);
      auto wfns_file = op.wfns_file(basis, custom_dir);
      qcx_utils::writeEigenBinary(wfns_file, evals, evecs);
    }

    // Analyze wfns
    print("Analyze the first two states\n");

    auto subspaces_u3 = sam::U3_wavefunction_analyzer(basis, evecs, 2);
    for (const auto& subspace : subspaces_u3)
      std::cout << subspace << std::endl;

    auto subspaces_u4 = sam::U4_wavefunction_analyzer(basis, evecs, 2);
    for (const auto& subspace : subspaces_u4)
      std::cout << subspace << std::endl;

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
