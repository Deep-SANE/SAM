// reads eigen pairs written by MatrixOperatorJT_truncated (utils::writeEigenPairs)
// and analyzes the U(3) and U(4) content of the lowest states; no matrix
// elements are computed and nothing is diagonalized.
//
// the model space below (nucleus, Nmax, JJ, hw, truncation, interaction and
// custom_dir) must match the run that wrote the eigen pairs: the eigenvector
// components are indexed by that basis.

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include <print.h>
#include <readwrite.h>

#include <sam.h>
#include <OperatorJT.h>
#include <WavefunctionAnalyzer.h>

int main(int argc, char* argv[])
{
  using utils::print;

  sam::initialize(&argc, &argv);

  int Z = 2, N = 2, JJ = 0, TT = 0, Nmax = 10, hw = 20;
  int n_states = 2; // number of lowest states to analyze

  U3xU4::hw_subspace
    hw0(0, {}, {}, {}, {TT}),
    hw2(2, {}, {}, {}, {TT}),
    hw4(4, {}, {}, {}, {TT}),
    hw6(6, {}, {}, {}, {TT}),
    hw8(8, {}, {}, {}, {TT}),
    hw10(10, {}, {}, {}, {TT});

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ, {hw0, hw2, hw4, hw6, hw8, hw10});
  basis.construct();

  std::string custom_dir = "Truncation_TT0";

  // std::string name = "Hamiltonian with N2LO_opt_bare";
  // std::string name = "Hamiltonian with N3LO_EM500_srg2.0";
  // std::string name = "Hamiltonian with N3LO_EMN500_bare";
  std::string name = "Hamiltonian with N4LO_EMN500_bare";
  // std::string name = "Magic_Be7_e3_hw16_E39";

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

#ifdef USE_MPI
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (mpi_rank == 0)
#endif
  {
    print("N={}, Z={}, Nmax={}, JJ={}, TT={}, hw={} with {}\n",
                                N, Z, Nmax, JJ, TT, hw, name);

    print("Basis dimension: {}\n", basis.dim());

    HO::Table_nljmjmt HO_labels(basis.maxShell());
    U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

    // the operator is used only to locate the eigen-pair file:
    JT::Operator op(name, N, Z, hw, Nmax, HO_labels, gelfand_labels);

    // read eigen pairs:
    auto wfns_file = op.wfns_file(basis, custom_dir);
    auto [evals, evecs] = utils::readEigenPairs(wfns_file);

    if (static_cast<size_t>(evecs.rows()) != basis.dim())
      throw std::runtime_error(
          "Sorry, the eigenvectors in " + wfns_file + " have " +
          std::to_string(evecs.rows()) + " components, but the basis has " +
          std::to_string(basis.dim()) + " states; the model space settings "
          "must match the run that wrote the file");

    print("Eigenvalues are\n");
    for (auto ev : evals) print("{:.6f}\n", ev);

    // analyze wfns:
    n_states = std::min(n_states, static_cast<int>(evecs.cols()));
    print("Analyze the first {} states\n", n_states);

    auto t0 = std::chrono::high_resolution_clock::now();

    auto subspaces_u3 = sam::U3_wavefunction_analyzer(basis, evecs, n_states);
    for (const auto& subspace : subspaces_u3)
      std::cout << subspace << std::endl;

    auto subspaces_u4 = sam::U4_wavefunction_analyzer(basis, evecs, n_states);
    for (const auto& subspace : subspaces_u4)
      std::cout << subspace << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    // output the time in seconds:
    std::chrono::duration<double> duration = t1 - t0;
    std::cout << "Elapsed time for analysis: "
              << duration.count() << " seconds or "
              << duration.count() / 60.0 << " minutes" << std::endl;
  }

  sam::finalize();
}
