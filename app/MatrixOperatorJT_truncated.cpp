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

  int Z = 3, N = 3, JJ = 2, TT = 0, Nmax = 6, hw = 20, write = 1, solve=0;

  u4::irreps irrs = {{1,1,0,0}, {2,0,0,0}, {3,2,1,0}};

  U3xU4::hw_subspace 
    hw0(0, {}, {}, {}, {TT}),
    hw2(2, {}, {}, {}, {TT}),
    hw4(4, {}, {irrs}, {}, {TT}),
    hw6(6, {}, {irrs}, {}, {TT});

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ, {hw0, hw2, hw4, hw6});
  basis.construct();

  std::string name = "Hamiltonian with N2LO_opt_srg2.0";
  print("N={}, Z={}, Nmax={}, JJ={}, TT={}, hw={} with {}\n", 
                                N, Z, Nmax, JJ, TT, hw, name);

  print("Basis dimension: {}\n", basis.dim());
  print("Keep only u4::irreps: {} in N=4,6\n", irrs);

  HO::Table_nljmjmt HO_labels(basis.maxShell());
  U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

  JT::Operator op(name, N, Z, hw, Nmax, HO_labels, gelfand_labels);
  op.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();
  print("Start computing matrix\n");
  auto mat = op.matrix(basis);
  print("Finish computing matrix\n");
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  // Diagonalize
  auto [evals, evecs] = qcx_utils::eigensolver_symmetric(mat, 10);
  print("Eigenvalues are\n");
  for (auto ev : evals) print("{}\n", ev);

  if (write == 1)
  {
    std::string custom_dir = "Truncation_TT0_24Feb26_2";

    auto matrix_file = op.matrix_file(basis, custom_dir);
    qcx_utils::writeBinary(matrix_file, mat);

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
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
