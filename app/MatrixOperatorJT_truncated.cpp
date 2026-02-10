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

  int Z = 3, N = 3, JJ = 6, Nmax = 6, hw = 20, write = 1;

  u4::irreps irrs = {{1,1,0,0}, {2,0,0,0}, {3,2,1,0}};

  U3xU4::hw_subspace 
    hw0(0, {}, irrs, {}, {}), hw2(2, {}, irrs, {}, {}),
    hw4(4, {}, irrs, {}, {}), hw6(6, {}, irrs, {}, {});

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ, {hw6});
  basis.construct();

  std::string name = "Hamiltonian with N2LO_opt_srg2.0";
  print("N={}, Z={}, Nmax={}, JJ={}, hw={} with {}\n", N, Z, Nmax, JJ, hw, name);

  print("Basis dimension: {}\n", basis.dim());
  print("Keep only u4::irreps: {} in N=6\n", irrs);

  HO::Table_nljmjmt HO_labels(basis.maxShell());
  U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

  JT::Operator op(name, N, Z, hw, Nmax, HO_labels, gelfand_labels);
  // op.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto mat = op.matrix(basis);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  // Diagonalize
  auto [evals, evecs] = qcx_utils::eigensolver_symmetric(mat, 10);
  std::cout << "Eigenvalues are" << std::endl;
  for (auto ev : evals)
  {
    if (std::fabs(ev) < 1e-10)
      std::cout << 0 << std::endl;
    else
      std::cout << std::setprecision(8) << ev << std::endl;
  }

  // Analyze w.fn. to double check
  auto subspaces_u4 = sam::U4_wavefunction_analyzer(basis, evecs, 1);
  for (const auto& subspace : subspaces_u4)
    std::cout << subspace << std::endl;

  if (write == 1)
  {
    auto filename = op.filename(basis);
    filename = filename.substr(0, filename.size() - 4);
    filename += "Truncation_09Feb26_4/hw20";
    qcx_utils::writeBinary(filename, mat);
  }

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
