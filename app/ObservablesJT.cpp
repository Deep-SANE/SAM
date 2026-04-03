#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <eigensolver.h>
#include <qcx_readwrite.h>

#include <sam.h>

#include <OperatorJT.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  // Specify Hamiltonian
  std::string Hamil_name = "Hamiltonian with N2LO_opt_bare";

  // Specify operator name
  // Example 1: E2
  // std::string name = "E2";

  // Example 2: M1
  // std::string name = "M1";

  // Example 3: instrinsic radius squared
  // std::string name = "Radius";

  // Example 11: Fermi for beta decay
  // std::string name = "FermiMinus";
  // std::string name = "FermiPlus";

  // Example 12: Gamow-Teller for beta decay
  std::string name = "GamowTellerMinus";
  // std::string name = "GamowTellerPlus";

  ////////////// WITHIN THE SAME BASIS //////////////
  /*
  int Z = 3, N = 3, Nmax = 8, JJ = 2, TT = 0, hw = 20;

  u4::irreps irrs = {{1,1,0,0}, {2,0,0,0}, {3,2,1,0}};

  U3xU4::hw_subspace
    hw0(0, {}, {}, {}, {TT}),
    hw2(2, {}, {}, {}, {TT}),
    hw4(4, {}, {}, {}, {TT}),
    hw6(6, {}, {}, {}, {TT}),
    hw8(8, {}, {irrs}, {}, {TT});

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ, {hw0, hw2, hw4, hw6, hw8});
  basis.construct();

  std::string custom_dir = "Truncation_TT0_6to8_31Mar26";

  HO::Table_nljmjmt HO_labels(basis.maxShell());
  U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

  JT::Operator Hamil(Hamil_name, N, Z, hw, Nmax, HO_labels, gelfand_labels),
               op(name, N, Z, hw, Nmax, HO_labels, gelfand_labels);
  op.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  // read wave functions
  auto wfns_file = Hamil.wfns_file(basis, custom_dir);
  auto [evals, evecs] = qcx_utils::readEigenBinary(wfns_file);
  
  // moments
  
  auto vals = op.expectationValue(basis, 
    {{evecs.col(0), evecs.col(0)}, {evecs.col(1), evecs.col(1)}});

  std::cout << "Expectation values of " << name << " are" << std::endl;
  for (int i = 0; i < vals.size(); i++)
    std::cout << "State " << i << ": " << vals[i] << std::endl;
  

  // transition
  
  auto strengths = op.transitionStrength(basis, 
    {{evecs.col(0), evecs.col(1)}});

  std::cout << "Transition strengths of " << name << " are" << std::endl;
  for (int i = 0; i < strengths.size(); i++)
    std::cout << "Channel " << i << ": " << strengths[i] << std::endl;
  

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  ////////////// BETWEEN TWO DIFFERENT BASES //////////////

  int Zi = 2, Ni = 4, Nmax = 8, JJi = 0, TTi = 2, hw = 20;
  std::string dir_i = "Truncation_TT2_6to8_31Mar26";

  u4::irreps irrs = {{1,1,0,0}, {2,0,0,0}, {3,2,1,0}};

  U3xU4::hw_subspace
    hw0i(0, {}, {}, {}, {TTi}),
    hw2i(2, {}, {}, {}, {TTi}),
    hw4i(4, {}, {}, {}, {TTi}),
    hw6i(6, {}, {}, {}, {TTi}),
    hw8i(8, {}, {irrs}, {}, {TTi});

  U3xU4::ManyNucleonBasis kets(Ni, Zi, Nmax, JJi, {hw0i, hw2i, hw4i, hw6i, hw8i});
  kets.construct();
  std::cout << "Ket basis dimension: " << kets.dim() << std::endl;
  
  int Zf = 3, Nf = 3, JJf = 2, TTf = 0;
  std::string dir_f = "Truncation_TT0_6to8_31Mar26";

  U3xU4::hw_subspace
    hw0f(0, {}, {}, {}, {TTf}),
    hw2f(2, {}, {}, {}, {TTf}),
    hw4f(4, {}, {}, {}, {TTf}),
    hw6f(6, {}, {}, {}, {TTf}),
    hw8f(8, {}, {irrs}, {}, {TTf});

  U3xU4::ManyNucleonBasis bras(Nf, Zf, Nmax, JJf, {hw0f, hw2f, hw4f, hw6f, hw8f});
  bras.construct();
  std::cout << "Bra basis dimension: " << bras.dim() << std::endl;

  int maxShell = std::max(kets.maxShell(), bras.maxShell());
  HO::Table_nljmjmt HO_labels(maxShell);
  U3xU4::Table_gelfand gelfand_labels(maxShell);

  JT::Operator Hamil_i(Hamil_name, Ni, Zi, hw, Nmax, HO_labels, gelfand_labels),
               Hamil_f(Hamil_name, Nf, Zf, hw, Nmax, HO_labels, gelfand_labels),
               op(name, Ni, Zi, hw, Nmax, HO_labels, gelfand_labels);
  op.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  // read initial wave functions
  auto wfns_file_i = Hamil_i.wfns_file(kets, dir_i);
  auto [evals_i, evecs_i] = qcx_utils::readEigenBinary(wfns_file_i);

  // read final wave functions
  auto wfns_file_f = Hamil_f.wfns_file(bras, dir_f);
  auto [evals_f, evecs_f] = qcx_utils::readEigenBinary(wfns_file_f);
  
  // transition
  
  auto strengths = op.transitionStrength(bras, kets, 
    {{evecs_f.col(0), evecs_i.col(0)}});

  std::cout << "Transition strengths of " << name << " are" << std::endl;
  for (int i = 0; i < strengths.size(); i++)
    std::cout << "Channel " << i << ": " << strengths[i] << std::endl;
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  

  ////////////// DON'T EDIT ANYTHING HERE //////////////
  
  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;


  sam::finalize();
}
