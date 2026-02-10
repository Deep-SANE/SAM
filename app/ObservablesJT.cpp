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
  std::string name = "M1";

  // Example 3: instrinsic radius squared
  // std::string name = "Radius";

  // Example 11: Fermi for beta decay
  // std::string name = "FermiMinus";
  // std::string name = "FermiPlus";

  // Example 12: Gamow-Teller for beta decay
  // std::string name = "GamowTellerMinus";
  // std::string name = "GamowTellerPlus";

  ////////////// WITHIN THE SAME BASIS //////////////
  /*
  int Z = 3, N = 3, Nmax = 6, JJ = 2, hw = 20;

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);
  basis.construct();
  std::cout << "Basis dimension: " << basis.dim() << std::endl;

  HO::Table_nljmjmt HO_labels(basis.maxShell());
  U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

  JT::Operator Hamil(Hamil_name, N, Z, hw, Nmax, HO_labels, gelfand_labels),
               op(name, N, Z, hw, Nmax, HO_labels, gelfand_labels);
  op.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  // read Hamiltonian matrix
  auto filename = Hamil.filename(basis);
  auto mat = qcx_utils::readBinary(filename);

  auto [evals, evecs] = qcx_utils::eigensolver_symmetric(mat, 3);
  
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
  
  int Nmax = 6, hw = 20, 
      Zi = 3, Ni = 3, JJi = 4, Zf = 3, Nf = 3, JJf = 2;

  U3xU4::ManyNucleonBasis kets(Ni, Zi, Nmax, JJi);
  kets.construct();
  std::cout << "Ket basis dimension: " << kets.dim() << std::endl;

  U3xU4::ManyNucleonBasis bras(Nf, Zf, Nmax, JJf);
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

  // read Hamiltonian matrix
  auto filename_i = Hamil_i.filename(kets);
  auto mat_i = qcx_utils::readBinary(filename_i);

  auto [evals_i, evecs_i] = qcx_utils::eigensolver_symmetric(mat_i, 2);
  std::cout << "Initial energies: ";
  for (const auto& val : evals_i) std::cout << val << "\t";
  std::cout << std::endl;

  // read Hamiltonian matrix
  auto filename_f = Hamil_f.filename(bras);
  auto mat_f = qcx_utils::readBinary(filename_f);

  auto [evals_f, evecs_f] = qcx_utils::eigensolver_symmetric(mat_f, 2);
  std::cout << "Final energies: ";
  for (const auto& val : evals_f) std::cout << val << "\t";
  std::cout << std::endl;
  
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
