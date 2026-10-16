#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <eigensolver.h>
#include <readwrite.h>

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
  // std::string name = "GamowTellerMinus";
  // std::string name = "GamowTellerPlus";

  ////////////// WITHIN THE SAME BASIS //////////////
  
  int Z = 3, N = 3, Nmax = 6, JJ = 6, hw = 20;

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);
  basis.construct();
  std::cout << "Basis dimension: " << basis.dim() << std::endl;

  HO::Table_nljmjmt HO_labels(basis.maxShell());
  U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

  JT::Operator Hamil(Hamil_name, N, Z, hw, Nmax, HO_labels, gelfand_labels),
               E2("E2", N, Z, hw, Nmax, HO_labels, gelfand_labels),
               M1("M1", N, Z, hw, Nmax, HO_labels, gelfand_labels),
               RMS_p("Radius_p", N, Z, hw, Nmax, HO_labels, gelfand_labels);

  // read Hamiltonian matrix
  auto filename = Hamil.filename(basis);
  auto mat = utils::readBinary(filename);

  auto [evals, evecs] = utils::eigensolver_symmetric(mat, 2);
  
  // Q2
  auto Q2 = E2.expectationValue(basis, {{evecs.col(0), evecs.col(0)}});

  std::cout << "Expectation values of Q2 are" << std::endl;
  for (int i = 0; i < Q2.size(); i++)
    std::cout << "State " << i << ": " << Q2[i] << std::endl;
  
  // BE2
  auto BE2 = E2.transitionStrength(basis, {{evecs.col(0), evecs.col(1)}});

  std::cout << "Transition strengths of B(E2) are" << std::endl;
  for (int i = 0; i < BE2.size(); i++)
    std::cout << "Channel " << i << ": " << BE2[i] << std::endl;
  
  // Magnetic moments
  auto mu = M1.expectationValue(basis, {{evecs.col(0), evecs.col(0)}});

  std::cout << "Expectation values of mu are" << std::endl;
  for (int i = 0; i < mu.size(); i++)
    std::cout << "State " << i << ": " << mu[i] << std::endl;

  // Point-proton radius
  auto rms = RMS_p.expectationValue(basis, {{evecs.col(0), evecs.col(0)}});

  std::cout << "Expectation values of rms_p are" << std::endl;
  for (int i = 0; i < rms.size(); i++)
    std::cout << "State " << i << ": " << rms[i] << std::endl;

  ////////////// BETWEEN TWO DIFFERENT BASES //////////////
  /*
  int Nmax = 4, hw = 20, 
      Zi = 2, Ni = 4, JJi = 0, Zf = 3, Nf = 3, JJf = 2;

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
  auto mat_i = utils::readBinary(filename_i);

  auto [evals_i, evecs_i] = utils::eigensolver_symmetric(mat_i, mat_i.rows());

  // read Hamiltonian matrix
  auto filename_f = Hamil_f.filename(bras);
  auto mat_f = utils::readBinary(filename_f);

  auto [evals_f, evecs_f] = utils::eigensolver_symmetric(mat_f, mat_f.rows());
  
  // transition
  
  auto strengths = op.transitionStrength(bras, kets, 
    {{evecs_f.col(0), evecs_i.col(0)}});

  std::cout << "Transition strengths of " << name << " are" << std::endl;
  for (int i = 0; i < strengths.size(); i++)
  {
    std::cout << "Channel " << i << ": " << strengths[i] << std::endl;
  }
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  */

  ////////////// DON'T EDIT ANYTHING HERE //////////////
  
  sam::finalize();
}
