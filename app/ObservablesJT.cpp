#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <eigensolver.h>
#include <qcx_readwrite.h>
#include <qcx_utils.h>

#include <Databases.h>
#include <OperatorJT.h>

int main(int argc, char* argv[])
{
  qcx_utils::initialize(&argc, &argv);

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
  int Z = 3, N = 3, Nmax = 4, JJ = 4, hw = 20;

  U3xU4::ManyNucleonBasis basis(Z, N, Nmax, JJ);
  basis.construct();
  std::cout << "Basis dimension: " << basis.dim() << std::endl;

  int valenceShell = HO::valenceShell(Z + N),
      maxShell = valenceShell + Nmax;

  HO::Table_nljmjmt HO_labels(maxShell);

  JT::Operator Hamil(Hamil_name, N, Z, hw, Nmax, HO_labels),
               op(name, N, Z, hw, Nmax, HO_labels);
  op.is();

  sam::Databases dbs;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  // read Hamiltonian matrix
  auto filename = Hamil.filename(basis);
  auto mat = qcx_utils::readBinary(filename);

  auto [evals, evecs] = qcx_utils::eigensolver_symmetric(mat, mat.rows());
  
  // moments
  
  auto vals = op.expectationValue(basis, 
    {{evecs.col(0), evecs.col(0)}, {evecs.col(1), evecs.col(1)}},
    dbs);

  std::cout << "Expectation values of " << name << " are" << std::endl;
  for (int i = 0; i < vals.size(); i++)
  {
    std::cout << "State " << i << ": " << vals[i] << std::endl;
  }
  */

  // transition
  /*
  auto strengths = op.transitionStrength(basis, 
    {{evecs.col(0), evecs.col(1)}}, 
    dbs);

  std::cout << "Transition strengths of " << name << " are" << std::endl;
  for (int i = 0; i < strengths.size(); i++)
  {
    std::cout << "State " << i+1 << " to " << i << ": " 
              << strengths[i] << std::endl;
  }
  */

  // End time measurement
  // auto t1 = std::chrono::high_resolution_clock::now();
  

  ////////////// BETWEEN TWO DIFFERENT BASES //////////////
  
  int Nmax = 4, hw = 20, 
      Zi = 2, Ni = 4, JJi = 0, Zf = 3, Nf = 3, JJf = 2;

  U3xU4::ManyNucleonBasis kets(Zi, Ni, Nmax, JJi);
  kets.construct();
  std::cout << "Ket basis dimension: " << kets.dim() << std::endl;

  U3xU4::ManyNucleonBasis bras(Zf, Nf, Nmax, JJf);
  bras.construct();
  std::cout << "Bra basis dimension: " << bras.dim() << std::endl;

  int valenceShell_i = HO::valenceShell(Zi + Ni),
      valenceShell_f = HO::valenceShell(Zf + Nf),
      maxShell = std::max(valenceShell_i, valenceShell_f) + Nmax;

  HO::Table_nljmjmt HO_labels(maxShell);

  JT::Operator Hamil_i(Hamil_name, Ni, Zi, hw, Nmax, HO_labels),
               Hamil_f(Hamil_name, Nf, Zf, hw, Nmax, HO_labels),
               op(name, Ni, Zi, hw, Nmax, HO_labels);
  op.is();

  sam::Databases dbs;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  // read Hamiltonian matrix
  auto filename_i = Hamil_i.filename(kets);
  auto mat_i = qcx_utils::readBinary(filename_i);

  auto [evals_i, evecs_i] = qcx_utils::eigensolver_symmetric(mat_i, mat_i.rows());

  // read Hamiltonian matrix
  auto filename_f = Hamil_f.filename(bras);
  auto mat_f = qcx_utils::readBinary(filename_f);

  auto [evals_f, evecs_f] = qcx_utils::eigensolver_symmetric(mat_f, mat_f.rows());
  
  // transition
  
  auto strengths = op.transitionStrength(bras, kets, 
    {{evecs_f.col(0), evecs_i.col(0)}}, 
    dbs);

  std::cout << "Transition strengths of " << name << " are" << std::endl;
  for (int i = 0; i < strengths.size(); i++)
  {
    std::cout << "Channel " << i << ": " << strengths[i] << std::endl;
  }
  
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  

  ////////////// DON'T EDIT ANYTHING HERE //////////////
  
  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;


  qcx_utils::finalize();
}
