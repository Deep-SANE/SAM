#include <chrono>
#include <iomanip>
#include <iostream>

#include <Databases.h>
#include <OneBodyOperatorJT.h>
#include <TwoBodyOperatorJT.h>
#include <U3xU4ManyNucleonBasis.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  int Z, N, Nmax, MMT, JJ, valanceShell, hw;

  std::cout << "Enter number of protons: ";
  std::cin >> Z;
  std::cout << "Enter number of neutrons: ";
  std::cin >> N;
  std::cout << "Enter valance shell: ";
  std::cin >> valanceShell;
  std::cout << "Enter Nmax: ";
  std::cin >> Nmax;
  std::cout << "Enter JJ: ";
  std::cin >> JJ;
  std::cout << "Enter hw: ";
  std::cin >> hw;

  U3xU4::ManyNucleonBasis basis(Z, N, Nmax, JJ);
  basis.construct();
  std::cout << "Basis dimension: " << basis.dim() << std::endl;

  sam::Databases dbs;

  // Example 1: the L.L operator
  /*
  auto irreps = basis.irreps();
  auto labels = basis.decomposition();
  std::vector<double> analyticals;

  size_t i = 0;
  for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
  {
    for (size_t labels_id = 0; labels_id < labels[irrep_id].size(); labels_id++)
    {
      for (size_t j = 0; 
           j < labels[irrep_id][labels_id][0]*labels[irrep_id][labels_id][2]; j++)
      {
        auto l = labels[irrep_id][labels_id][1] / 2.0; 
        analyticals.push_back(l * (l + 1));
        i++;
      }
    }
  }
  
  int maxShell = valanceShell + Nmax;
  HO::Table_nljmjmt HO_labels(maxShell);
  JT::OneBodyOperator LL1("L.L",hw,maxShell);  
  JT::TwoBodyOperator LL2("L.L", HO_labels, hw, Nmax, valanceShell, maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  LL1.is();
  auto mat1 = LL1.matrix(basis, dbs);
  //std::cout << "Matrix representation is " << std::endl;
  //std::cout << mat1 << std::endl;

  LL2.is();
  auto mat2 = LL2.matrix(basis, dbs);
  //std::cout << "Matrix representation is " << std::endl;
  //std::cout << mat2 << std::endl;

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  Eigen::SparseMatrix<double> mat = mat1 + mat2;
  std::cout << "Total matrix representation is " << std::endl;
  //std::cout << mat << std::endl;
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
  */

  // Example 2: the L.S operator
  /*
  auto irreps = basis.irreps();
  auto labels = basis.decomposition();
  std::vector<double> analyticals;

  size_t i = 0;
  for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
  {
    for (size_t labels_id = 0; labels_id < labels[irrep_id].size(); labels_id++)
    {
      for (size_t j = 0; 
           j < labels[irrep_id][labels_id][0]*labels[irrep_id][labels_id][2]; j++)
      {
        auto L = labels[irrep_id][labels_id][1] / 2.0; 
        auto S = labels[irrep_id][labels_id][3] / 2.0; 
        auto J = JJ / 2.0;
        analyticals.push_back((J * (J + 1.0) - L * (L + 1.0) - S * (S + 1.0)) / 2.0);
        i++;
      }
    }
  }
  
  int maxShell = valanceShell + Nmax;
  HO::Table_nljmjmt HO_labels(maxShell);
  JT::OneBodyOperator LS1("L.S",hw,maxShell);  
  JT::TwoBodyOperator LS2("L.S", HO_labels, hw, Nmax, valanceShell, maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  LS1.is();
  auto mat1 = LS1.matrix(basis, dbs);
  //std::cout << "Matrix representation is " << std::endl;
  //std::cout << mat1 << std::endl;

  LS2.is();
  auto mat2 = LS2.matrix(basis, dbs);
  //std::cout << "Matrix representation is " << std::endl;
  //std::cout << mat2 << std::endl;

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  
  Eigen::SparseMatrix<double> mat = mat1 + mat2;
  std::cout << "Total matrix representation is " << std::endl;
  //std::cout << mat << std::endl;
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
  */

  // Example 3: the CoM number operator
  
  int maxShell = valanceShell + Nmax;
  HO::Table_nljmjmt HO_labels(maxShell);
  JT::OneBodyOperator CoM1("CoMNumberOp",hw,maxShell);  
  JT::TwoBodyOperator CoM2("CoMNumberOp", HO_labels, hw, Nmax, valanceShell, maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  CoM1.is();
  auto mat1 = CoM1.matrix(basis, dbs);
  //std::cout << "Matrix representation is " << std::endl;
  //std::cout << mat1 << std::endl;

  CoM2.is();
  auto mat2 = CoM2.matrix(basis, dbs);
  //std::cout << "Matrix representation is " << std::endl;
  //std::cout << mat2 << std::endl;

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
  
  Eigen::SparseMatrix<double> mat = mat1 + mat2;
  std::cout << "Total matrix representation is " << std::endl;
  //std::cout << mat << std::endl;
  for (int k = 0; k < mat.outerSize(); ++k) 
    for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) 
    {
      if (std::fabs(it.value()) > 1e-10)
      {
        std::cout << it.row() << "\t" << it.col() << "\t"
                  << std::setprecision(10) << it.value() << std::endl;
      }
    }
  

  // Example 4: 2-body T_rel / (2hw) + 1000 * the CoM number operator
  /*
  int maxShell = valanceShell + Nmax;
  HO::Table_nljmjmt HO_labels(maxShell);
  JT::OneBodyOperator CoM1("CoMNumberOp",hw,maxShell);  
  JT::TwoBodyOperator CoM2("CoMNumberOp",HO_labels,hw,Nmax,valanceShell,maxShell);
  JT::OneBodyOperator KE1("Kinetic",hw,maxShell);  
  JT::TwoBodyOperator KE2("Kinetic",HO_labels,hw,Nmax,valanceShell,maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  CoM1.is();
  auto com1 = CoM1.matrix(basis, dbs);

  CoM2.is();
  auto com2 = CoM2.matrix(basis, dbs);

  KE1.is();
  auto ke1 = KE1.matrix(basis, dbs);

  KE2.is();
  auto ke2 = KE2.matrix(basis, dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  Eigen::SparseMatrix<double> H = (1000 * (com1 + com2) + (Z + N - 1) * ke1 + ke2) / (Z + N);
  std::cout << "Total Hamiltonian matrix is " << std::endl;
  for (int k = 0; k < H.outerSize(); ++k) 
    for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it) 
    {
      if (std::fabs(it.value()) > 1e-11)
      {
        std::cout << it.row() << "\t" << it.col() << "\t"
                  << std::setprecision(10) << it.value() << std::endl;
      }
    }
  */
  
  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;


#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}
