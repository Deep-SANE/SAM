#include <chrono>
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

  int Z, N, Nmax, MMT, JJ, valanceShell;

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

  U3xU4::ManyNucleonBasis basis(Z, N, Nmax, JJ);
  basis.construct();
  auto irreps = basis.irreps();
  auto labels = basis.decomposition();
  std::vector<double> analyticals;

  sam::Databases dbs;

  // Example 1: the L.L operator
  /*
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
  
  int hw = -1, maxShell = valanceShell + Nmax;
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
  */

  // Example 2: the L.S operator
  
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
  
  int hw = -1, maxShell = valanceShell + Nmax;
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
