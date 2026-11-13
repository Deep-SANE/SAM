// U3xU4SingleShellTensorsDecomposition_Phys: a single-shell U(3)xU(4)
// tensor in the physical basis.
//
// In one HO shell n, creation (ad) and annihilation (at) operators are
// coupled step by step (the Omega irreps) to a tensor with definite U(3)xU(4)
// symmetry T, and projected to orbital rank LL = 2L, spin SS = 2S, isospin
// TT = 2T and their projections. expand() writes it as a sum of products of
// single-nucleon operators in the physical HO basis |n l j m m_t>; the
// program prints the number of terms and the normalization (the squared
// coefficients sum to 1 for each multiplicity).
//
// One example is active; the others are commented out. To try another,
// copy this file into app/work/ and swap the comments:
//   cp app/tutorial/U3xU4SingleShellTensorsDecomposition_Phys.cpp app/work/
//   make -j 8 apps && mpirun -np 1 bin/work/U3xU4SingleShellTensorsDecomposition_Phys
// The same examples, with their numbers checked, are in
// test/test_singleshelltensors_decomposition_phys.cpp.

#include <chrono>
#include <iostream>
#include <utility>

#include <U3xU4irrep.h>
#include <U3xU4SingleShellTensors.h>

#include <u3lib.h>
#include <u4lib.h>

#include <sam.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  int n = 0, ads = 0, ats = 0, 
      LL = 0, SS = 0, TT = 0, MMT = 0, JJ = 0, MMJ = 0;
  
  U3xU4::irreps Omega_ads, Omega_ats;

  /* Example 1: n = 2
               (3,2,0)(1,1,0,0)  (5,4,0)(1,0,0,0)
      [[at x at]             x at]
                                 kappa=1,L=1,SS=1,TT=1=MMT,JJ=3=MMJ
  */
  /*
  n = 2;
  ats = 3;
  U3xU4::irrep T1(0,0,0, 0,0,0,0), T2_0(3,2,0, 1,1,0,0), 
               T2(5,4,0, 1,0,0,0), T(5,4,0, 1,0,0,0);
  Omega_ats = {T2_0, T2};

  LL = 2;
  SS = 1;
  TT = 1;
  MMT = 1;
  JJ = 3;
  MMJ = 3;
  */

  /* Example 2: n = 5
              (7,3,0)(1,1,0,0)
      [ad x ad]
              kappa=2,L=5,SS=2,TT=0=MMT,JJ=10,MMJ=0
  */
  /*
  n = 5;
  ads = 2;
  U3xU4::irrep T1(7,3,0, 1,1,0,0), T2(0,0,0, 0,0,0,0), T(7,3,0, 1,1,0,0);
  Omega_ads = {T1};

  LL = 10;
  SS = 2;
  JJ = 10;
  */

  /* Example 3: n = 1
              (0,0,0)(0,0,0,0)
      [ad x at]
              kappa=1,L=0,SS=0,TT=0=MMT,JJ=0=MMJ
  */
  /*
  n = 1;
  ads = 1;
  ats = 1;
  U3xU4::irrep T1(1,0,0, 1,0,0,0), T2(1,1,0, 1,1,1,0), T(0,0,0, 0,0,0,0);
  */

  /* Example 4: n = 2
             (3,1,0)   (3,3,0)     (3,2,0)   (4,1,0)(1,0,0,0)
  { [[ad x ad]     x ad] x [at x at]         }
             (1,1,0,0) (2,1,0,0)   (1,1,0,0) kappa=1,L=1,SS=1,TT=1=MMT,JJ=1=MMJ
  */
  /*
  n = 2;
  ads = 3;
  ats = 2;
  U3xU4::irrep T1_0(3,1,0, 1,1,0,0), T1(3,3,0, 2,1,0,0),
               T2(3,2,0, 1,1,0,0), T(4,1,0, 1,0,0,0);
  Omega_ads = {T1_0, T1};
  Omega_ats = {T2};

  LL = 2; 
  SS = 1;
  TT = 1;
  MMT = 1;
  JJ = 1;
  MMJ = 1;
  */

  /* Example 5: n = 0
             (0,0,0)          (0,0,0)   (0,0,0)(0,0,0,0)
  { [[ad x ad]      x [at x at]         }
             (1,1,0,0)        (1,1,0,0) kappa=1,L=0,SS=0,TT=0=MMT,JJ=0=MMJ
  */
  
  n = 0;
  ads = 2;
  ats = 2;
  U3xU4::irrep T1(0,0,0, 1,1,0,0), T2(0,0,0, 1,1,0,0), T(0,0,0, 0,0,0,0);
  Omega_ads = {T1};
  Omega_ats = {T2};
  

  ////////////////// DO NOT EDIT ANYTHING BELOW ///////////////////

  // Multiplicities of the final tensor
  auto [rhomax, etamax] = U3xU4::RhoEta(T1, T2, T);

  HO::Table_nljmjmt labels(n);
  U3xU4::SingleShellTensor_phys Tadat(n, ads, ats, Omega_ads, Omega_ats, 
    T, rhomax, etamax, LL, SS, TT, MMT, JJ, MMJ, labels);
  
  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto exp = Tadat.expand();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  bool print;
  std::cout << "Finish expanding with " << exp.rows() << " terms."
            << " Do you want to print the results? (1 yes, 0 no) ";
  std::cin >> print;

  if (print)
    std::cout << exp;

  std::vector<double> norm = exp.coeffs.norm();
  std::cout << "Normalization:";
  for (double v : norm) std::cout << " " << v;
  std::cout << std::endl;

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
