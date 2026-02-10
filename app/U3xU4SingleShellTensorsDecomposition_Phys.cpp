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
  U3xU4::irrep T1(0,0,0, 0,0,0,0), T2_0(3,2,0, 1,1,0,0), T2(5,4,0, 1,0,0,0);
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

  auto [coeffsArr, adsArr, atsArr] = Tadat.expand();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  Eigen::RowVectorXd norm(rhomax * etamax * Tadat.kappamax() * Tadat.zetamax());
  norm.setZero();

  bool print;
  std::cout << "Finish expanding with " << coeffsArr.size() << " terms."
            << " Do you want to print the results? (1 yes, 0 no) ";
  std::cin >> print;

  for (size_t i = 0; i < coeffsArr.size(); i++)
  {
    if (print)
    {
      std::cout << coeffsArr[i] << "\t";
    
      if (!adsArr.empty())
        for (auto ad : adsArr[i]) std::cout << ad << "\t";

      if (!atsArr.empty())
        for (auto at : atsArr[i]) std::cout << at << "\t";

      std::cout << std::endl;
    }
    norm = norm + coeffsArr[i].cwiseProduct(coeffsArr[i]);
  }

  std::cout << "Normalization: " << norm << std::endl;

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
