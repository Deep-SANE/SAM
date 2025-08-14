#include <chrono>
#include <iostream>
#include <utility>

#include <Databases.h>
#include <U3xU4Irrep.h>
#include <U3xU4SingleShellTensors.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  std::vector<U3xU4::irrep> prod_ads, prod_ats;

  /* Example 1: n = 2
               (3,2,0)(1,1,0,0)  (5,4,0)(1,0,0,0)
      [[at x at]             x at]
                                 kappa=1,L=1,SS=1,TT=1=MMT,JJ=3=MMJ
  */
  /*
  U3xU4::irrep T2_0(3,2,0, 1,1,0,0), T2(5,4,0, 1,0,0,0);
  prod_ats.push_back(T2_0);
  prod_ats.push_back(T2);
  int rhomax = 1, etamax = 1;
  HO::Table_nljmjmt labels(2);
  U3xU4::Tadat_phys
  Tadat(2,0,3,prod_ads,prod_ats,T2,rhomax,etamax,2,1,1,1,3,3,labels);
  */

  /* Example 2: n = 5
              (7,3,0)(1,1,0,0)
      [ad x ad]
              kappa=2,L=5,SS=2,TT=0=MMT,JJ=10,MMJ=0
  */
  /*
  U3xU4::irrep T1(7,3,0, 1,1,0,0);
  prod_ads.push_back(T1);
  int rhomax = 1, etamax = 1;
  HO::Table_nljmjmt labels(5);
  U3xU4::Tadat_phys
  Tadat(5,2,0,prod_ads,prod_ats,T1,rhomax,etamax,10,2,0,0,10,0,labels);
  */

  /* Example 3: n = 1
              (0,0,0)(0,0,0,0)
      [ad x at]
              kappa=1,L=0,SS=0,TT=0=MMT,JJ=0=MMJ
  */
  /*
  U3xU4::irrep T(0,0,0, 0,0,0,0);
  int rhomax = 1, etamax = 1;
  HO::Table_nljmjmt labels(1);
  U3xU4::Tadat_phys
  Tadat(1,1,1,prod_ads,prod_ats,T,rhomax,etamax,0,0,0,0,0,0,labels);
  */

  /* Example 4: n = 2
             (3,1,0)   (3,3,0)     (3,2,0)   (4,1,0)(1,0,0,0)
  { [[ad x ad]     x ad] x [at x at]         }
             (1,1,0,0) (2,1,0,0)   (1,1,0,0) kappa=1,L=1,SS=1,TT=1=MMT,JJ=1=MMJ
  */

  U3xU4::irrep T1_0(3, 1, 0, 1, 1, 0, 0), T1(3, 3, 0, 2, 1, 0, 0),
      T2(3, 2, 0, 1, 1, 0, 0), T(4, 1, 0, 1, 0, 0, 0);
  prod_ads.push_back(T1_0);
  prod_ads.push_back(T1);
  prod_ats.push_back(T2);
  int rhomax = 1, etamax = 1;
  HO::Table_nljmjmt labels(2);
  U3xU4::Tadat_phys Tadat(2, 3, 2, prod_ads, prod_ats, T, rhomax, etamax, 2, 1,
                          1, 1, 1, 1, labels);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto [coeffsArr, adsArr, atsArr] = Tadat.expand(dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  Eigen::RowVectorXd norm(rhomax * etamax * Tadat.kappamax * Tadat.zetamax);
  norm.setZero();

  for (std::size_t i = 0; i < coeffsArr.size(); i++)
  {
    std::cout << coeffsArr[i] << "\t";
    norm = norm + coeffsArr[i].cwiseProduct(coeffsArr[i]);

    if (!adsArr.empty())
      for (auto ad : adsArr[i])
        std::cout << ad << "\t";

    if (!atsArr.empty())
      for (auto at : atsArr[i])
        std::cout << at << "\t";

    std::cout << std::endl;
  }

  std::cout << "Normalization: " << norm << std::endl;

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
