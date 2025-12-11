#include <chrono>
#include <iostream>
#include <utility>

#include <U3basis.h>
#include <U4basis.h>

#include <Databases.h>
#include <U3xU4SingleShellTensors.h>

#ifdef USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  int8_t n = 0, ads = 0, ats = 0;
  std::vector<U3xU4::irrep> Omega_ads, Omega_ats;

  /* Example 1: n = 2
                  (3,1,0)       (3,3,0)             (3,2,0)    (4,1,0; 4,0; 2)
      { [[ad x ad]         x ad]          x [at x at]         }
                  (1,1,0,0)     (2,1,0,0)           (1,1,0,0)  (1,0,0,0; 1,0,0;
     1,0; 0)
  */
  /*
  n = 2;
  ads = 3;
  ats = 2;
  U3xU4::irrep T1_0(3,1,0, 1,1,0,0), T1(3,3,0, 2,1,0,0),
               T2(3,2,0, 1,1,0,0), T(4,1,0, 1,0,0,0);
  Omega_ads.push_back(T1_0);
  Omega_ads.push_back(T1);
  Omega_ats.push_back(T2);

  u3::gelfand g3(4,1,0, 4,0, 2);
  u4::gelfand g4(1,0,0,0, 1,0,0, 1,0, 0);
  */

  /* Example 2: n = 1
        (1,0,0)       (1,1,0)    (0,0,0; 0,0; 0)
    { ad          x at          }
        (1,0,0,0)     (1,1,1,0)  (0,0,0,0; 0,0,0; 0,0; 0)
  */

  n = 1;
  ads = 1;
  ats = 1;
  U3xU4::irrep T1(1,0,0, 1,0,0,0), T2(1,1,0, 1,1,1,0), T(0,0,0, 0,0,0,0);
  u3::gelfand g3(0,0,0, 0,0, 0);
  u4::gelfand g4(0,0,0,0, 0,0,0, 0,0, 0);

  /* Example 3: n = 3
        (3,0,0)              (3,0,0)   (6,0,0; 6,0; 3)
    { ad          x [at x at]         }
        (1,0,0,0)            (1,1,0,0) (2,1,0,0; 1,1,0; 1,0; 0)
  */
  /*
  n = 3;
  ads = 1;
  ats = 2;
  U3xU4::irrep T1(3,0,0, 1,0,0,0), T2(3,0,0, 1,1,0,0), T(6,0,0, 2,1,0,0);
  Omega_ats.push_back(T2);

  u3::gelfand g3(6,0,0, 6,0, 3);
  u4::gelfand g4(2,1,0,0, 1,1,0, 1,0, 0);  
  */

  ////////////////// DON'T EDIT BELOW /////////////////////

  // Construct single particle states
  u3::irrep x(n, 0, 0), x_conj(n, n, 0);
  std::vector<u3::gelfand> sps3 = u3::gelfandBasis(x),
                           sps3_conj = u3::gelfandBasis(x_conj);

  u4::irrep y(1, 0, 0, 0), y_conj(1, 1, 1, 0);
  std::vector<u4::gelfand> sps4 = u4::gelfandBasis(y),
                           sps4_conj = u4::gelfandBasis(y_conj);

  // Multiplicities of the final tensor
  auto [rhomax, etamax] = U3xU4::RhoEta(T1, T2, T);

  U3xU4::SingleShellTensor_gelfand 
    Tadat(n, sps3, sps3_conj, sps4, sps4_conj, ads, ats,
          Omega_ads, Omega_ats, T, rhomax, etamax, g3, g4);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto [coeffs, adArr, aArr] = Tadat.expand(dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  Eigen::RowVectorXd norm(rhomax * etamax);
  norm.setZero();

  for (std::size_t i = 0; i < coeffs.size(); i++)
  {
    std::cout << coeffs[i] << "\t";
    norm = norm + coeffs[i].cwiseProduct(coeffs[i]);

    if (!adArr.empty())
      for (std::size_t j = 0; j < adArr[i].size(); j++)
        std::cout << adArr[i][j] << "\t";

    if (!aArr.empty())
      for (std::size_t j = 0; j < aArr[i].size(); j++)
      {
        assert(!aArr[i][j].isTensor());
        std::cout << aArr[i][j] << "\t";
      }

    std::cout << std::endl;
  }

  std::cout << "Normalization: " << norm << std::endl;

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

#ifdef USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}
