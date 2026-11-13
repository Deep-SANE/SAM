#include <chrono>
#include <fstream>
#include <iostream>

#include <sam.h>

#include <U3xU4irrep.h>
#include <U3xU4SingleShellTensors.h>
#include <U3xU4SingleShellTensorsRME_Database.h>

#include <ND.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  U3xU4::irreps Omega_ads, Omega_ats;
  int n = 0, ads = 0, ats = 0;

  /* Example 1:
    < alpha_i (4,3,1) (3,1,0,0) |||
                 (3,1,0)       (3,3,0)            (3,2,0)   (4,1,0)
      { [[ad x ad]         x ad]        x [at x at]        }
                 (1,1,0,0)     (2,1,0,0)          (1,1,0,0) (1,0,0,0)
                                        ||| alpha_f (5,3,2) (3,1,1,0) >
  */
  /*
  n = 2; 
  ads = 3; 
  ats = 2;
  U3xU4::GammaElement i(n,{4,3,1},{3,1,0,0}), f(n,{5,3,2},{3,1,1,0});
  U3xU4::irrep T1_0(3,1,0, 1,1,0,0), T1(3,3,0, 2,1,0,0),
               T2(3,2,0, 1,1,0,0), T(4,1,0, 1,0,0,0);

  Omega_ads.push_back(T1_0);
  Omega_ads.push_back(T1);
  Omega_ats.emplace_back(T2);

  auto [rho0max, eta0max] = U3xU4::RhoEta(T1,T2,T);
  */

  /* Example 2:
    < alpha_i (14,2,0) (1,1,1,1) |||
                (6,2,0)             (6,4,0)   (4,2,0)
      { [ad x ad]         x [at x at]         }
                (1,1,0,0)           (1,1,0,0) (0,0,0,0)
                ||| alpha_f (14,2,0) (1,1,1,1) >
  */
  /*
  n = 4; 
  ads = 2; 
  ats = 2;
  U3xU4::GammaElement i(n,{14,2,0},{1,1,1,1}), f(n,{14,2,0},{1,1,1,1});
  U3xU4::irrep T1(6, 2, 0, 1, 1, 0, 0), 
    T2(6, 4, 0, 1, 1, 0, 0), T(4, 2, 0, 0, 0, 0, 0);

  Omega_ads.push_back(T1);
  Omega_ats.push_back(T2);

  auto [rho0max, eta0max] = U3xU4::RhoEta(T1,T2,T);
  */

  /* Example 3:
                              (n,0,0)
    < (n,0,0) (1,0,0,0) ||| ad         ||| (0,0,0) (0,0,0,0) >
                              (1,0,0,0)

  */
  /*
  n = 4; 
  ads = 1;
  U3xU4::GammaElement i(n,{0,0,0},{0,0,0,0}), f(n,{n,0,0},{1,0,0,0});
  U3xU4::irrep T(n,0,0, 1,0,0,0);
  uint8_t rho0max = 1, eta0max = 1;
  */

  /* Example 4:
                              (n,n,0)
    < (0,0,0) (0,0,0,0) ||| at         ||| (n,0,0) (1,0,0,0) >
                              (1,1,1,0)

  */
  /*
  n = 4;
  ats = 1;
  U3xU4::GammaElement i(n,{n,0,0},{1,0,0,0}), f(n,{0,0,0}, {0,0,0,0});
  U3xU4::irrep T(n,n,0, 1,1,1,0);
  uint8_t rho0max = 1, eta0max = 1;
  */

  /* Example 5: scalar
                                    (0,0,0)
    / (n13 n23 n33)     ||| [ad x at]         ||| (n13 n23 n33)     \
    \ (n14 n24 n34 n44) |||         (0,0,0,0) ||| (n14 n24 n34 n44) /
  */
  /*
  n = 8; 
  ads = 1; 
  ats = 1;
  U3xU4::GammaElement i(n,{n,0,0},{1,0,0,0}), f(n,{n,0,0},{1,0,0,0});
  U3xU4::irrep T(0,0,0, 0,0,0,0);
  int8_t rho0max = 1, eta0max = 1;
  */

  /* Example 6:          
    < alpha_i (5,3,2) (3,1,1,0) ||| I ||| alpha_f (5,3,2) (3,1,1,0) >
  */
  
  n = 2; 
  U3xU4::GammaElement i(n,{5,3,2},{3,1,1,0}), f(n,{5,3,2},{3,1,1,0});
  U3xU4::irrep T(0,0,0, 0,0,0,0);
  uint8_t rho0max = 1, eta0max = 1;
  

  ////////////////// DON'T EDIT BELOW /////////////////////

  U3xU4::SingleShellTensor 
    Tadxat(n, ads, ats, Omega_ads, Omega_ats, T, rho0max, eta0max);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto RMEs = Tadxat.RME(f, i);
/*
  U3xU4::Table_gelfand gelfands(n);
  U3xU4::RME_db RMEdb;
  auto RMEs = RMEdb.get(Tadxat, f, i, gelfands);
*/
  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "< " << f << " ||| " << Tadxat << " ||| " << i << " >" 
            << RMEs.rhomax() << "," << RMEs.etamax() << std::endl;

  std::cout << "af\tai\trho0\teta0\trho\teta" << std::endl;

  utils::NDIndex multis;
  multis.add_dimensions(f.alphamax(), i.alphamax(), rho0max, eta0max, 
                        RMEs.rhomax(), RMEs.etamax());

  std::vector<size_t> multi(6);
  for (size_t i = 0; i < RMEs.size(); i++)
  {
    multis.to_multi(i, multi);
    std::cout << multi[0] + 1 << "\t" << multi[1] + 1 << "\t"
              << multi[2] + 1 << "\t" << multi[3] + 1 << "\t"
              << multi[4] + 1 << "\t" << multi[5] + 1 << "\t"
              << RMEs[i] << std::endl;
  }

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
