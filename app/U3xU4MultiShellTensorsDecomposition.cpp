#include <chrono>
#include <iostream>
#include <utility>

#include <qcx_convert.h>

#include <sam.h>

#include <U3xU4irrep.h>
#include <U3xU4MultiShellTensors.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  int maxShell, LL = 0, SS = 0, TT = 0, MMT = 0, JJ = 0, MMJ = 0;
  U3xU4::SingleShellTensors Gammas;
  U3xU4::OmegaElements Omegas;

  // Example 1
  /*
  U3xU4::irrep Gamma1(2,1,0, 2,1,1,0), 
               Gamma2(2,2,0, 1,1,1,0),
               Gamma3(3,0,0, 1,0,0,0);

  U3xU4::SingleShellTensor tadat1(1,1,1,{},{},Gamma1,1,1),
                           tadat2(2,0,1,{},{},Gamma2,1,1),
                           tadat3(3,1,0,{},{},Gamma3,1,1);

  U3xU4::OmegaElement Omega1(1, {4,3,0}, 1, {3,2,2,0}), 
                      Omega2(1, {7,3,0}, 1, {4,2,2,0});

  Gammas = {tadat1,tadat2,tadat3};
  Omegas = {Omega1,Omega2};

  maxShell = 3;
  LL = 4;
  SS = 4;
  */

  // Example 2
  /*
  U3xU4::irrep Gamma1(0, 0, 0, 1, 0, 0, 0), 
               Gamma2(1, 1, 0, 1, 1, 1, 0),
               Gamma3(2, 2, 0, 1, 1, 1, 0),
               Gamma4(3, 0, 0, 1, 0, 0, 0);

  U3xU4::SingleShellTensor 
      tadat1(0, 1, 0, {}, {}, Gamma1, 1, 1),
      tadat2(1, 0, 1, {}, {}, Gamma2, 1, 1),
      tadat3(2, 0, 1, {}, {}, Gamma3, 1, 1),
      tadat4(3, 1, 0, {}, {}, Gamma4, 1, 1);

  U3xU4::OmegaElement Omega1(1, {1,1,0}, 1, {2,1,1,0}), 
                      Omega2(1, {3,3,0}, 1, {3,2,2,0}), 
                      Omega3(1, {6,3,0}, 1, {4,2,2,0});

  Gammas = {tadat1, tadat2, tadat3, tadat4};
  Omegas = {Omega1, Omega2, Omega3};

  maxShell = 3;
  LL = 4;
  SS = 4;
  */

  // Example 3
  /*
  U3xU4::irrep Gamma1_ad(0, 0, 0, 2, 0, 0, 0),
               Gamma1(0, 0, 0, 3, 1, 1, 0),
               Gamma2(1, 1, 0, 1, 1, 1, 0);

  U3xU4::SingleShellTensor 
    tadat1(0, 2, 1, {Gamma1_ad}, {}, Gamma1, 1, 1),
    tadat2(1, 0, 1, {}, {}, Gamma2, 1, 1);

  U3xU4::OmegaElement Omega1(1, {1,1,0}, 1, {3,1,0,0});

  Gammas = {tadat1, tadat2};
  Omegas = {Omega1};

  maxShell = 1;
  LL = 2;
  SS = 2;
  */

  // Example 4
  /*
  U3xU4::irrep Gamma1(0, 0, 0, 1, 1, 0, 0), 
               Gamma2(2, 2, 0, 1, 1, 0, 0);

  U3xU4::SingleShellTensor 
      tadat1(0, 2, 0, {Gamma1}, {}, Gamma1, 1, 1),
      tadat2(1, 0, 2, {}, {Gamma2}, Gamma2, 1, 1);

  U3xU4::OmegaElement Omega1(1, {2,2,0}, 1, {0,0,0,0});

  Gammas = {tadat1, tadat2};
  Omegas = {Omega1};
  maxShell = 1;
  */

  // Example 5
  /*
  U3xU4::irrep Gamma1(2, 1, 0, 0, 0, 0, 0),
               Gamma2_ad(3, 1, 0, 1, 1, 0, 0),
               Gamma2_at(3, 2, 0, 1, 1, 0, 0),
               Gamma2(2, 1, 0, 0, 0, 0, 0);

  U3xU4::SingleShellTensor 
      tadat1(1, 1, 1, {}, {}, Gamma1, 1, 1),
      tadat2(2, 2, 2, {Gamma2_ad}, {Gamma2_at}, Gamma2, 2, 1);

  U3xU4::OmegaElement Omega1(2, {2,1,0}, 1, {0,0,0,0});

  Gammas = {tadat1, tadat2};
  Omegas = {Omega1};
  maxShell = 2;
  LL = 2;
  JJ = 2;
  */

  // Example 6
  
  U3xU4::irrep Gamma(2, 1, 0, 0, 0, 0, 0);

  U3xU4::SingleShellTensor 
      tadat1(1, 1, 1, {}, {}, Gamma, 1, 1),
      tadat2(2, 1, 1, {}, {}, Gamma, 1, 1),
      tadat3(3, 1, 1, {}, {}, Gamma, 1, 1);

  U3xU4::OmegaElement Omega(2, {2,1,0}, 1, {0,0,0,0});

  Gammas = {tadat1, tadat2, tadat3};
  Omegas = {Omega, Omega};
  maxShell = 3;
  LL = 2;
  JJ = 2;
  

  // Example 7
  /*
  U3xU4::irrep Gamma1_ad(2, 0, 0, 1, 1, 0, 0),
               Gamma1_at(2, 2, 0, 1, 1, 0, 0),
               Gamma(2, 1, 0, 0, 0, 0, 0);

  U3xU4::SingleShellTensor 
      tadat1(1, 2, 2, {Gamma1_ad}, {Gamma1_at}, Gamma, 1, 1),
      tadat2(3, 1, 1, {}, {}, Gamma, 1, 1);

  U3xU4::OmegaElement Omega(2, {2,1,0}, 1, {0,0,0,0});

  Gammas = {tadat1, tadat2};
  Omegas = {Omega};
  maxShell = 3;
  LL = 2;
  JJ = 2;
  */
  
  // Example 8: should not run
  /*
  U3xU4::irrep Gamma1(0, 0, 0, 1, 0, 0, 0), 
               Gamma2(2, 2, 0, 1, 1, 1, 0),
               Gamma3(3, 3, 0, 1, 1, 1, 0),
               Gamma4(4, 0, 0, 1, 0, 0, 0),
               Gamma5(4, 2, 0, 0, 0, 0, 0);

  U3xU4::SingleShellTensor 
      tadat1(0, 1, 0, {}, {}, Gamma1, 1, 1),
      tadat2(2, 0, 1, {}, {}, Gamma2, 1, 1),
      tadat3(3, 0, 1, {}, {}, Gamma3, 1, 1),
      tadat4(4, 1, 0, {}, {}, Gamma4, 1, 1),
      tadat5(6, 1, 1, {}, {}, Gamma5, 1, 1);

  U3xU4::OmegaElement Omega1(1, {2,2,0}, 1, {0,0,0,0}), 
                      Omega2(1, {5,5,0}, 1, {1,1,1,0}), 
                      Omega3(1, {1,1,0}, 1, {0,0,0,0}),
                      Omega4(1, {3,2,0}, 1, {0,0,0,0});

  Gammas = {tadat1, tadat2, tadat3, tadat4, tadat5};
  Omegas = {Omega1, Omega2, Omega3, Omega4};

  maxShell = 6;
  LL = 2;
  JJ = 2;
  */

  ////////////////// DO NOT EDIT ANYTHING BELOW ///////////////////
  
  HO::Table_nljmjmt labels(maxShell);
  U3xU4::MultiShellTensor_phys T(Gammas, Omegas, MMT, JJ, MMJ, labels);

  int kappamax = u3::Kappa(T.getLastIrrep().u3(), LL / 2);
  int zetamax = u4::Zeta(T.getLastIrrep().u4(), SS, TT);

  T.setLLSSTTstrengths({{qcx_utils::itou8(kappamax), 
                         qcx_utils::ito8(LL), 
                         qcx_utils::itou8(zetamax), 
                         qcx_utils::ito8(SS), 
                         qcx_utils::ito8(TT), {}}});
  
  std::cout << T << std::endl;

  // Decomposition cache (constructed outside the timed region). Note: the
  // first expand of a given tensor is a cache miss, so this still times the
  // full decomposition (plus a small pack/store overhead).
  U3xU4::SSTDecomp_phys_db decompdb(maxShell);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto exp = T.expand(LL, SS, TT, decompdb);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  decompdb.finalize();

  bool print;
  std::cout << "Finish expanding with " << exp.rows() << " terms."
            << " Do you want to print the results? (1 yes, 0 no) ";
  std::cin >> print;

  if (print)
    std::cout << exp;

  std::vector<double> norm = exp.coeffs.norm();
  std::cout << "Normalization:";
  for (double n : norm) std::cout << " " << n;
  std::cout << std::endl;

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
