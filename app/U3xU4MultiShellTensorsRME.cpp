#include <chrono>
#include <iostream>

#include <U3xU4Irrep.h>
#include <U4xSU3Irrep.h>
#include <U3xU4MultiShellTensors.h>
#include <U4xSU3ManyNucleonBasis.h>

#include <qcx_ND.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  /* Example 1: n = 1 shell
                                 [2,0,0]         [2,2,0]   [0,0,0]
  < [3,3,0,0]1(0,0) ||| [[ad x ad]    x  [at x at]         ]        ||| [3,3,0,0]1(0,0) >
                                 [1,1,0,0]       [1,1,0,0] [0,0,0,0]
  */
  /*
  U4xSU3::ManyNucleonBasis::GammaElement gamma0(u4::irrep(3,3,0,0),1,su3::irrep(0,0));
  U4xSU3::Irrep ket({},{gamma0},{}), bra({},{gamma0},{});

  U3xU4::Tadat tadat(1,2,2,{U3xU4::irrep(2,0,0, 1,1,0,0)},
    {U3xU4::irrep(2,2,0, 1,1,0,0)},U3xU4::irrep(0,0,0, 0,0,0,0),1,1);
  U3xU4::MultiShellTensor tensor({tadat},{});
  */

  /* Example 2: n = 1 and n = 2 shells
                                 
  < ([3,1,0,0]1(1,0) x [2,0,0,0]1(2,1)) 1[3,2,1,0]1(1,2) ||| 
               [2,1,0]         [2,1,0]   [4,2,0]
      [[ad x at]    x  [ad x at]         ]        
               [0,0,0,0]       [0,0,0,0] [0,0,0,0]
          ||| ([3,1,0,0]1(1,0) x [2,0,0,0]1(2,1)) 1[3,2,1,0]1(1,2) >
  */
  /*
  U4xSU3::ManyNucleonBasis::GammaElement 
    gamma0(u4::irrep(3,1,0,0),1,su3::irrep(1,0)),
    gamma1(u4::irrep(2,0,0,0),1,su3::irrep(2,1));
  U4xSU3::ManyNucleonBasis::OmegaElement 
    omega0(1,u4::irrep(3,2,1,0),1,u3::irrep(3,2,0));
  U4xSU3::Irrep ket({},{gamma0,gamma1},{omega0}), 
                bra({},{gamma0,gamma1},{omega0});

  U3xU4::Tadat tadat0(1,1,1,{},{},U3xU4::irrep(2,1,0, 0,0,0,0),1,1),
               tadat1(2,1,1,{},{},U3xU4::irrep(2,1,0, 0,0,0,0),1,1);
  U3xU4::irrep t(4,2,0, 0,0,0,0);
  U3xU4::MultiShellTensor tensor({tadat0,tadat1},{{t,1,1}});
  */

  /* Example 3: n = 1 and n = 2 shells
                                 
  < ([3,1,0,0]1(1,0) x [2,0,0,0]1(2,1)) 1[3,2,1,0]1(1,2) ||| 
                     [2,1,0]   [2,1,0]
      [ I x  [ad x at]         ]        
                     [0,0,0,0] [0,0,0,0]
          ||| ([3,1,0,0]1(1,0) x [2,0,0,0]1(2,1)) 1[3,2,1,0]1(1,2) >
  */
  /*
  U4xSU3::ManyNucleonBasis::GammaElement 
    gamma0(u4::irrep(3,1,0,0),1,su3::irrep(1,0)),
    gamma1(u4::irrep(2,0,0,0),1,su3::irrep(2,1));
  U4xSU3::ManyNucleonBasis::OmegaElement 
    omega0(1,u4::irrep(3,2,1,0),1,u3::irrep(3,2,0));
  U4xSU3::Irrep ket({},{gamma0,gamma1},{omega0}), 
                bra({},{gamma0,gamma1},{omega0});

  U3xU4::Tadat tadat0(1,0,0,{},{},U3xU4::irrep(0,0,0, 0,0,0,0),1,1),
               tadat1(2,1,1,{},{},U3xU4::irrep(2,1,0, 0,0,0,0),1,1);
  U3xU4::irrep t(2,1,0, 0,0,0,0);
  U3xU4::MultiShellTensor tensor({tadat0,tadat1},{{t,1,1}});
  */

  /* Example 4: n = 1 and n = 2 shells
                                 
  < ([2,1,1,0]1(2,1) x [2,0,0,0]1(2,1)) 1[2,0,0,0]2(1,2) ||| 
                [2,1,0]     [2,1,0]
      [ [ad x at]      x I  ]        
                [0,0,0,0]   [0,0,0,0]
          ||| ([2,1,1,0]1(2,1) x [2,0,0,0]1(2,1)) 1[2,0,0,0]2(1,2) >
  */

  U4xSU3::ManyNucleonBasis::GammaElement 
    gamma0(u4::irrep(2,1,1,0),1,su3::irrep(2,1)),
    gamma1(u4::irrep(2,0,0,0),1,su3::irrep(2,1));
  U4xSU3::ManyNucleonBasis::OmegaElement 
    omega0(1,u4::irrep(2,0,0,0),2,u3::irrep(3,2,0));
  U4xSU3::Irrep ket({},{gamma0,gamma1},{omega0}), 
                bra({},{gamma0,gamma1},{omega0});

  U3xU4::Tadat tadat0(1,1,1,{},{},U3xU4::irrep(2,1,0, 0,0,0,0),1,1),
               tadat1(2,0,0,{},{},U3xU4::irrep(0,0,0, 0,0,0,0),1,1);
  U3xU4::irrep t(2,1,0, 0,0,0,0);
  U3xU4::MultiShellTensor tensor({tadat0,tadat1},{{t,1,1}});
  

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto RMEs = tensor.RME(ket, bra, dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "RMEs are:" << std::endl;
  for (std::size_t i = 0; i < RMEs.size(); i++)
    std::cout << RMEs[i] << std::endl;

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
