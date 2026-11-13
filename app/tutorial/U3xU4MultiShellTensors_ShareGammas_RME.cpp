// U3xU4MultiShellTensors_ShareGammas_RME: reduced matrix elements of
// tensors that share their single-shell parts.
//
// An operator expansion contains many multi-shell tensors with the same
// single-shell tensors (Gammas) and different couplings (Omegas).
// MultiShellTensor_ShareGammas_phys groups them, and the RME database
// (RME_db) computes the single-shell matrix elements once for all couplings,
// as the matrix-element code does. Highest-weight states are cached in
// <data dir>/hws.
//
// One example is active; the others are commented out. To try another,
// copy this file into app/work/ and swap the comments:
//   cp app/tutorial/U3xU4MultiShellTensors_ShareGammas_RME.cpp app/work/
//   make -j 8 apps && mpirun -np 1 bin/work/U3xU4MultiShellTensors_ShareGammas_RME
// The same examples, with their numbers checked, are in
// test/test_multishelltensors_sharegammas_rme.cpp.

// clang-format off

#include <chrono>
#include <iostream>

#include <sam.h>

#include <U3xU4irrep.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4MultiShellTensors.h>
#include <U3xU4ManyNucleonBasis.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  /* Example 1: n = 0, 1, 2 and 3 shells
   /                        1[3,2,0,0]     1[3,2,1,0]       1[3,1,0,0] |||
  / {{{[1,0,0,0] x [3,1,0,0]}  x  [1,0,0,0]}   x   [1,1,0,0]}          ||| 
  \ {{{1[0,0,0]  x 1[2,1,1] }  x  1[2,0,0] }   x   1[4,2,0] }          |||
   \                        1[1,0,0]       1[2,1,0]         2[4,2,0]   |||

      [0,0,0]    [2,1,0]   [2,1,0]   [2,2,0]   [2,2,0]    [2,2,0]
  [[ ad x [ad x at]        ]    x   at         ]      x I ]
      [1,0,0,0]  [2,1,1,0] [3,1,1,0] [1,1,1,0] [4,2,2,0]  [4,2,2,0]

    |||                        1[2,1,1,0]     1[1,1,0,0]      1[2,2,0,0] \
    ||| {{[0,0,0,0] x [2,1,1,0]}  x  [2,0,0,0]}  x   [1,1,0,0]}           \
    ||| {{1[0,0,0]  x 1[3,1,0] }  x  1[3,1,0] }  x   1[4,2,0] }           /
    |||                        1[3,1,0]       2[3,2,0]        1[2,0,0]   /
  */
  
  U3xU4::GammaElement 
    gamma0_ket(0,1,u3::irrep(0,0,0),u4::irrep(0,0,0,0)),
    gamma1_ket(1,1,u3::irrep(3,1,0),u4::irrep(2,1,1,0)),
    gamma2_ket(2,1,u3::irrep(3,1,0),u4::irrep(2,0,0,0)),
    gamma3_ket(3,1,u3::irrep(4,2,0),u4::irrep(1,1,0,0)),

    gamma0_bra(0,1,u3::irrep(0,0,0),u4::irrep(1,0,0,0)),
    gamma1_bra(1,1,u3::irrep(2,1,1),u4::irrep(3,1,0,0)),
    gamma2_bra(2,1,u3::irrep(2,0,0),u4::irrep(1,0,0,0)),
    gamma3_bra(3,1,u3::irrep(4,2,0),u4::irrep(1,1,0,0));

  U3xU4::OmegaElement 
    omega0_ket(1,u3::irrep(3,1,0),1,u4::irrep(2,1,1,0)),
    omega1_ket(2,u3::irrep(3,2,0),1,u4::irrep(1,1,0,0)),
    omega2_ket(1,u3::irrep(2,0,0),1,u4::irrep(2,2,0,0)),

    omega0_bra(1,u3::irrep(1,0,0),1,u4::irrep(3,2,0,0)),
    omega1_bra(1,u3::irrep(2,1,0),1,u4::irrep(3,2,1,0)),
    omega2_bra(2,u3::irrep(4,2,0),1,u4::irrep(3,1,0,0)),

    Omega0(1,u3::irrep(2,1,0),1,u4::irrep(3,1,1,0)),
    Omega1(1,u3::irrep(2,2,0),1,u4::irrep(4,2,2,0)),
    Omega1_2(1,u3::irrep(3,1,0),1,u4::irrep(4,2,2,0)),
    Omega1_3(1,u3::irrep(1,0,0),1,u4::irrep(4,2,2,0));

  U3xU4::ManyNucleonBasis::Irrep 
    ket({gamma0_ket,gamma1_ket,gamma2_ket,gamma3_ket},{omega0_ket,omega1_ket,omega2_ket}), 
    bra({gamma0_bra,gamma1_bra,gamma2_bra,gamma3_bra},{omega0_bra,omega1_bra,omega2_bra});

  U3xU4::SingleShellTensor 
    Gamma0(0,1,0,{},{},U3xU4::irrep(0,0,0, 1,0,0,0),1,1),
    Gamma1(1,1,1,{},{},U3xU4::irrep(2,1,0, 2,1,1,0),1,1),
    Gamma2(2,0,1,{},{},U3xU4::irrep(2,2,0, 1,1,1,0),1,1);

  U3xU4::MultiShellTensor_ShareGammas_phys 
    T({Gamma0, Gamma1, Gamma2},
      {{{Omega0, Omega1},{{}}}, {{Omega0, Omega1_2},{{}}}, {{Omega0, Omega1_3},{{}}}},
      0, 0, 0);
  

  /* Example 2: n = 1 shell
                           [2,0,0]       [2,2,0]   [0,0,0]
  / [3,3,0,0] ||| [[ad x ad]  x  [at x at]         ]         ||| [3,3,0,0] \
  \ 1[2,2,2]  |||          [1,1,0,0]     [1,1,0,0] [0,0,0,0] ||| 1[2,2,2]  /
  */
  /*
  U3xU4::GammaElement gamma0(1,1,u3::irrep(2,2,2),u4::irrep(3,3,0,0));
  U3xU4::ManyNucleonBasis::Irrep ket({gamma0},{}), bra({gamma0},{});

  U3xU4::SingleShellTensor 
    Gamma(1,2,2,{U3xU4::irrep(2,0,0, 1,1,0,0)},
                {U3xU4::irrep(2,2,0, 1,1,0,0)},
                U3xU4::irrep(0,0,0, 0,0,0,0),1,1);

  U3xU4::MultiShellTensor_ShareGammas_phys 
    T({Gamma}, {{{},{{}}}}, 0, 0, 0);
  */

  /////////////////// DO NOT EDIT ANYTHING BELOW ///////////////////

  std::cout << "Tensor: " << T << std::endl;

  auto uT = U3xU4::unify_tensor(T, bra, ket);
  if (!uT.has_value())
  {
    std::cout << "Tensor acting out of states" << std::endl;
    return 1;
  }

  std::cout << "Unified tensor: " << *uT << std::endl;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  U3xU4::Table_gelfand gelfand_labels(3);
  U3xU4::RME_db RMEdb;
  auto rmes = uT->RME(bra, ket, gelfand_labels, RMEdb);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  for (const auto& rme : rmes) rme.print();

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
