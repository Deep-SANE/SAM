// clang-format off
//
// Test copy of app/tutorial/U3xU4MultiShellTensors_ShareGammas_RME.cpp: RMEs of a
// MultiShellTensor_ShareGammas_phys (one set of single-shell tensors shared by
// several couplings), computed through the RME database (RME_db), as used in
// the matrix-element code. Both examples are compared value by value with the
// numbers they produced, and cross-checked against MultiShellTensor::RME():
//   - Example 1, first coupling = Example 7 of test_multishelltensors_rme.cpp
//     (the same tensor; there the shell-3 identity is written out);
//   - Example 2 = Example 1 of test_multishelltensors_rme.cpp.
// Example 1 has three couplings; RMEs come back for two of them.
//
// The run uses a temporary data directory (set by the test Makefile), since
// RME_db caches highest-weight states in <data dir>/hws.
//
// Reference numbers were produced by this code (SAM main, Nov 2026).

#include <string>
#include <vector>

#include <sam.h>

#include <U3xU4irrep.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4MultiShellTensors.h>
#include <U3xU4ManyNucleonBasis.h>
#include <U3xU4SingleShellTensorsRME_Database.h>

#include "test_helpers.h"

namespace
{
  using Irrep = U3xU4::ManyNucleonBasis::Irrep;

  void check(const U3xU4::MultiShellTensor_ShareGammas_phys& T,
             const Irrep& bra, const Irrep& ket,
             const std::vector<std::vector<double>>& expected)
  {
    auto uT = U3xU4::unify_tensor(T, bra, ket);
    test::require(uT.has_value(), "tensor unifies with the states");

    U3xU4::Table_gelfand gelfand_labels(3);
    U3xU4::RME_db RMEdb;
    auto rmes = uT->RME(bra, ket, gelfand_labels, RMEdb);

    test::require(rmes.size() == expected.size(), "number of couplings is " +
                  std::to_string(rmes.size()) + ", expected " +
                  std::to_string(expected.size()));
    for (size_t c = 0; c < rmes.size(); c++)
    {
      test::require(rmes[c].size() == expected[c].size(),
                    "number of RMEs in coupling " + std::to_string(c));
      for (size_t k = 0; k < rmes[c].size(); k++)
        test::near(rmes[c][k], expected[c][k], "coupling " +
                   std::to_string(c) + ", RME #" + std::to_string(k),
                   1e-12, 1e-11);
    }
  }
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  using namespace U3xU4;
  test::Suite suite;

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
  suite.run("Example 1: shells 0, 1, 2, 3; three couplings share the Gammas",
            [&]
  {
    GammaElement
      gamma0_ket(0,1,u3::irrep(0,0,0),u4::irrep(0,0,0,0)),
      gamma1_ket(1,1,u3::irrep(3,1,0),u4::irrep(2,1,1,0)),
      gamma2_ket(2,1,u3::irrep(3,1,0),u4::irrep(2,0,0,0)),
      gamma3_ket(3,1,u3::irrep(4,2,0),u4::irrep(1,1,0,0)),
      gamma0_bra(0,1,u3::irrep(0,0,0),u4::irrep(1,0,0,0)),
      gamma1_bra(1,1,u3::irrep(2,1,1),u4::irrep(3,1,0,0)),
      gamma2_bra(2,1,u3::irrep(2,0,0),u4::irrep(1,0,0,0)),
      gamma3_bra(3,1,u3::irrep(4,2,0),u4::irrep(1,1,0,0));
    OmegaElement
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
    Irrep
      ket({gamma0_ket,gamma1_ket,gamma2_ket,gamma3_ket},{omega0_ket,omega1_ket,omega2_ket}),
      bra({gamma0_bra,gamma1_bra,gamma2_bra,gamma3_bra},{omega0_bra,omega1_bra,omega2_bra});
    SingleShellTensor
      Gamma0(0,1,0,{},{},irrep(0,0,0, 1,0,0,0),1,1),
      Gamma1(1,1,1,{},{},irrep(2,1,0, 2,1,1,0),1,1),
      Gamma2(2,0,1,{},{},irrep(2,2,0, 1,1,1,0),1,1);
    MultiShellTensor_ShareGammas_phys
      T({Gamma0, Gamma1, Gamma2},
        {{{Omega0, Omega1},{{}}}, {{Omega0, Omega1_2},{{}}}, {{Omega0, Omega1_3},{{}}}},
        0, 0, 0);
    check(T, bra, ket,
          // first coupling: the same as Example 7 of test_multishelltensors_rme
          {{0.0059982179371087458, 0.051946091109716389,
            -0.0054634172342265825, -0.047314581163139149},
           {0.030559604828714865, 0.023727594720458468,
            0.063348421649236514, 0.0099747895341612686}});
  });

  /* Example 2: n = 1 shell
                           [2,0,0]       [2,2,0]   [0,0,0]
  / [3,3,0,0] ||| [[ad x ad]  x  [at x at]         ]         ||| [3,3,0,0] \
  \ 1[2,2,2]  |||          [1,1,0,0]     [1,1,0,0] [0,0,0,0] ||| 1[2,2,2]  /
  */
  suite.run("Example 2: one shell (n = 1); the same as Example 1 of "
            "test_multishelltensors_rme", [&]
  {
    GammaElement gamma0(1,1,u3::irrep(2,2,2),u4::irrep(3,3,0,0));
    Irrep ket({gamma0},{}), bra({gamma0},{});
    SingleShellTensor
      Gamma(1,2,2,{irrep(2,0,0, 1,1,0,0)},
                  {irrep(2,2,0, 1,1,0,0)},
                  irrep(0,0,0, 0,0,0,0),1,1);
    MultiShellTensor_ShareGammas_phys T({Gamma}, {{{},{{}}}}, 0, 0, 0);
    check(T, bra, ket, {{-1.9999999999999967}});
  });

  sam::finalize();
  return suite.result();
}
