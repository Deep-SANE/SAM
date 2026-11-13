// clang-format off
//
// Test copy of app/tutorial/U3xU4MultiShellTensorsRME.cpp (Examples 1-7):
// multi-shell triple-reduced matrix elements
//   < bra ||| T ||| ket >
// computed with MultiShellTensor::RME() and compared value by value with the
// numbers it produced.
//
// Tolerance: absolute 1e-12 plus relative 1e-11. Example 5 builds its
// highest-weight states with the iterative null-space solver (KraNS LOBPCG,
// tolerance 1e-14, random start), so its last digits vary slightly from run
// to run (about 1e-14 relative).
//
// The run uses a temporary data directory (set by the test Makefile), since
// highest-weight states are cached in <data dir>/hws.
//
// Reference numbers were produced by this code (SAM main, Nov 2026) and agree
// with the app.

#include <string>
#include <vector>

#include <sam.h>

#include <U3xU4irrep.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4MultiShellTensors.h>
#include <U3xU4ManyNucleonBasis.h>

#include "test_helpers.h"

namespace
{
  void check(const U3xU4::MultiShellTensor& tensor,
             const U3xU4::ManyNucleonBasis::Irrep& bra,
             const U3xU4::ManyNucleonBasis::Irrep& ket,
             size_t rhomax, size_t etamax, const std::vector<double>& expected)
  {
    auto rmes = tensor.RME(bra, ket);
    test::require(rmes.rhomax() == rhomax && rmes.etamax() == etamax,
                  "multiplicities rho, eta");
    test::require(rmes.size() == expected.size(), "number of RMEs is " +
                  std::to_string(rmes.size()) + ", expected " +
                  std::to_string(expected.size()));
    for (size_t k = 0; k < rmes.size(); k++)
      test::near(rmes[k], expected[k], "RME #" + std::to_string(k), 1e-12, 1e-11);
  }
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  using namespace U3xU4;
  using Irrep = ManyNucleonBasis::Irrep;
  test::Suite suite;

  /* Example 1: n = 1 shell
                           [2,0,0]       [2,2,0]   [0,0,0]
  / [3,3,0,0] ||| [[ad x ad]  x  [at x at]         ]         ||| [3,3,0,0] \
  \ 1[2,2,2]  |||          [1,1,0,0]     [1,1,0,0] [0,0,0,0] ||| 1[2,2,2]  /
  */
  suite.run("Example 1: one shell (n = 1)", [&]
  {
    GammaElement gamma0(1,1,u3::irrep(2,2,2),u4::irrep(3,3,0,0));
    Irrep ket({gamma0},{}), bra({gamma0},{});
    SingleShellTensor
      tadat(1,2,2,{irrep(2,0,0, 1,1,0,0)},
            {irrep(2,2,0, 1,1,0,0)},irrep(0,0,0, 0,0,0,0),1,1);
    MultiShellTensor tensor({tadat},{});
    check(tensor, bra, ket, 1, 1, {-1.9999999999999967});
  });

  /* Example 2: n = 1 and n = 2 shells
   /                      1[3,2,1,0]|||
  / {[3,1,0,0] x [2,0,0,0]}         |||
  \ {1[2,1,1]  x 1[3,1,0] }         |||
   \                      1[3,2,0]  |||

               [2,1,0]         [2,1,0]   [4,2,0]
      [[ad x at]    x  [ad x at]         ]
               [0,0,0,0]       [0,0,0,0] [0,0,0,0]
  */
  suite.run("Example 2: shells 1, 2; [ad x at] x [ad x at]", [&]
  {
    GammaElement
      gamma0(1,1,u3::irrep(2,1,1),u4::irrep(3,1,0,0)),
      gamma1(2,1,u3::irrep(3,1,0),u4::irrep(2,0,0,0));
    OmegaElement
      omega0(1,u3::irrep(3,2,0),1,u4::irrep(3,2,1,0)),
      irr0(1,u3::irrep(4,2,0),1,u4::irrep(0,0,0,0));
    Irrep ket({gamma0,gamma1},{omega0}), bra({gamma0,gamma1},{omega0});
    SingleShellTensor
      tadat0(1,1,1,{},{},irrep(2,1,0, 0,0,0,0),1,1),
      tadat1(2,1,1,{},{},irrep(2,1,0, 0,0,0,0),1,1);
    MultiShellTensor tensor({tadat0,tadat1},{irr0});
    check(tensor, bra, ket, 2, 1, {0.050000000000000142, -0.13228756555322968});
  });

  /* Example 3: n = 1 and n = 2 shells, [ I x [ad x at] ] */
  suite.run("Example 3: shells 1, 2; I x [ad x at]", [&]
  {
    GammaElement
      gamma0(1,1,u3::irrep(2,1,1),u4::irrep(3,1,0,0)),
      gamma1(2,1,u3::irrep(3,1,0),u4::irrep(2,0,0,0));
    OmegaElement
      omega0(1,u3::irrep(3,2,0),1,u4::irrep(3,2,1,0)),
      irr0(1,u3::irrep(2,1,0),1,u4::irrep(0,0,0,0));
    Irrep ket({gamma0,gamma1},{omega0}), bra({gamma0,gamma1},{omega0});
    SingleShellTensor
      tadat0(1,0,0,{},{},irrep(0,0,0, 0,0,0,0),1,1),
      tadat1(2,1,1,{},{},irrep(2,1,0, 0,0,0,0),1,1);
    MultiShellTensor tensor({tadat0,tadat1},{irr0});
    check(tensor, bra, ket, 2, 1, {0.35856858280031778, 0.55205244747388338});
  });

  /* Example 4: n = 1 and n = 2 shells, [ [ad x at] x I ] */
  suite.run("Example 4: shells 1, 2; [ad x at] x I", [&]
  {
    GammaElement
      gamma0(1,1,u3::irrep(3,1,0),u4::irrep(2,1,1,0)),
      gamma1(2,1,u3::irrep(3,1,0),u4::irrep(2,0,0,0));
    OmegaElement
      omega0(2,u3::irrep(3,2,0),1,u4::irrep(2,0,0,0)),
      irr0(1,u3::irrep(2,1,0),1,u4::irrep(0,0,0,0));
    Irrep ket({gamma0,gamma1},{omega0}), bra({gamma0,gamma1},{omega0});
    SingleShellTensor
      tadat0(1,1,1,{},{},irrep(2,1,0, 0,0,0,0),1,1),
      tadat1(2,0,0,{},{},irrep(0,0,0, 0,0,0,0),1,1);
    MultiShellTensor tensor({tadat0,tadat1},{irr0});
    check(tensor, bra, ket, 2, 1,
          {-0.40089186286863809, 1.0029717747535996, -0.38575837490522985,
            0.1336306209562125, -0.38575837490522985, 0.13363062095621256,
            0.935414346693487, 0.54006172486732251});
  });

  /* Example 5: n = 0, 1 and 2 shells, [[ ad x [ad x at] ] x at ] */
  suite.run("Example 5: shells 0, 1, 2; [[ad x [ad x at]] x at]", [&]
  {
    GammaElement
      gamma0_ket(0,1,u3::irrep(0,0,0),u4::irrep(1,1,1,0)),
      gamma1_ket(1,1,u3::irrep(1,1,0),u4::irrep(2,0,0,0)),
      gamma2_ket(2,1,u3::irrep(2,0,0),u4::irrep(1,0,0,0)),
      gamma0_bra(0,1,u3::irrep(0,0,0),u4::irrep(1,1,1,1)),
      gamma1_bra(1,1,u3::irrep(1,1,0),u4::irrep(2,0,0,0)),
      gamma2_bra(2,1,u3::irrep(0,0,0),u4::irrep(0,0,0,0));
    OmegaElement
      omega0_ket(1,u3::irrep(1,1,0),1,u4::irrep(3,1,1,0)),
      omega1_ket(1,u3::irrep(1,0,0),1,u4::irrep(3,2,1,0)),
      omega0_bra(1,u3::irrep(1,1,0),1,u4::irrep(2,0,0,0)),
      omega1_bra(1,u3::irrep(1,1,0),1,u4::irrep(2,0,0,0)),
      irr0(1,u3::irrep(2,1,0),1,u4::irrep(3,1,1,0)),
      irr1(1,u3::irrep(2,2,0),1,u4::irrep(4,2,2,0));
    Irrep
      ket({gamma0_ket,gamma1_ket,gamma2_ket},{omega0_ket,omega1_ket}),
      bra({gamma0_bra,gamma1_bra,gamma2_bra},{omega0_bra,omega1_bra});
    SingleShellTensor
      tadat0(0,1,0,{},{},irrep(0,0,0, 1,0,0,0),1,1),
      tadat1(1,1,1,{},{},irrep(2,1,0, 2,1,1,0),1,1),
      tadat2(2,0,1,{},{},irrep(2,2,0, 1,1,1,0),1,1);
    MultiShellTensor tensor({tadat0,tadat1,tadat2},{irr0,irr1});
    check(tensor, bra, ket, 1, 1, {1.9720265943665432});
  });

  /* Example 6: n = 0, 1 and 2 shells, same tensor, other states */
  suite.run("Example 6: shells 0, 1, 2; [[ad x [ad x at]] x at]", [&]
  {
    GammaElement
      gamma0_ket(0,1,u3::irrep(0,0,0),u4::irrep(0,0,0,0)),
      gamma1_ket(1,1,u3::irrep(3,1,0),u4::irrep(2,1,1,0)),
      gamma2_ket(2,1,u3::irrep(3,1,0),u4::irrep(2,0,0,0)),
      gamma0_bra(0,1,u3::irrep(0,0,0),u4::irrep(1,0,0,0)),
      gamma1_bra(1,1,u3::irrep(2,1,1),u4::irrep(3,1,0,0)),
      gamma2_bra(2,1,u3::irrep(2,0,0),u4::irrep(1,0,0,0));
    OmegaElement
      omega0_ket(1,u3::irrep(3,1,0),1,u4::irrep(2,1,1,0)),
      omega1_ket(2,u3::irrep(3,2,0),1,u4::irrep(1,1,0,0)),
      omega0_bra(1,u3::irrep(1,0,0),1,u4::irrep(3,2,0,0)),
      omega1_bra(1,u3::irrep(2,1,0),1,u4::irrep(3,2,1,0)),
      irr0(1,u3::irrep(2,1,0),1,u4::irrep(3,1,1,0)),
      irr1(1,u3::irrep(2,2,0),1,u4::irrep(4,2,2,0));
    Irrep
      ket({gamma0_ket,gamma1_ket,gamma2_ket},{omega0_ket,omega1_ket}),
      bra({gamma0_bra,gamma1_bra,gamma2_bra},{omega0_bra,omega1_bra});
    SingleShellTensor
      tadat0(0,1,0,{},{},irrep(0,0,0, 1,0,0,0),1,1),
      tadat1(1,1,1,{},{},irrep(2,1,0, 2,1,1,0),1,1),
      tadat2(2,0,1,{},{},irrep(2,2,0, 1,1,1,0),1,1);
    MultiShellTensor tensor({tadat0,tadat1,tadat2},{irr0,irr1});
    check(tensor, bra, ket, 1, 1, {0.031004898176538339, 0.26851029462631892});
  });

  /* Example 7: n = 0, 1, 2 and 3 shells, [[[ad x [ad x at]] x at] x I] */
  suite.run("Example 7: shells 0, 1, 2, 3; [[[ad x [ad x at]] x at] x I]", [&]
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
      irr0(1,u3::irrep(2,1,0),1,u4::irrep(3,1,1,0)),
      irr1(1,u3::irrep(2,2,0),1,u4::irrep(4,2,2,0)),
      irr2(1,u3::irrep(2,2,0),1,u4::irrep(4,2,2,0));
    Irrep
      ket({gamma0_ket,gamma1_ket,gamma2_ket,gamma3_ket},{omega0_ket,omega1_ket,omega2_ket}),
      bra({gamma0_bra,gamma1_bra,gamma2_bra,gamma3_bra},{omega0_bra,omega1_bra,omega2_bra});
    SingleShellTensor
      tadat0(0,1,0,{},{},irrep(0,0,0, 1,0,0,0),1,1),
      tadat1(1,1,1,{},{},irrep(2,1,0, 2,1,1,0),1,1),
      tadat2(2,0,1,{},{},irrep(2,2,0, 1,1,1,0),1,1),
      tadat3(3,0,0,{},{},irrep(0,0,0, 0,0,0,0),1,1);
    MultiShellTensor tensor({tadat0,tadat1,tadat2,tadat3},{irr0,irr1,irr2});
    check(tensor, bra, ket, 1, 1,
          {0.005998217937108744, 0.051946091109716382,
           -0.0054634172342265816, -0.047314581163139136});
  });

  sam::finalize();
  return suite.result();
}
