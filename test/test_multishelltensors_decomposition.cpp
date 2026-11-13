// Test copy of app/tutorial/U3xU4MultiShellTensorsDecomposition.cpp: Examples 1-7,
// each expanded with MultiShellTensor_phys::expand() and checked against the
// numbers it produced. (Example 8 of the app is marked "should not run" and
// is left out.)
//
// For each example we check
//   - the number of terms in the expansion (exact);
//   - the normalization: per multiplicity column, the sum of squared
//     coefficients is 1;
//   - a signed checksum over all terms (expansion_checksum.h).
// One more case checks the single-shell decomposition cache on disk
// (SSTDecomp_phys_db): Example 1 is expanded again with a fresh database that
// reads the pieces written by the first run, and must give the same result.
//
// The run uses a temporary data directory (set by the test Makefile), since
// the decomposition database writes into <data dir>/decomp_phys.
//
// Reference numbers were produced by this code (SAM main, Nov 2026) and agree
// with the app.

#include <string>
#include <vector>

#include <convert.h>

#include <sam.h>

#include <U3xU4irrep.h>
#include <U3xU4MultiShellTensors.h>
#include <U3xU4SingleShellTensorsDecomposition_Database.h>

#include "expansion_checksum.h"
#include "test_helpers.h"

namespace
{
  struct Example
  {
    std::string name;
    U3xU4::SingleShellTensors Gammas;
    U3xU4::OmegaElements Omegas;
    int maxShell, LL, SS, TT, MMT, JJ, MMJ;
    // expected:
    size_t terms;
    std::vector<double> norm;
    double checksum;
  };

  // expand an example as the app does; returns the expansion's checksum
  double expand_and_check(const Example& x)
  {
    HO::Table_nljmjmt labels(x.maxShell);
    U3xU4::MultiShellTensor_phys T(x.Gammas, x.Omegas, x.MMT, x.JJ, x.MMJ,
                                   labels);

    int kappamax = u3::Kappa(T.getLastIrrep().u3(), x.LL / 2);
    int zetamax = u4::Zeta(T.getLastIrrep().u4(), x.SS, x.TT);
    T.setLLSSTTstrengths({{utils::itou8(kappamax), utils::ito8(x.LL),
                           utils::itou8(zetamax), utils::ito8(x.SS),
                           utils::ito8(x.TT), {}}});

    U3xU4::SSTDecomp_phys_db decompdb(x.maxShell);
    auto e = T.expand(x.LL, x.SS, x.TT, decompdb);
    decompdb.finalize();

    test::require(e.rows() == x.terms, "number of terms is " +
                  std::to_string(e.rows()) + ", expected " +
                  std::to_string(x.terms));
    auto norm = e.coeffs.norm();
    test::require(norm.size() == x.norm.size(), "number of columns");
    for (size_t c = 0; c < norm.size(); c++)
      test::near(norm[c], x.norm[c], "normalization", 1e-9, 0);
    double sum = test::checksum(e);
    test::near(sum, x.checksum, "checksum", 1e-10, 1e-8);
    return sum;
  }
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  using U3xU4::irrep;
  using U3xU4::OmegaElement;
  using U3xU4::SingleShellTensor;

  std::vector<Example> examples;

  // Example 1
  {
    irrep Gamma1(2,1,0, 2,1,1,0), Gamma2(2,2,0, 1,1,1,0), Gamma3(3,0,0, 1,0,0,0);
    SingleShellTensor tadat1(1,1,1,{},{},Gamma1,1,1),
                      tadat2(2,0,1,{},{},Gamma2,1,1),
                      tadat3(3,1,0,{},{},Gamma3,1,1);
    OmegaElement Omega1(1, {4,3,0}, 1, {3,2,2,0}),
                 Omega2(1, {7,3,0}, 1, {4,2,2,0});
    examples.push_back({"Example 1: shells 1, 2, 3; LL = SS = 4",
      {tadat1, tadat2, tadat3}, {Omega1, Omega2}, 3, 4, 4, 0, 0, 0, 0,
      79820, {1.0}, 3.1615590239191431});
  }

  // Example 2
  {
    irrep Gamma1(0,0,0, 1,0,0,0), Gamma2(1,1,0, 1,1,1,0),
          Gamma3(2,2,0, 1,1,1,0), Gamma4(3,0,0, 1,0,0,0);
    SingleShellTensor tadat1(0,1,0,{},{},Gamma1,1,1),
                      tadat2(1,0,1,{},{},Gamma2,1,1),
                      tadat3(2,0,1,{},{},Gamma3,1,1),
                      tadat4(3,1,0,{},{},Gamma4,1,1);
    OmegaElement Omega1(1, {1,1,0}, 1, {2,1,1,0}),
                 Omega2(1, {3,3,0}, 1, {3,2,2,0}),
                 Omega3(1, {6,3,0}, 1, {4,2,2,0});
    examples.push_back({"Example 2: shells 0, 1, 2, 3; LL = SS = 4",
      {tadat1, tadat2, tadat3, tadat4}, {Omega1, Omega2, Omega3},
      3, 4, 4, 0, 0, 0, 0,
      8640, {1.0}, 0.69444382300115026});
  }

  // Example 3
  {
    irrep Gamma1_ad(0,0,0, 2,0,0,0), Gamma1(0,0,0, 3,1,1,0),
          Gamma2(1,1,0, 1,1,1,0);
    SingleShellTensor tadat1(0,2,1,{Gamma1_ad},{},Gamma1,1,1),
                      tadat2(1,0,1,{},{},Gamma2,1,1);
    OmegaElement Omega1(1, {1,1,0}, 1, {3,1,0,0});
    examples.push_back({"Example 3: shells 0, 1; LL = SS = 2",
      {tadat1, tadat2}, {Omega1}, 1, 2, 2, 0, 0, 0, 0,
      100, {1.0}, -1.3772655761187389});
  }

  // Example 4
  {
    irrep Gamma1(0,0,0, 1,1,0,0), Gamma2(2,2,0, 1,1,0,0);
    SingleShellTensor tadat1(0,2,0,{Gamma1},{},Gamma1,1,1),
                      tadat2(1,0,2,{},{Gamma2},Gamma2,1,1);
    OmegaElement Omega1(1, {2,2,0}, 1, {0,0,0,0});
    examples.push_back({"Example 4: shells 0, 1; scalar",
      {tadat1, tadat2}, {Omega1}, 1, 0, 0, 0, 0, 0, 0,
      216, {1.0}, 0.011764841311701815});
  }

  // Example 5
  {
    irrep Gamma1(2,1,0, 0,0,0,0), Gamma2_ad(3,1,0, 1,1,0,0),
          Gamma2_at(3,2,0, 1,1,0,0), Gamma2(2,1,0, 0,0,0,0);
    SingleShellTensor tadat1(1,1,1,{},{},Gamma1,1,1),
                      tadat2(2,2,2,{Gamma2_ad},{Gamma2_at},Gamma2,2,1);
    OmegaElement Omega1(2, {2,1,0}, 1, {0,0,0,0});
    examples.push_back({"Example 5: shells 1, 2; LL = JJ = 2; rho = 2",
      {tadat1, tadat2}, {Omega1}, 2, 2, 0, 0, 0, 2, 0,
      14503296,
      {1.0, 1.0, 1.0, 1.0},
      -1.6334760649952949});
  }

  // Example 6
  {
    irrep Gamma(2,1,0, 0,0,0,0);
    SingleShellTensor tadat1(1,1,1,{},{},Gamma,1,1),
                      tadat2(2,1,1,{},{},Gamma,1,1),
                      tadat3(3,1,1,{},{},Gamma,1,1);
    OmegaElement Omega(2, {2,1,0}, 1, {0,0,0,0});
    examples.push_back({"Example 6: shells 1, 2, 3; LL = JJ = 2",
      {tadat1, tadat2, tadat3}, {Omega, Omega}, 3, 2, 0, 0, 0, 2, 0,
      6211776, {1.0, 1.0, 1.0, 1.0}, 0.17100599212792358});
  }

  // Example 7
  {
    irrep Gamma1_ad(2,0,0, 1,1,0,0), Gamma1_at(2,2,0, 1,1,0,0),
          Gamma(2,1,0, 0,0,0,0);
    SingleShellTensor tadat1(1,2,2,{Gamma1_ad},{Gamma1_at},Gamma,1,1),
                      tadat2(3,1,1,{},{},Gamma,1,1);
    OmegaElement Omega(2, {2,1,0}, 1, {0,0,0,0});
    examples.push_back({"Example 7: shells 1, 3; LL = JJ = 2",
      {tadat1, tadat2}, {Omega}, 3, 2, 0, 0, 0, 2, 0,
      4473248, {1.0, 1.0}, 0.15809199130500257});
  }

  test::Suite suite;
  for (const auto& x : examples)
    suite.run(x.name + ": terms, normalization, checksum",
              [&] { expand_and_check(x); });

  // the first run wrote the single-shell pieces to disk; a fresh database
  // reads them back and must reproduce Example 1 exactly
  suite.run("Example 1 again, single-shell pieces read back from disk", [&]
  {
    expand_and_check(examples[0]);
  });

  sam::finalize();
  return suite.result();
}
