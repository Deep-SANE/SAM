// Test copy of app/tutorial/U3xU4SingleShellTensorsDecomposition_Phys.cpp: the same
// five examples, each expanded with SingleShellTensor_phys::expand() and
// checked against the numbers it produced.
//
// For each example we check
//   - the number of terms in the expansion (exact);
//   - the normalization: per multiplicity column, the sum of squared
//     coefficients is 1 (the expansion is a unitary change of basis);
//   - a signed checksum over all terms (expansion_checksum.h), which does
//     not depend on the order of the terms but changes with any coefficient,
//     sign or label, so it guards the full expansion.
// Example 3 (the scalar ad . a in shell n = 1) is also checked analytically:
// every one of its 12 terms has coefficient 1/sqrt(12).
//
// Reference numbers were produced by this code (SAM main, Nov 2026) and agree
// with the app.

#include <cmath>
#include <string>
#include <vector>

#include <sam.h>

#include <U3xU4irrep.h>
#include <U3xU4SingleShellTensors.h>

#include "expansion_checksum.h"
#include "test_helpers.h"

namespace
{
  struct Example
  {
    std::string name;
    int n, ads, ats, LL, SS, TT, MMT, JJ, MMJ;
    U3xU4::irreps Omega_ads, Omega_ats;
    U3xU4::irrep T1, T2, T;
    // expected:
    size_t terms;
    std::vector<double> norm;
    double checksum;
  };
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  using U3xU4::irrep;

  std::vector<Example> examples = {
    /* Example 1: n = 2
                 (3,2,0)(1,1,0,0)  (5,4,0)(1,0,0,0)
        [[at x at]             x at]
                                   kappa=1,L=1,SS=1,TT=1=MMT,JJ=3=MMJ
    */
    {"Example 1: [[at x at](3,2,0)(1,1,0,0) x at](5,4,0)(1,0,0,0), n = 2",
     2, 0, 3, 2, 1, 1, 1, 3, 3,
     {}, {irrep(3,2,0, 1,1,0,0), irrep(5,4,0, 1,0,0,0)},
     irrep(0,0,0, 0,0,0,0), irrep(5,4,0, 1,0,0,0), irrep(5,4,0, 1,0,0,0),
     1848, {1.0}, 0.0051524700533063031},

    /* Example 2: n = 5
                (7,3,0)(1,1,0,0)
        [ad x ad]
                kappa=2,L=5,SS=2,TT=0=MMT,JJ=10,MMJ=0
    */
    {"Example 2: [ad x ad](7,3,0)(1,1,0,0), n = 5",
     5, 2, 0, 10, 2, 0, 0, 10, 0,
     {irrep(7,3,0, 1,1,0,0)}, {},
     irrep(7,3,0, 1,1,0,0), irrep(0,0,0, 0,0,0,0), irrep(7,3,0, 1,1,0,0),
     216, {1.0, 1.0}, 0.020732010188121812},

    /* Example 3: n = 1
                (0,0,0)(0,0,0,0)
        [ad x at]
                kappa=1,L=0,SS=0,TT=0=MMT,JJ=0=MMJ
    */
    {"Example 3: [ad x at](0,0,0)(0,0,0,0), n = 1",
     1, 1, 1, 0, 0, 0, 0, 0, 0,
     {}, {},
     irrep(1,0,0, 1,0,0,0), irrep(1,1,0, 1,1,1,0), irrep(0,0,0, 0,0,0,0),
     12, {1.0}, -0.60708854122050815},

    /* Example 4: n = 2
               (3,1,0)   (3,3,0)     (3,2,0)   (4,1,0)(1,0,0,0)
    { [[ad x ad]     x ad] x [at x at]         }
               (1,1,0,0) (2,1,0,0)   (1,1,0,0) kappa=1,L=1,SS=1,TT=1=MMT,JJ=1=MMJ
    */
    {"Example 4: {[[ad x ad] x ad] x [at x at]}(4,1,0)(1,0,0,0), n = 2",
     2, 3, 2, 2, 1, 1, 1, 1, 1,
     {irrep(3,1,0, 1,1,0,0), irrep(3,3,0, 2,1,0,0)}, {irrep(3,2,0, 1,1,0,0)},
     irrep(3,3,0, 2,1,0,0), irrep(3,2,0, 1,1,0,0), irrep(4,1,0, 1,0,0,0),
     5838212, {1.0}, 0.9460099548110642},

    /* Example 5: n = 0
               (0,0,0)          (0,0,0)   (0,0,0)(0,0,0,0)
    { [[ad x ad]      x [at x at]         }
               (1,1,0,0)        (1,1,0,0) kappa=1,L=0,SS=0,TT=0=MMT,JJ=0=MMJ
    */
    {"Example 5: {[ad x ad] x [at x at]}(0,0,0)(0,0,0,0), n = 0",
     0, 2, 2, 0, 0, 0, 0, 0, 0,
     {irrep(0,0,0, 1,1,0,0)}, {irrep(0,0,0, 1,1,0,0)},
     irrep(0,0,0, 1,1,0,0), irrep(0,0,0, 1,1,0,0), irrep(0,0,0, 0,0,0,0),
     48, {1.0}, -0.0074227655275087567},
  };

  test::Suite suite;
  for (const auto& x : examples)
    suite.run(x.name + ": terms, normalization, checksum", [&]
    {
      HO::Table_nljmjmt labels(x.n);
      auto [rhomax, etamax] = U3xU4::RhoEta(x.T1, x.T2, x.T);
      U3xU4::SingleShellTensor_phys tensor(x.n, x.ads, x.ats, x.Omega_ads,
        x.Omega_ats, x.T, rhomax, etamax, x.LL, x.SS, x.TT, x.MMT, x.JJ,
        x.MMJ, labels);
      auto e = tensor.expand();

      test::require(e.rows() == x.terms, "number of terms is " +
                    std::to_string(e.rows()) + ", expected " +
                    std::to_string(x.terms));
      auto norm = e.coeffs.norm();
      test::require(norm.size() == x.norm.size(), "number of columns");
      for (size_t c = 0; c < norm.size(); c++)
        test::near(norm[c], x.norm[c], "normalization", 1e-10, 0);
      test::near(test::checksum(e), x.checksum, "checksum", 1e-10, 1e-8);

      // analytic: the scalar ad . a in shell n = 1 has 12 equal terms
      if (x.name.rfind("Example 3", 0) == 0)
        for (size_t r = 0; r < e.rows(); r++)
          test::near(e.coeff_row(r)[0], 1.0 / std::sqrt(12.0),
                     "coefficient 1/sqrt(12)", 1e-12, 0);
    });

  sam::finalize();
  return suite.result();
}
