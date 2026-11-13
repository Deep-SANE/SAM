// Test copy of app/tutorial/U3xU4SingleShellTensorsDecomposition_Gelfand.cpp: the same
// three examples, each expanded with SingleShellTensor_gelfand::expand() and
// checked against the numbers it produced.
//
// For each example we check
//   - the number of terms (exact);
//   - the normalization: per multiplicity column, the sum of squared
//     coefficients is 1;
//   - a signed checksum, sum_r w(r) c_r, where w(r) is a fixed function of
//     the Gelfand labels of the operators in term r (read from their printed
//     form). It does not depend on the order of the terms, but changes with
//     any coefficient, sign or label.
// Example 2 (the scalar ad . a in shell n = 1) is also checked analytically:
// every one of its 12 terms has coefficient 1/sqrt(12).
//
// Reference numbers were produced by this code (SAM main, Nov 2026) and agree
// with the app.

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <U3basis.h>
#include <U4basis.h>

#include <sam.h>

#include <U3xU4SingleShellTensors.h>

#include "test_helpers.h"

namespace
{
  struct Example
  {
    std::string name;
    int n, ads, ats;
    std::vector<U3xU4::irrep> Omega_ads, Omega_ats;
    U3xU4::irrep T1, T2, T;
    u3::gelfand g3;
    u4::gelfand g4;
    // expected:
    size_t terms;
    std::vector<double> norm;
    double checksum;
  };

  // fixed weight of one operator, from its printed Gelfand labels
  template <class Op>
  double weight(const Op& op, int position)
  {
    std::ostringstream s;
    s << op;
    double v = 0;
    int i = 0;
    for (unsigned char ch : s.str())
      v += (++i) * 0.013 * ch;
    return (position + 1.0) * v;
  }
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  using U3xU4::irrep;

  std::vector<Example> examples = {
    /* Example 1: n = 2
                    (3,1,0)       (3,3,0)             (3,2,0)    (4,1,0; 4,0; 2)
        { [[ad x ad]         x ad]          x [at x at]         }
                    (1,1,0,0)     (2,1,0,0)           (1,1,0,0)  (1,0,0,0; 1,0,0;
       1,0; 0)
    */
    {"Example 1: {[[ad x ad] x ad] x [at x at]}(4,1,0)(1,0,0,0), n = 2",
     2, 3, 2,
     {irrep(3,1,0, 1,1,0,0), irrep(3,3,0, 2,1,0,0)}, {irrep(3,2,0, 1,1,0,0)},
     irrep(3,3,0, 2,1,0,0), irrep(3,2,0, 1,1,0,0), irrep(4,1,0, 1,0,0,0),
     u3::gelfand(4,1,0, 4,0, 2), u4::gelfand(1,0,0,0, 1,0,0, 1,0, 0),
     24768, {1.0}, -0.00049021437328555795},

    /* Example 2: n = 1
          (1,0,0)       (1,1,0)    (0,0,0; 0,0; 0)
      { ad          x at          }
          (1,0,0,0)     (1,1,1,0)  (0,0,0,0; 0,0,0; 0,0; 0)
    */
    {"Example 2: {ad x at}(0,0,0)(0,0,0,0), n = 1",
     1, 1, 1,
     {}, {},
     irrep(1,0,0, 1,0,0,0), irrep(1,1,0, 1,1,1,0), irrep(0,0,0, 0,0,0,0),
     u3::gelfand(0,0,0, 0,0, 0), u4::gelfand(0,0,0,0, 0,0,0, 0,0, 0),
     12, {1.0}, -0.81405448381954482},

    /* Example 3: n = 3
          (3,0,0)              (3,0,0)   (6,0,0; 6,0; 3)
      { ad          x [at x at]         }
          (1,0,0,0)            (1,1,0,0) (2,1,0,0; 1,1,0; 1,0; 0)
    */
    {"Example 3: {ad x [at x at]}(6,0,0)(2,1,0,0), n = 3",
     3, 1, 2,
     {}, {irrep(3,0,0, 1,1,0,0)},
     irrep(3,0,0, 1,0,0,0), irrep(3,0,0, 1,1,0,0), irrep(6,0,0, 2,1,0,0),
     u3::gelfand(6,0,0, 6,0, 3), u4::gelfand(2,1,0,0, 1,1,0, 1,0, 0),
     120, {1.0}, -0.013832501524033399},
  };

  test::Suite suite;
  for (const auto& x : examples)
    suite.run(x.name + ": terms, normalization, checksum", [&]
    {
      // single-particle states of shell n and their conjugates
      u3::irrep sp3(x.n, 0, 0), sp3_conj(x.n, x.n, 0);
      auto sps3 = u3::gelfandBasis(sp3), sps3_conj = u3::gelfandBasis(sp3_conj);
      u4::irrep sp4(1, 0, 0, 0), sp4_conj(1, 1, 1, 0);
      auto sps4 = u4::gelfandBasis(sp4), sps4_conj = u4::gelfandBasis(sp4_conj);

      auto [rhomax, etamax] = U3xU4::RhoEta(x.T1, x.T2, x.T);
      U3xU4::SingleShellTensor_gelfand tensor(x.n, sps3, sps3_conj, sps4,
        sps4_conj, x.ads, x.ats, x.Omega_ads, x.Omega_ats, x.T, rhomax, etamax,
        x.g3, x.g4);
      auto [coeffs, adArr, aArr] = tensor.expand();

      test::require(coeffs.size() == x.terms, "number of terms is " +
                    std::to_string(coeffs.size()) + ", expected " +
                    std::to_string(x.terms));

      std::vector<double> norm(rhomax * etamax, 0.0);
      double sum = 0;
      for (size_t r = 0; r < coeffs.size(); r++)
      {
        test::require(coeffs[r].size() == int(norm.size()),
                      "number of columns");
        double w = 0;
        int k = 0;
        if (!adArr.empty())
          for (const auto& op : adArr[r]) w += weight(op, k++);
        if (!aArr.empty())
          for (const auto& op : aArr[r]) w += weight(op, k++);
        for (size_t c = 0; c < norm.size(); c++)
        {
          norm[c] += coeffs[r][c] * coeffs[r][c];
          sum += (c + 1.0) * std::sin(w) * coeffs[r][c];
        }
      }
      for (size_t c = 0; c < norm.size(); c++)
        test::near(norm[c], x.norm[c], "normalization", 1e-10, 0);
      test::near(sum, x.checksum, "checksum", 1e-10, 1e-8);

      // analytic: the scalar ad . a in shell n = 1 has 12 equal terms
      if (x.name.rfind("Example 2", 0) == 0)
        for (size_t r = 0; r < coeffs.size(); r++)
          test::near(coeffs[r][0], 1.0 / std::sqrt(12.0),
                     "coefficient 1/sqrt(12)", 1e-12, 0);
    });

  sam::finalize();
  return suite.result();
}
