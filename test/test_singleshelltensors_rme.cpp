// Test copy of app/U3xU4SingleShellTensorsRME.cpp: the same six examples of
// single-shell triple-reduced matrix elements
//   < f ||| T ||| i >,
// computed with SingleShellTensor::RME() and compared with the full list of
// values it produced (indices: alpha_f, alpha_i, rho0, eta0, rho, eta).
//
// Examples 3-6 are also exact. With N = 2(n+1)(n+2) the number of
// single-particle states in shell n:
//   Example 3:  < (n,0,0)(1,0,0,0) ||| ad ||| vacuum >          = 1
//   Example 4:  < vacuum ||| at ||| (n,0,0)(1,0,0,0) >          = sqrt(N)
//   Example 5:  < (n,0,0)(1,0,0,0) ||| [ad x at]^0 ||| same >   = 1/sqrt(N)
//   Example 6:  < alpha_f ||| identity ||| alpha_i >            = delta
// Examples 3-5 are checked for every shell n = 0..8, not only the one in the
// app.
//
// Reference numbers were produced by this code (SAM main, Nov 2026) and agree
// with the app.

#include <cmath>
#include <string>
#include <vector>

#include <sam.h>

#include <U3xU4irrep.h>
#include <U3xU4SingleShellTensors.h>

#include "test_helpers.h"

namespace
{
  struct Example
  {
    std::string name;
    int n, ads, ats;
    U3xU4::irreps Omega_ads, Omega_ats;
    U3xU4::irrep T;
    int rho0max, eta0max;
    U3xU4::GammaElement i, f;
    // expected:
    size_t rhomax, etamax;
    std::vector<double> rmes;
  };

  void check(const Example& x)
  {
    U3xU4::SingleShellTensor tensor(x.n, x.ads, x.ats, x.Omega_ads,
                                    x.Omega_ats, x.T, x.rho0max, x.eta0max);
    auto rmes = tensor.RME(x.f, x.i);
    test::require(rmes.rhomax() == x.rhomax && rmes.etamax() == x.etamax,
                  "multiplicities rho, eta");
    test::require(rmes.size() == x.rmes.size(), "number of RMEs is " +
                  std::to_string(rmes.size()) + ", expected " +
                  std::to_string(x.rmes.size()));
    for (size_t k = 0; k < rmes.size(); k++)
      test::near(rmes[k], x.rmes[k], "RME #" + std::to_string(k), 1e-12, 1e-11);
  }
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  using U3xU4::irrep;
  using U3xU4::GammaElement;

  auto rho_eta = [](const irrep& a, const irrep& b, const irrep& c)
  {
    auto [rho, eta] = U3xU4::RhoEta(a, b, c);
    return std::pair<int, int>(rho, eta);
  };

  /* Example 1:
    < alpha_i (4,3,1) (3,1,0,0) |||
                 (3,1,0)       (3,3,0)            (3,2,0)   (4,1,0)
      { [[ad x ad]         x ad]        x [at x at]        }
                 (1,1,0,0)     (2,1,0,0)          (1,1,0,0) (1,0,0,0)
                                        ||| alpha_f (5,3,2) (3,1,1,0) >
  */
  auto [rho1, eta1] = rho_eta(irrep(3,3,0, 2,1,0,0), irrep(3,2,0, 1,1,0,0),
                              irrep(4,1,0, 1,0,0,0));
  Example ex1{"Example 1: <(5,3,2)(3,1,1,0)||| {[[ad x ad] x ad] x [at x at]} "
              "|||(4,3,1)(3,1,0,0)>, n = 2",
    2, 3, 2,
    {irrep(3,1,0, 1,1,0,0), irrep(3,3,0, 2,1,0,0)}, {irrep(3,2,0, 1,1,0,0)},
    irrep(4,1,0, 1,0,0,0), rho1, eta1,
    GammaElement(2, {4,3,1}, {3,1,0,0}), GammaElement(2, {5,3,2}, {3,1,1,0}),
    2, 1, {0.0, 0.0, 0.0, 0.0}};

  /* Example 2:
    < alpha_i (14,2,0) (1,1,1,1) |||
                (6,2,0)             (6,4,0)   (4,2,0)
      { [ad x ad]         x [at x at]         }
                (1,1,0,0)           (1,1,0,0) (0,0,0,0)
                ||| alpha_f (14,2,0) (1,1,1,1) >
  */
  auto [rho2, eta2] = rho_eta(irrep(6,2,0, 1,1,0,0), irrep(6,4,0, 1,1,0,0),
                              irrep(4,2,0, 0,0,0,0));
  Example ex2{"Example 2: <(14,2,0)(1,1,1,1)||| {[ad x ad] x [at x at]}(4,2,0) "
              "|||(14,2,0)(1,1,1,1)>, n = 4",
    4, 2, 2,
    {irrep(6,2,0, 1,1,0,0)}, {irrep(6,4,0, 1,1,0,0)},
    irrep(4,2,0, 0,0,0,0), rho2, eta2,
    GammaElement(4, {14,2,0}, {1,1,1,1}), GammaElement(4, {14,2,0}, {1,1,1,1}),
    3, 1,
    {-0.49323864060396178, 0.83855973096388836, -0.4018780682035567,
      0.61361599251978094, 0.28371852348778948, -0.51408415217164438,
     -0.18138863917378673, -0.43772688259652698, -0.7148050060404415}};

  /* Example 6:
    < alpha_i (5,3,2) (3,1,1,0) ||| I ||| alpha_f (5,3,2) (3,1,1,0) >
  */
  Example ex6{"Example 6: <(5,3,2)(3,1,1,0)||| identity |||(5,3,2)(3,1,1,0)>, "
              "n = 2 (alpha_max = 2)",
    2, 0, 0, {}, {}, irrep(0,0,0, 0,0,0,0), 1, 1,
    GammaElement(2, {5,3,2}, {3,1,1,0}), GammaElement(2, {5,3,2}, {3,1,1,0}),
    1, 1, {1.0, 0.0, 0.0, 1.0}};

  test::Suite suite;
  for (const auto& x : {ex1, ex2, ex6})
    suite.run(x.name, [&] { check(x); });

  /* Examples 3-5, for every shell n = 0..8:
                              (n,0,0)
    < (n,0,0) (1,0,0,0) ||| ad         ||| (0,0,0) (0,0,0,0) >
                              (1,0,0,0)
                              (n,n,0)
    < (0,0,0) (0,0,0,0) ||| at         ||| (n,0,0) (1,0,0,0) >
                              (1,1,1,0)
                                    (0,0,0)
    < (n,0,0) (1,0,0,0) ||| [ad x at]         ||| (n,0,0) (1,0,0,0) >
                                    (0,0,0,0)
  */
  for (int n = 0; n <= 8; n++)
  {
    const double N = 2.0 * (n + 1) * (n + 2);
    GammaElement vacuum(n, {0,0,0}, {0,0,0,0}), one(n, {n,0,0}, {1,0,0,0});
    std::string shell = ", n = " + std::to_string(n);

    Example ex3{"Example 3: <(n,0,0)(1,0,0,0)||| ad ||| vacuum> = 1" + shell,
      n, 1, 0, {}, {}, irrep(n,0,0, 1,0,0,0), 1, 1, vacuum, one,
      1, 1, {1.0}};
    Example ex4{"Example 4: <vacuum ||| at |||(n,0,0)(1,0,0,0)> = sqrt(N)" +
                shell,
      n, 0, 1, {}, {}, irrep(n,n,0, 1,1,1,0), 1, 1, one, vacuum,
      1, 1, {std::sqrt(N)}};
    Example ex5{"Example 5: <(n,0,0)(1,0,0,0)||| [ad x at]^0 |||same> = "
                "1/sqrt(N)" + shell,
      n, 1, 1, {}, {}, irrep(0,0,0, 0,0,0,0), 1, 1, one, one,
      1, 1, {1.0 / std::sqrt(N)}};

    for (const auto& x : {ex3, ex4, ex5})
      suite.run(x.name, [&] { check(x); });
  }

  sam::finalize();
  return suite.result();
}
