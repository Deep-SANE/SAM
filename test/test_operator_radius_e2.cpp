// Proton- and neutron-only one-body operators with JT::Operator::matrix_CSR:
// Radius_p, Radius_n (sum of r^2 over protons / neutrons) and E2 (protons
// only). Their matrix elements carry (1 -+ tau_z)/2, i.e. an isoscalar and an
// isovector part; the test checks that both survive the expansion.
//
// Exact checks, with b_p^2, b_n^2 the oscillator lengths SAM uses for protons
// and neutrons (r^2 conserving HO quanta is b^2 (N + 3/2) per nucleon):
//   - Nmax = 0, every state: Radius_p = b_p^2 sum_protons (N + 3/2), Radius_n
//     the same over neutrons, Radius = b_p^2 sum_all (N + 3/2):
//       4He: 3 b_p^2, 3 b_n^2;  6Li: 5.5 b_p^2, 5.5 b_n^2;
//       6He: 3 b_p^2, 8 b_n^2   (the two extra neutrons are in the p shell)
//   - 6He at Nmax = 2: Radius_p + (b_p^2/b_n^2) Radius_n = Radius element by
//     element, while Radius_p and (b_p^2/b_n^2) Radius_n differ (isovector);
//   - E2 from J = 0 to J = 2 in 6He at Nmax = 0 vanishes exactly (the protons
//     fill the s shell), but not in 6Li from J = 1 to J = 3 (a p-shell
//     proton).
// Before the isovector part was kept, Radius_p and Radius_n were both half of
// Radius, and the 6He E2 came from the neutrons.
//
// The expansions are made by the test itself, in the temporary data
// directory.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <sam.h>

#include <HObasics.h>
#include <OperatorJT.h>

#include "test_helpers.h"

namespace
{
  constexpr int hw = 20;
  const double bp2 = HO::oscilatorLength2(hw, -1), // proton
               bn2 = HO::oscilatorLength2(hw, 1);  // neutron

  using Entries = std::map<std::pair<long long, long long>, double>;

  void write_expansion(const std::string& name, int maxShell)
  {
    HO::Table_nljmjmt HO_labels(maxShell);
    U3xU4::Table_gelfand gelfand_labels(maxShell);
    JT::OneBodyOperator op(name, hw, maxShell, 0, maxShell, HO_labels,
                           gelfand_labels);
    std::filesystem::create_directories(op.dirname() + "/Expansion");
    op.writeBinary(op.dirname() + "/Expansion/" + name + "_hw20",
                   op.expand_new());
  }

  Entries entries(const utils::CSRMatrix& m)
  {
    Entries e;
    const auto& row_ptr = m.get_row_ptr();
    const auto& col_idx = m.get_col_idx();
    const auto& v = m.get_values();
    for (long long i = 0; i < m.nrows(); i++)
      for (auto k = row_ptr[i]; k < row_ptr[i + 1]; k++)
        e[{i, static_cast<long long>(col_idx[k])}] += v[k];
    return e;
  }

  // the matrix of a one-body operator in the basis of a nucleus
  Entries matrix(const std::string& name, int Z, int N, int JJ, int Nmax)
  {
    U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);
    basis.construct();
    HO::Table_nljmjmt HO_labels(basis.maxShell());
    U3xU4::Table_gelfand gelfand_labels(basis.maxShell());
    JT::Operator op(name, N, Z, hw, Nmax, HO_labels, gelfand_labels);
    return entries(op.matrix_CSR(basis));
  }

  // value * identity, with dim states
  void require_multiple_of_identity(const Entries& e, size_t dim,
                                    double value, const std::string& what)
  {
    size_t diagonal = 0;
    for (const auto& [ij, v] : e)
      if (ij.first == ij.second)
      {
        test::near(v, value, what + " diagonal", 1e-10, 1e-12);
        diagonal++;
      }
      else
        test::near(v, 0.0, what + " off-diagonal", 1e-10, 0);
    test::require(diagonal == dim, what + ": every state on the diagonal");
  }

  size_t dimension(int Z, int N, int JJ, int Nmax)
  {
    U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);
    basis.construct();
    return basis.dim();
  }

  // largest |element| of the E2 matrix from J_i to J_f
  double max_E2(int Z, int N, int JJf, int JJi, int Nmax)
  {
    U3xU4::ManyNucleonBasis bras(N, Z, Nmax, JJf), kets(N, Z, Nmax, JJi);
    bras.construct();
    kets.construct();
    int maxShell = std::max(bras.maxShell(), kets.maxShell());
    HO::Table_nljmjmt HO_labels(maxShell);
    U3xU4::Table_gelfand gelfand_labels(maxShell);
    JT::Operator op("E2", N, Z, hw, Nmax, HO_labels, gelfand_labels);
    double largest = 0;
    for (const auto& [ij, v] : entries(op.matrix_CSR(bras, kets)))
      largest = std::max(largest, std::fabs(v));
    return largest;
  }
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  test::Suite suite;

  // one-body maxShell = valence shell + Nmax = 3 covers every case below
  for (const std::string name : {"Radius", "Radius_p", "Radius_n", "E2"})
    write_expansion(name, 3);

  struct Closed
  {
    std::string name;
    int Z, N, JJ;
    double p, n;   // sum over protons / neutrons of (N + 3/2)
  };
  for (const auto& x : std::vector<Closed>{{"4He, J = 0", 2, 2, 0, 3.0, 3.0},
                                           {"6Li, J = 1", 3, 3, 2, 5.5, 5.5},
                                           {"6He, J = 0", 2, 4, 0, 3.0, 8.0}})
    suite.run("Radius_p, Radius_n, Radius: " + x.name + ", Nmax = 0, exact",
              [&]
    {
      size_t dim = dimension(x.Z, x.N, x.JJ, 0);
      require_multiple_of_identity(matrix("Radius_p", x.Z, x.N, x.JJ, 0),
                                   dim, x.p * bp2, "Radius_p");
      require_multiple_of_identity(matrix("Radius_n", x.Z, x.N, x.JJ, 0),
                                   dim, x.n * bn2, "Radius_n");
      require_multiple_of_identity(matrix("Radius", x.Z, x.N, x.JJ, 0),
                                   dim, (x.p + x.n) * bp2, "Radius");
    });

  suite.run("6He, J = 0, Nmax = 2: Radius_p + (b_p/b_n)^2 Radius_n = Radius, "
            "Radius_p != (b_p/b_n)^2 Radius_n", [&]
  {
    auto p = matrix("Radius_p", 2, 4, 0, 2), n = matrix("Radius_n", 2, 4, 0, 2),
         all = matrix("Radius", 2, 4, 0, 2);
    Entries sum = p, difference = p;
    for (const auto& [ij, v] : n)
    {
      sum[ij] += bp2 / bn2 * v;
      difference[ij] -= bp2 / bn2 * v;
    }
    for (const auto& [ij, v] : all) sum[ij] -= v;
    double largest_sum = 0, largest_difference = 0;
    for (const auto& [ij, v] : sum)
      largest_sum = std::max(largest_sum, std::fabs(v));
    for (const auto& [ij, v] : difference)
      largest_difference = std::max(largest_difference, std::fabs(v));
    test::near(largest_sum, 0.0, "Radius_p + Radius_n - Radius", 1e-10, 0);
    test::require(largest_difference > 1.0,
                  "Radius_p and Radius_n should differ in 6He");
  });

  suite.run("E2: 6He 0+ -> 2+ at Nmax = 0 vanishes (protons fill the s "
            "shell); 6Li 1+ -> 3+ does not", [&]
  {
    test::near(max_E2(2, 4, 4, 0, 0), 0.0, "6He E2", 1e-10, 0);
    test::require(max_E2(3, 3, 6, 2, 0) > 0.1, "6Li E2 should not vanish");
  });

  sam::finalize();
  return suite.result();
}
