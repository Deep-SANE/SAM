// Matrices of L.L and L.S with JT::Operator::matrix_CSR() in the U(3)xU(4)
// basis (replaces the commented-out check in app/MatrixOperatorJT.cpp).
//
// The basis states carry good L, S and J, so both operators are diagonal
// with exact values
//   L.L = L(L+1),    L.S = [J(J+1) - L(L+1) - S(S+1)] / 2,
// for every nucleus, Nmax and J. Checked for 4He (J = 0) and 6He (J = 0) at
// Nmax = 2, and for 6Li (J = 1) at Nmax = 0 and 2: every diagonal element
// equals the exact value, and every off-diagonal element vanishes.
//
// The one- and two-body expansions of L.L and L.S are made by the test
// itself (expand_new() + writeBinary() into the temporary data directory), at
// the sizes these calculations need: one-body up to maxShell, two-body up to
// maxShell + valence shell (pairs of up to 2 v + Nmax quanta).
//
// Basis order: within irrep alpha, the states run over the U(3)xU(4)
// multiplicity a (outermost), then the (kappa, L, zeta, S, T) labels of
// basis.decomposition(), then kappa and zeta, as in the matrix-element code.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <sam.h>

#include <OperatorJT.h>

#include "test_helpers.h"

namespace
{
  constexpr int hw = 20;

  struct Nucleus
  {
    std::string name;
    int Z, N, JJ, Nmax;
  };

  // one- and two-body expansions of an operator, written where expand()
  // looks for them
  void write_expansions(const std::string& name, int maxShell1, int maxShell2)
  {
    int maxShell = std::max(maxShell1, maxShell2);
    HO::Table_nljmjmt HO_labels(maxShell);
    U3xU4::Table_gelfand gelfand_labels(maxShell);

    JT::OneBodyOperator op1(name, hw, maxShell1, 0, maxShell1, HO_labels,
                            gelfand_labels);
    std::filesystem::create_directories(op1.dirname() + "/Expansion");
    op1.writeBinary(op1.dirname() + "/Expansion/" + name + "_hw20",
                    op1.expand_new());

    JT::TwoBodyOperator op2(name, hw, maxShell2, 0, maxShell2, HO_labels,
                            gelfand_labels);
    std::filesystem::create_directories(op2.dirname() + "/Expansion");
    op2.writeBinary(op2.dirname() + "/Expansion/" + name + "_hw20",
                    op2.expand_new());
  }

  // exact diagonal of L.L or L.S, in basis order (see top of file)
  std::vector<double> exact(const std::string& name,
                            const U3xU4::ManyNucleonBasis& basis, int JJ)
  {
    std::vector<double> values;
    auto irreps = basis.irreps();
    const auto& labels = basis.decomposition();
    const double jj = JJ * (JJ + 2.0) / 4.0;
    for (size_t id = 0; id < irreps.size(); id++)
      for (size_t a = 0; a < irreps[id].alphamax(); a++)
        for (const auto& [kmax, LL, zmax, SS, TT, offset] : labels[id])
          for (size_t kz = 0; kz < size_t(kmax) * zmax; kz++)
          {
            double ll = LL * (LL + 2.0) / 4.0, ss = SS * (SS + 2.0) / 4.0;
            values.push_back(name == "L.L" ? ll : (jj - ll - ss) / 2.0);
          }
    return values;
  }

  void check(const std::string& name, const Nucleus& x)
  {
    U3xU4::ManyNucleonBasis basis(x.N, x.Z, x.Nmax, x.JJ);
    basis.construct();
    HO::Table_nljmjmt HO_labels(basis.maxShell());
    U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

    JT::Operator op(name, x.N, x.Z, hw, x.Nmax, HO_labels, gelfand_labels);
    auto mat = op.matrix_CSR(basis);

    auto values = exact(name, basis, x.JJ);
    test::require(values.size() == size_t(basis.dim()) &&
                  mat.nrows() == basis.dim(), "dimension");

    const auto& row_ptr = mat.get_row_ptr();
    const auto& col_idx = mat.get_col_idx();
    const auto& v = mat.get_values();
    std::vector<bool> seen(basis.dim(), false);
    for (long long i = 0; i < mat.nrows(); i++)
      for (auto k = row_ptr[i]; k < row_ptr[i + 1]; k++)
        if (static_cast<long long>(col_idx[k]) == i)
        {
          test::near(v[k], values[i], name + " diagonal, state " +
                     std::to_string(i), 1e-10, 1e-12);
          seen[i] = true;
        }
        else
          test::near(v[k], 0.0, name + " off-diagonal (" + std::to_string(i) +
                     "," + std::to_string(col_idx[k]) + ")", 1e-10, 0);

    // states missing from the matrix must have value 0
    for (size_t i = 0; i < seen.size(); i++)
      if (!seen[i])
        test::near(0.0, values[i], name + " missing diagonal, state " +
                   std::to_string(i), 1e-10, 0);
  }
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  test::Suite suite;

  std::vector<Nucleus> nuclei = {
    {"4He, J = 0, Nmax = 2", 2, 2, 0, 2},
    {"6Li, J = 1, Nmax = 0", 3, 3, 2, 0},
    {"6Li, J = 1, Nmax = 2", 3, 3, 2, 2},
    {"6He, J = 0, Nmax = 2", 2, 4, 0, 2},
  };

  // largest sizes needed: one-body maxShell = v + Nmax = 3 (6Li, 6He),
  // two-body maxShell + v = 4
  for (const std::string name : {"L.L", "L.S"})
    write_expansions(name, 3, 4);

  for (const std::string name : {"L.L", "L.S"})
    for (const auto& x : nuclei)
      suite.run(name + ", " + x.name + ": diagonal with exact values",
                [&] { check(name, x); });

  sam::finalize();
  return suite.result();
}
