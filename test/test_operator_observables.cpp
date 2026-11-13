// One-body observables with JT::Operator::matrix_CSR: radii (Radius,
// Radius_p, Radius_n), E2, M1, Fermi and Gamow-Teller. The proton-only and
// neutron-only operators carry (1 -+ tau_z)/2, M1 has different g factors for
// protons and neutrons, and Fermi and Gamow-Teller change protons into
// neutrons or back; the checks below test that each keeps its isoscalar and
// isovector parts with the right weights.
//
// Exact checks, with b_p^2, b_n^2 the oscillator lengths SAM uses for protons
// and neutrons (r^2 conserving HO quanta is b^2 (N + 3/2) per nucleon):
//   - NumberOp (total HO quanta) in 6Li, J = 1, Nmax = 2: diagonal, equal to
//     N_0 = 2 on as many states as the Nmax = 0 space has, and 4 on the rest.
//   - CoMNumberOp (one-body + two-body parts, 1/A each) in 6Li, J = 1: the
//     Nmax truncation keeps the center-of-mass motion exactly separable, so
//     the matrix vanishes at Nmax = 0, and at Nmax = 2 its eigenvalues are
//     0, 1 and 2 only: M (M - 1) (M - 2) = 0, with spurious states present
//     (trace > 0).
//   - Radii, Nmax = 0, every state: Radius_p = b_p^2 sum_protons (N + 3/2),
//     Radius_n the same over neutrons, Radius = b_p^2 sum_all (N + 3/2):
//       4He: 3 b_p^2, 3 b_n^2;  6Li: 5.5 b_p^2, 5.5 b_n^2;
//       6He: 3 b_p^2, 8 b_n^2   (the two extra neutrons are in the p shell)
//   - 6He at Nmax = 2: Radius_p + (b_p^2/b_n^2) Radius_n = Radius element by
//     element, while Radius_p and (b_p^2/b_n^2) Radius_n differ (isovector);
//   - E2 from J = 0 to J = 2 in 6He at Nmax = 0 vanishes exactly (the protons
//     fill the s shell), but not in 6Li from J = 1 to J = 3 (a p-shell
//     proton).
//   - M1 = sqrt(3/4pi) sum_i (g_l l_i + g_s s_i), in 6Li (N = Z, J = 1) at
//     Nmax = 0 and 2. Between states of equal isospin T only the isoscalar
//     part acts (<T 0; 1 0|T 0> = 0), and L and S are generators, so the
//     matrix is diagonal there, with
//       sqrt(3/4pi) [ (g_l^p + g_l^n)/2 <L_z> + (g_s^p + g_s^n)/2 <S_z> ],
//       <L_z> = [J(J+1) + L(L+1) - S(S+1)] / 2(J+1),  <S_z> = J - <L_z>
//     in the M = J state. The isovector part connects T and T +- 1 and must
//     be there (nonzero at Nmax = 0).
//   - M1 in 6He and 6Be (J = 1) at Nmax = 0: one kind of nucleon fills the s
//     shell and contributes nothing, so the same formula holds with the
//     neutron g factors in 6He and the proton ones in 6Be (catches g factors
//     given to the wrong nucleon, which the N = Z case cannot see).
//   - Fermi, 6He (J = 0) -> 6Li and 6H, Nmax = 0 and 2: FermiMinus = T_-
//     takes each 6He state (isospin T, T_z = 1) to the same state in 6Li with
//     weight T(T+1); FermiPlus = T_+ to 6H with T(T+1) - 2. Every column has
//     at most one nonzero element, between states with the same L, S and T.
//     Hence the Fermi sum rule S_- - S_+ = N - Z = 2 for every 6He state.
//   - Gamow-Teller sigma t_-+, 6He (J = 0) -> 6Li and 6H (J = 1), Nmax = 0
//     and 2: the Ikeda sum rule S_- - S_+ = 3 (N - Z) for every 6He state.
//     The matrix holds the M = 0, mu = 0 elements, one third of the sum over
//     mu, so sum_f |GT-|^2 - sum_f |GT+|^2 = N - Z = 2 per column.
//   - FermiPlus and GamowTellerPlus from 6Li to 6He are the transposes of
//     FermiMinus and GamowTellerMinus from 6He to 6Li.
// Before the isovector part was kept, Radius_p and Radius_n were both half of
// Radius, and the 6He E2 came from the neutrons.
//
// The expansions are made by the test itself, in the temporary data
// directory.
//
// Basis order: within irrep alpha, the states run over the U(3)xU(4)
// multiplicity a (outermost), then the (kappa, L, zeta, S, T) labels of
// basis.decomposition(), then kappa and zeta, as in the matrix-element code.
// matrix_CSR(bras, kets) of a rank-J0 operator holds
// <J_f M| T^J0_0 |J_i M> with M = min(J_f, J_i).

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <sam.h>

#include <GlobalConstants.h>
#include <HObasics.h>
#include <OperatorJT.h>
#include <TwoBodyOperatorJT.h>

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

  // (L, S, T) of every basis state, doubled, in basis order
  struct Labels
  {
    int LL, SS, TT;
  };

  std::vector<Labels> labels(const U3xU4::ManyNucleonBasis& basis)
  {
    std::vector<Labels> v;
    auto irreps = basis.irreps();
    const auto& decomposition = basis.decomposition();
    for (size_t id = 0; id < irreps.size(); id++)
      for (size_t a = 0; a < irreps[id].alphamax(); a++)
        for (const auto& [kmax, LL, zmax, SS, TT, offset] : decomposition[id])
          for (size_t kz = 0; kz < size_t(kmax) * zmax; kz++)
            v.push_back({LL, SS, TT});
    return v;
  }

  // a nucleus (Z, N) with total angular momentum JJ/2 at Nmax
  struct Nucleus
  {
    int Z, N, JJ;
  };

  // the matrix of a one-body operator from kets to bras; also returns the
  // labels of both bases
  struct Rectangular
  {
    Entries e;
    std::vector<Labels> bras, kets;
  };

  Rectangular matrix(const std::string& name, const Nucleus& f,
                     const Nucleus& i, int Nmax)
  {
    U3xU4::ManyNucleonBasis bras(f.N, f.Z, Nmax, f.JJ),
                            kets(i.N, i.Z, Nmax, i.JJ);
    bras.construct();
    kets.construct();
    int maxShell = std::max(bras.maxShell(), kets.maxShell());
    HO::Table_nljmjmt HO_labels(maxShell);
    U3xU4::Table_gelfand gelfand_labels(maxShell);
    JT::Operator op(name, i.N, i.Z, hw, Nmax, HO_labels, gelfand_labels);
    auto m = op.matrix_CSR(bras, kets);
    test::require(m.nrows() == bras.dim(), name + ": rows = bra dimension");
    return {entries(m), labels(bras), labels(kets)};
  }

  // sum over bras of |element|^2, for every ket
  std::vector<double> column_norms(const Rectangular& m)
  {
    std::vector<double> norms(m.kets.size(), 0.0);
    for (const auto& [ij, v] : m.e) norms[ij.second] += v * v;
    return norms;
  }

  // sqrt(3/4pi) (g_l <L_z> + g_s <S_z>) in the M = J state with good L, S, J
  double M1_exact(int JJ, const Labels& x, double gl, double gs)
  {
    double J = JJ / 2.0, L = x.LL / 2.0, S = x.SS / 2.0,
           Lz = (J * (J + 1) + L * (L + 1) - S * (S + 1)) / (2 * (J + 1));
    return std::sqrt(3.0 / (4.0 * PI)) * (gl * Lz + gs * (J - Lz));
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
  for (const std::string name : {"NumberOp", "Radius", "Radius_p", "Radius_n",
                                  "E2", "M1",
                                  "FermiMinus", "FermiPlus",
                                  "GamowTellerMinus", "GamowTellerPlus"})
    write_expansion(name, 3);

  {
    HO::Table_nljmjmt HO_labels(4);
    U3xU4::Table_gelfand gelfand_labels(4);
    JT::TwoBodyOperator op("CoMNumberOp", hw, 4, 0, 4, HO_labels,
                           gelfand_labels);
    std::filesystem::create_directories(op.dirname() + "/Expansion");
    op.writeBinary(op.dirname() + "/Expansion/CoMNumberOp_hw20",
                   op.expand_new());
    write_expansion("CoMNumberOp", 3);
  }

  suite.run("NumberOp and CoMNumberOp, 6Li, J = 1: N = 2 or 4 at Nmax = 2; "
            "N_cm zero at Nmax = 0, eigenvalues 0, 1, 2 at Nmax = 2", [&]
  {
    size_t dim0 = dimension(3, 3, 2, 0), dim2 = dimension(3, 3, 2, 2),
           n0 = 0, n2 = 0;
    for (const auto& [ij, v] : matrix("NumberOp", 3, 3, 2, 2))
      if (ij.first != ij.second)
        test::near(v, 0.0, "NumberOp off-diagonal", 1e-10, 0);
      else if (std::fabs(v - 2.0) < 1e-10)
        n0++;
      else
      {
        test::near(v, 4.0, "NumberOp diagonal, state " +
                   std::to_string(ij.first), 1e-10, 1e-12);
        n2++;
      }
    test::require(n0 == dim0, "states with N = 2: " + std::to_string(n0) +
                  ", expected the Nmax = 0 dimension " + std::to_string(dim0));
    test::require(n0 + n2 == dim2, "NumberOp: every state on the diagonal");

    for (const auto& [ij, v] : matrix("CoMNumberOp", 3, 3, 2, 0))
      test::near(v, 0.0, "N_cm at Nmax = 0", 1e-10, 0);

    // dense symmetric matrix (matrix_CSR stores one triangle)
    size_t n = dimension(3, 3, 2, 2);
    std::vector<double> M(n * n, 0.0);
    for (const auto& [ij, v] : matrix("CoMNumberOp", 3, 3, 2, 2))
      M[ij.first * n + ij.second] = M[ij.second * n + ij.first] = v;
    auto product = [n](const std::vector<double>& A,
                       const std::vector<double>& B)
    {
      std::vector<double> C(n * n, 0.0);
      for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < n; k++)
          for (size_t j = 0; j < n; j++)
            C[i * n + j] += A[i * n + k] * B[k * n + j];
      return C;
    };
    auto M1 = M, M2 = M;   // M - 1, M - 2
    double trace = 0;
    for (size_t i = 0; i < n; i++)
    {
      M1[i * n + i] -= 1;
      M2[i * n + i] -= 2;
      trace += M[i * n + i];
    }
    double largest = 0;
    for (double x : product(product(M, M1), M2))
      largest = std::max(largest, std::fabs(x));
    test::near(largest, 0.0, "M (M - 1) (M - 2)", 1e-9, 0);
    test::require(trace > 0.5, "spurious states expected at Nmax = 2");
  });

  const Nucleus He6{2, 4, 0}, He6_J1{2, 4, 2}, Be6_J1{4, 2, 2}, Li6_J0{3, 3, 0},
                Li6_J1{3, 3, 2}, H6_J0{1, 5, 0}, H6_J1{1, 5, 2};

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

  for (int Nmax : {0, 2})
    suite.run("M1, 6Li, J = 1, Nmax = " + std::to_string(Nmax) +
              ": isoscalar part diagonal and exact, isovector part changes T",
              [&]
    {
      auto m = matrix("M1", Li6_J1, Li6_J1, Nmax);
      const double gl = (gl_p + gl_n) / 2, gs = (gs_p + gs_n) / 2;
      std::vector<bool> seen(m.kets.size(), false);
      double largest_isovector = 0;
      for (const auto& [ij, v] : m.e)
      {
        const auto &f = m.bras[ij.first], &i = m.kets[ij.second];
        if (ij.first == ij.second)
        {
          test::near(v, M1_exact(Li6_J1.JJ, i, gl, gs), "M1 diagonal, state " +
                     std::to_string(ij.first), 1e-10, 1e-12);
          seen[ij.first] = true;
        }
        else if (f.TT == i.TT)
          test::near(v, 0.0, "M1 between states of equal T (" +
                     std::to_string(ij.first) + "," +
                     std::to_string(ij.second) + ")", 1e-10, 0);
        else
        {
          test::require(std::abs(f.TT - i.TT) == 2, "M1 changes T by 1 at most");
          largest_isovector = std::max(largest_isovector, std::fabs(v));
        }
      }
      for (size_t k = 0; k < seen.size(); k++)
        test::require(seen[k], "M1 diagonal, state " + std::to_string(k) +
                      " missing (cannot vanish: g_s > g_l > 0 and S, L >= 0)");
      test::require(largest_isovector > 0.1, "M1 isovector part missing");
    });

  suite.run("M1, 6He and 6Be, J = 1, Nmax = 0: only the p-shell neutrons "
            "(protons) act, with g_n (g_p)", [&]
  {
    for (const auto& [x, gl, gs, what] :
         std::vector<std::tuple<Nucleus, double, double, std::string>>{
           {He6_J1, gl_n, gs_n, "6He"}, {Be6_J1, gl_p, gs_p, "6Be"}})
    {
      auto m = matrix("M1", x, x, 0);
      std::vector<bool> seen(m.kets.size(), false);
      for (const auto& [ij, v] : m.e)
        if (ij.first == ij.second)
        {
          test::near(v, M1_exact(x.JJ, m.kets[ij.second], gl, gs), what +
                     " M1 diagonal, state " + std::to_string(ij.first), 1e-10,
                     1e-12);
          seen[ij.first] = true;
        }
        else
          test::near(v, 0.0, what + " M1 off-diagonal", 1e-10, 0);
      for (size_t k = 0; k < seen.size(); k++)
        test::require(seen[k] || std::fabs(M1_exact(x.JJ, m.kets[k], gl, gs)) <
                      1e-10, what + " M1 diagonal, state " +
                      std::to_string(k) + " missing");
      test::require(!seen.empty(), what + ": J = 1 states exist");
    }
  });

  for (int Nmax : {0, 2})
    suite.run("Fermi, 6He, J = 0, Nmax = " + std::to_string(Nmax) +
              ": T_- to 6Li with T(T+1), T_+ to 6H with T(T+1) - 2", [&]
    {
      for (bool minus : {true, false})
      {
        std::string name = minus ? "FermiMinus" : "FermiPlus";
        auto m = matrix(name, minus ? Li6_J0 : H6_J0, He6, Nmax);
        std::vector<int> nonzero(m.kets.size(), 0);
        for (const auto& [ij, v] : m.e)
          if (std::fabs(v) > 1e-10)
          {
            const auto &f = m.bras[ij.first], &i = m.kets[ij.second];
            test::require(f.LL == i.LL && f.SS == i.SS && f.TT == i.TT,
                          name + " keeps L, S and T");
            nonzero[ij.second]++;
          }
        auto norms = column_norms(m);
        for (size_t k = 0; k < norms.size(); k++)
        {
          double T = m.kets[k].TT / 2.0;
          test::require(nonzero[k] <= 1, name + ": one final state per state");
          test::near(norms[k], T * (T + 1) - (minus ? 0.0 : 2.0),
                     name + " weight, state " + std::to_string(k), 1e-10,
                     1e-12);
        }
      }
    });

  for (int Nmax : {0, 2})
    suite.run("Gamow-Teller, 6He, J = 0, Nmax = " + std::to_string(Nmax) +
              ": Ikeda sum rule for every state", [&]
    {
      auto minus = matrix("GamowTellerMinus", Li6_J1, He6, Nmax),
           plus = matrix("GamowTellerPlus", H6_J1, He6, Nmax);
      auto Sm = column_norms(minus), Sp = column_norms(plus);
      double largest = 0;
      for (size_t k = 0; k < Sm.size(); k++)
      {
        test::near(Sm[k] - Sp[k], 2.0, "S_- - S_+, state " +
                   std::to_string(k), 1e-10, 1e-12);
        largest = std::max(largest, Sp[k]);
      }
      if (Nmax == 0)  // the protons fill the s shell: no beta+ strength
        test::near(largest, 0.0, "6He GT+ at Nmax = 0", 1e-10, 0);
    });

  suite.run("FermiPlus, GamowTellerPlus 6Li -> 6He = transposes of "
            "FermiMinus, GamowTellerMinus 6He -> 6Li, Nmax = 2", [&]
  {
    for (const auto& [minus, plus, Li6] :
         std::vector<std::tuple<std::string, std::string, Nucleus>>{
           {"FermiMinus", "FermiPlus", Li6_J0},
           {"GamowTellerMinus", "GamowTellerPlus", Li6_J1}})
    {
      auto down = matrix(minus, Li6, He6, 2), up = matrix(plus, He6, Li6, 2);
      Entries difference = up.e;
      for (const auto& [ij, v] : down.e) difference[{ij.second, ij.first}] -= v;
      double largest = 0, size = 0;
      for (const auto& [ij, v] : difference)
        largest = std::max(largest, std::fabs(v));
      for (const auto& [ij, v] : down.e) size = std::max(size, std::fabs(v));
      test::near(largest, 0.0, plus + " - " + minus + "^T", 1e-10, 0);
      test::require(size > 0.1, minus + " should not vanish");
    }
  });

  sam::finalize();
  return suite.result();
}
