#pragma once

#include <cmath>
#include <cstddef>

#include <U3xU4Expansion.h>
#include <U3xU4MultiShellTensors.h>

namespace test
{
  // signed, order-independent checksum of an uncoupled expansion:
  //   sum_r sum_c (c+1) sin(x_r) coeff(r, c),
  // where x_r is a fixed function of the labels (n, l, j, mj, mt, ad or a)
  // of the operators in term r. It does not depend on the order of the
  // terms, but changes with any coefficient, sign or label.
  inline double checksum(const U3xU4::Expansion& e)
  {
    double sum = 0;
    for (std::size_t r = 0; r < e.rows(); r++)
    {
      double x = 0;
      for (std::size_t k = 0; k < e.nops; k++)
      {
        auto [n, ll, jj, mmj, mmt] = e.op_labels(r, k);
        x += (k + 1.0) * (0.31 * n + 0.17 * ll + 0.13 * jj + 0.07 * mmj +
                          0.05 * mmt + (e.is_ad(r, k) ? 0.5 : 0.0));
      }
      for (std::size_t c = 0; c < e.ncols(); c++)
        sum += (c + 1.0) * std::sin(x) * e.coeff_row(r)[c];
    }
    return sum;
  }

  // signed, order-independent checksum of an operator expansion (a list of
  // multi-shell tensors with their strengths):
  //   sum_t sum_(labels, a) sin(x_t + y) strength,
  // where x_t is a fixed function of tensor t's single-shell tensors (shell,
  // number of ad and a, U(3)xU(4) irrep) and couplings, and y of the
  // (kappa, LL, zeta, SS, TT) labels and the multiplicity index a of each
  // strength. Strengths enter linearly, so scaling all of them by a factor
  // scales the checksum by the same factor.
  inline double checksum(const U3xU4::MultiShellTensors_phys& tensors)
  {
    double sum = 0;
    for (const auto& t : tensors)
    {
      double x = 0;
      int i = 0;
      for (const auto& G : t.Gammas())
        x += (++i) * (0.31 * G.shell() + 0.17 * G.num_ads() +
                      0.13 * G.num_ats() + 0.07 * G.lam() + 0.05 * G.mu() +
                      0.03 * G.P1() + 0.02 * G.P2() + 0.011 * G.P3());
      int j = 0;
      for (const auto& O : t.Omegas())
        x += (++j) * (0.19 * O.lam() + 0.11 * O.mu() + 0.023 * O.P1() +
                      0.017 * O.P2() + 0.013 * O.P3());
      for (const auto& [k, LL, z, SS, TT, strengths] : t.LLSSTTstrengths())
        for (std::size_t a = 0; a < strengths.size(); a++)
          sum += std::sin(x + 0.11 * k + 0.07 * LL + 0.05 * z + 0.03 * SS +
                          0.02 * TT + 0.01 * a) * strengths[a];
    }
    return sum;
  }
}
