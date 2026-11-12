#pragma once

#include <cmath>
#include <cstddef>

#include <U3xU4Expansion.h>

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
}
