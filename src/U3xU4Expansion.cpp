#include <iostream>

#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  std::ostream& operator<<(std::ostream& s, const Expansion& e)
  {
    // One line per term: the coefficients, then the operators (ad_phys/at_phys
    // are materialized here, purely for a human-readable dump).
    for (std::size_t r = 0; r < e.rows(); ++r)
    {
      const double* c = e.coeff_row(r);
      for (uint8_t j = 0; j < e.coeffs.ncols; ++j) s << c[j] << " ";
      s << "\t";
      for (uint8_t k = 0; k < e.nops; ++k)
      {
        if (e.is_ad(r, k)) s << ad_phys(e.index(r, k), *e.labels) << " ";
        else               s << at_phys(e.index(r, k), *e.labels) << " ";
      }
      s << "\n";
    }
    return s;
  }
} // namespace U3xU4
