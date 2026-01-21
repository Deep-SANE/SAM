#pragma once

#include <U3xU4irrep.h>

namespace U3xU4
{
  class Table_gelfand
  {
  private:
    // Gelfand array for [n,0,0] for 0 <= n <= maxShell
    std::vector<u3::gelfands> sps3_;
    // Gelfand array for [n,n,0] for 0 <= n <= maxShell
    std::vector<u3::gelfands> sps3_conj_;

    // Gelfand array for [1,0,0,0]
    u4::gelfands sps4_;
    // Gelfand array for [1,1,1,0]
    u4::gelfands sps4_conj_;

  public:
    // Constructors
    Table_gelfand() = default;
    Table_gelfand(int maxShell)
    {
      for (int n = 0; n <= maxShell; n++)
      {
        u3::irrep x(n, 0, 0), x_conj(n, n, 0);
        u3::gelfands sps3 = u3::gelfandBasis(x),
                     sps3_conj = u3::gelfandBasis(x_conj);

        sps3_.push_back(std::move(sps3));
        sps3_conj_.push_back(std::move(sps3_conj));
      }

      u4::irrep y(1, 0, 0, 0), y_conj(1, 1, 1, 0);
      sps4_ = std::move(u4::gelfandBasis(y)),
      sps4_conj_ = std::move(u4::gelfandBasis(y_conj));
    }

    // Getters
    const auto& sps3(int shell) const { return sps3_[shell]; }
    const auto& sps3_conj(int shell) const { return sps3_conj_[shell]; }
    const auto& sps4() const { return sps4_; }
    const auto& sps4_conj() const {return sps4_conj_; }
  };
} // namespace U3xU4
