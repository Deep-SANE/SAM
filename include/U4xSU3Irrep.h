#ifndef SAM_U4XSU3_IRREP_H
#define SAM_U4XSU3_IRREP_H

#include <cassert>

#include <U3irrep.h>
#include <U4irrep.h>
#include <U3xU4Irrep.h>

namespace U4xSU3
{
  class irrep
  {
    u4::irrep u4_;
    su3::irrep su3_;

  public:
    irrep() = default;

    irrep(const u4::irrep& u4, const su3::irrep& su3) : u4_(u4), su3_(su3) { }

    void u4(const u4::irrep& u4) { u4_ = u4; }
    void su3(const su3::irrep& su3) { su3_ = su3; }

    const u4::irrep& u4() const { return u4_; }
    const su3::irrep& su3() const { return su3_; }

    // comparison operator
    bool operator==(const irrep& other) const
    {
      return (su3_ == other.su3()) && (u4_ == other.u4());
    }

    // converting to U3xU4::irrep
    U3xU4::irrep toU3xU4(int shell_number) const
    {
      auto A = u4_.quanta();
      auto N = shell_number * A;

      int n33x3 = N - su3_.n13() - su3_.n23();
      assert(n33x3 % 3 == 0);
      int n33 = n33x3 / 3;

      u3::irrep u3_(su3_.n13() + n33, su3_.n23() + n33, n33);
      return U3xU4::irrep(u3_, u4_);
    }

    U3xU4::irrep toU3xU4() const
    {
      u3::irrep u3_(su3_);
      return U3xU4::irrep(u3_, u4_);
    }

    bool vacuum() const { return u4_.zero() && su3_.zero(); }
  };
} // namespace U4xSU3

#endif
