#ifndef SAM_U4XSU3_IRREP_H
#define SAM_U4XSU3_IRREP_H

#include <U3irrep.h>
#include <U4irrep.h>

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

    bool vacuum() const { return u4_.zero() && su3_.zero(); }
  };
} // namespace U4xSU3

#endif
