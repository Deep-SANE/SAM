#ifndef SAM_RME_H
#define SAM_RME_H

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include <qcx_ND.h>

namespace sam
{
  class RME
  {
    std::vector<double> rmes_;
    qcx_utils::NDIndex ndi_;

  public:
    RME()
    {
      ndi_.add_dimensions(0, 0, 0, 0, 0);

      assert(rmes_.size() == ndi_.cardinality());
    }

    RME(std::vector<double>&& rmes, size_t afmax, size_t aimax, size_t aTmax,
        size_t rhomax, size_t etamax)
        : rmes_(std::move(rmes))
    {
      ndi_.add_dimensions(afmax, aimax, aTmax, rhomax, etamax);

      assert(rmes_.size() == ndi_.cardinality());
    }

    size_t size() const { return rmes_.size(); }

    size_t afmax() const { return ndi_.dimension(0); }
    size_t aimax() const { return ndi_.dimension(1); }
    size_t aTmax() const { return ndi_.dimension(2); }
    size_t rhomax() const { return ndi_.dimension(3); }
    size_t etamax() const { return ndi_.dimension(4); }

    double operator[](size_t i) const
    {
      assert(i < rmes_.size());
      return rmes_[i];
    }

    double get(size_t af, size_t ai, size_t aT, size_t rho, size_t eta) const 
    {
      std::array multi_index = {af, ai, aT, rho, eta};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }
  };

} // namespace sam

#endif
