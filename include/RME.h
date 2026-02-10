#ifndef SAM_RME_H
#define SAM_RME_H

#include <cassert>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include <qcx_ND.h>

namespace sam
{
  class RME
  {
    std::vector<double> rmes_;
    qcx_utils::NDIndex5 ndi_;

  public:
    RME() = default;

    RME(std::vector<double>&& rmes, size_t afmax, size_t aimax, size_t a0max,
        size_t rhomax, size_t etamax)
        : rmes_(std::move(rmes))
    {
      ndi_.add_dimensions(afmax, aimax, a0max, rhomax, etamax);

      assert(rmes_.size() == ndi_.cardinality());
    }

    size_t size() const { return rmes_.size(); }
     
    bool empty() const { return size() == 0; }

    size_t afmax() const { return ndi_.dimension(0); }
    size_t aimax() const { return ndi_.dimension(1); }
    size_t a0max() const { return ndi_.dimension(2); }
    size_t rhomax() const { return ndi_.dimension(3); }
    size_t etamax() const { return ndi_.dimension(4); }

    double operator[](size_t i) const
    {
      assert(i < rmes_.size());
      return rmes_[i];
    }

    double get(size_t af, size_t ai, size_t a0, size_t rho, size_t eta) const 
    {
      std::array multi_index = {af, ai, a0, rho, eta};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }

    const auto& get() const { return rmes_; }

    void phase(int ph)
    {
      assert((ph == 1) || (ph == -1));

      if (ph == -1)
        for (auto& rme : rmes_)
          rme *= -1.0;
    }

    void print() const
    {
      std::cout << "rmes: ";
      for (const auto rme : rmes_)
        std::cout << rme << " ";
      std::cout << "\n";
    }
  };

} // namespace sam

#endif
