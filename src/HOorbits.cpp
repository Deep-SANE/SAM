#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <HOorbits.h>

#include <qcx_convert.h>
#include <qcx_print.h>

namespace HO
{
  //////////// CLASS Table_nljmjmt /////////////

  // Constructor
  Table_nljmjmt::Table_nljmjmt(int maxShell)
  {
    assert(maxShell <= 44);
    uint16_t index = 0;
    for (qn_t n = 0; n <= maxShell; n++)
      for (qn_t l = n; l >= 0; l -= 2)
        for (qn_t jj = 2 * l + 1; jj >= std::abs(2 * l - 1); jj -= 2)
          for (qn_t mmj = jj; mmj >= -jj; mmj -= 2)
            for (qn_t mmt = 1; mmt >= -1; mmt -= 2)
            {
              key_type key = {n, 2 * l, jj, mmj, mmt};
              state_to_index[pack_mscheme_state(n, 2 * l, jj, mmj, mmt)] = index;
              index_to_state.push_back(key);
              index++;
            }
  }

  // Get index from quantum numbers
  uint16_t Table_nljmjmt::getIndex(qn_t n, qn_t ll,
                        qn_t jj, qn_t mmj, qn_t mmt) const
  {
    auto it = state_to_index.find(pack_mscheme_state(n, ll, jj, mmj, mmt));
    if (it == state_to_index.end())
      throw std::out_of_range("Labels not found");
    return it->second;
  }

  // Get quantum numbers from index
  const std::tuple<qn_t, qn_t, qn_t, qn_t, qn_t>&
  Table_nljmjmt::getLabels(uint16_t index) const
  {
    return index_to_state[index];
  }

  ////////////// CLASS Orbit ///////////////

  bool Orbit::operator==(const Orbit& other) const
  {
    return other.n_ == n_ && other.l_ == l_ && 
           other.jj_ == jj_ && other.mmt_ == mmt_;
  }

  bool Orbit::operator<(const Orbit& other) const
  {
    return std::tie(n_, l_, jj_, mmt_) <
           std::tie(other.n_, other.l_, other.jj_, other.mmt_);
  }

  //////////// CLASS OrbitsMap //////////////

  // Constructor
  OrbitsMap::OrbitsMap(int emax, std::string style)
  {
    assert(emax <= 44);
    // It does not take in lmax, it is set to default lmax = emax
    emax_ = qcx_utils::ito8(emax);

    if (style == "NuHamil")
    {
      uint16_t idx = 1;
      for (qn_t e = 0; e <= emax; ++e)
      {
        for (qn_t l = e % 2; l <= e; l += 2)
        {
          qn_t n = (e - l) / 2;
          for (qn_t jj = abs(2 * l - 1); jj <= 2 * l + 1; jj += 2)
          {
            // add proton
            orbits.push_back({n, l, jj, -1});
            index_map[pack(n, l, jj, -1)] = idx;
            idx++;

            // add neutron
            orbits.push_back({n, l, jj, 1});
            index_map[pack(n, l, jj, 1)] = idx;
            idx++;
          }
        }
      }
    }
    else if (style == "IMSRG")
    {
      uint16_t idx = 1;
      for (qn_t e = 0; e <= emax; ++e)
      {
        for (qn_t l = e; l >= 0; l -= 2)
        {
          qn_t n = (e - l) / 2;
          for (qn_t jj = 2 * l + 1; jj >= abs(2 * l - 1); jj -= 2)
          {
            // add proton
            orbits.push_back({n, l, jj, -1});
            index_map[pack(n, l, jj, -1)] = idx;
            idx++;

            // add neutron
            orbits.push_back({n, l, jj, 1});
            index_map[pack(n, l, jj, 1)] = idx;
            idx++;
          }
        }
      }
    }
    else
      throw std::runtime_error("Style of orbit map not known");
  }

  // Get index of the orbit
  // Caution: n = (e - l)/2 is the nodal quantum number
  // whereas e is the shell number
  uint16_t OrbitsMap::getIndex(qn_t n, qn_t l, qn_t jj, qn_t mmt) const
  {
    auto it = index_map.find(pack(n, l, jj, mmt));
    if (it != index_map.end())
      return it->second;
    else
      throw std::runtime_error("Orbit not found");
  }
} // namespace HO
