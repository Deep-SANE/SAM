#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <boost/container_hash/hash.hpp>

#include <GlobalConstants.h>

namespace HO
{
  //////////// CLASS Table_nljmjmt /////////////

  // Assume n = 20 => max(ll) = 40, max(jj) = 41, 
  // min(mmj) = -41, max(mmj) = 41, mmt = +-1 
  // Thus these quantum numbers can be represented by int8: -128 to 127
  using qn_t = int8_t;

  // Pack the labels into a single 32-bit key (faster than hashing a 5-tuple).
  // n <= 44, ll = 2l <= 88, jj <= 89 fit 7 bits each; mmj in [-89,89] is biased
  // by +128 into 8 bits; mmt = +-1 is one bit. 30 bits total, all disjoint.
  inline uint32_t pack_mscheme_state(qn_t n, qn_t ll, qn_t jj, qn_t mmj, qn_t mmt)
  {
    return  (uint32_t(uint8_t(n))    & 0x7Fu)
          | ((uint32_t(uint8_t(ll))  & 0x7Fu) << 7)
          | ((uint32_t(uint8_t(jj))  & 0x7Fu) << 14)
          | ((uint32_t(uint8_t(mmj + 128)))   << 21)
          | ((mmt > 0 ? 1u : 0u)              << 29);
  }

  class Table_nljmjmt
  {
  private:
    // labels = (n, ll, jj, mmj, mmt)
    using key_type = std::tuple<qn_t, qn_t, qn_t, qn_t, qn_t>;

    // Each shell, dimension is 2(n+1)(n+2)
    // For n = 0 to 44: total number of single-particle states = 64860
    // Thus index can be uint16 (0 to 65,535)
    std::unordered_map<uint32_t, uint16_t> state_to_index;
    std::vector<key_type> index_to_state;

  public:
    // Constructor
    Table_nljmjmt() = default;
    Table_nljmjmt(int maxShell);

    size_t dim() const { return index_to_state.size(); }

    // Get index from quantum numbers
    uint16_t getIndex(qn_t n, qn_t ll, qn_t jj, qn_t mmj, qn_t mmt) const;

    // Get quantum numbers from index
    const key_type& getLabels(uint16_t index) const;
  };


  ////////////// STRUCT Orbit ///////////////

  struct Orbit
  {
    qn_t n_,   // nodal quantum number (!!!not shell number e)
         l_,   // orbital angular momentum
         jj_,  // twice total angular momentum
         mmt_; // twice isospin projection: +1 for neutron, -1 for proton

    // Constructor
    Orbit() : n_(0), l_(0), jj_(0), mmt_(0) {}
    Orbit(qn_t n, qn_t l, qn_t jj, qn_t mmt)
      : n_(n), l_(l), jj_(jj), mmt_(mmt)
    { }

    bool operator==(const Orbit& other) const;
    bool operator<(const Orbit& other) const;
    qn_t shell() const { return 2 * n_ + l_; }
  };

  inline std::ostream& operator<<(std::ostream& s, const Orbit& o)
  {
    return s << static_cast<int>(o.n_) << "\t" 
             << static_cast<int>(o.l_) << "\t" 
             << static_cast<int>(o.jj_) << "\t" 
             << static_cast<int>(o.mmt_);
  }


  //////////// CLASS OrbitsMap //////////////

  class OrbitsMap
  {
    qn_t emax_; // e is the shell number

    // All orbits
    std::vector<Orbit> orbits;
    std::unordered_map<uint32_t, uint16_t> index_map;

    // Pack an orbit's (n, l, jj, mmt) into a single 32-bit key (faster than a
    // std::map<Orbit> tree): n, l, jj fit 7 bits each (n<=22, l<=44, jj=2j<=89);
    // mmt = +-1 is one bit.
    static uint32_t pack(qn_t n, qn_t l, qn_t jj, qn_t mmt)
    {
      return  (uint32_t(uint8_t(n))   & 0x7Fu)
            | ((uint32_t(uint8_t(l))   & 0x7Fu) << 7)
            | ((uint32_t(uint8_t(jj))  & 0x7Fu) << 14)
            | ((mmt > 0 ? 1u : 0u)              << 21);
    }

  public:
    // Constructor
    OrbitsMap() : emax_(0), orbits(), index_map() { }
    OrbitsMap(int emax, std::string style); // style: "NuHamil" or "IMSRG"

    size_t num_orbits() const { return orbits.size(); }

    // Get index of the orbit
    // Caution: n = (e - l)/2 is the nodal quantum number
    // whereas e is the shell number
    uint16_t getIndex(qn_t n, qn_t l, qn_t jj, qn_t mmt) const;

    // Get orbit
    const Orbit& getOrbit(int idx) const { return orbits[idx - 1]; }

    // Print out the list of orbits with there
    void print() const
    {
      for (size_t i = 1; i <= orbits.size(); i++)
        std::cout << i << "\t" << orbits[i - 1] << std::endl;
    }
  };
} // namespace HO
