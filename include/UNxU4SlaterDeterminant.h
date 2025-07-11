#pragma once

#include <boost/dynamic_bitset.hpp>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace UNxU4
{

  /* Slater Determinant for the n-th shell is represented by 4 bit strings,
     whose length is N = (n+1)(n+2)/2. Each bit represents a single-particle
     U(N)xU(4) state and can be labeled  by two indices (r,c):
     - r stands for "row", corresponding to one of the four Gelfand pattern
       associated with u4::irrep(1,0,0,0)
     - c stands for "column", corresponding to one of the N Gelfand pattern
       associated with u3::irrep(n,0,0)

           c
           |
           v
      r -> - - - - - -
           - - - - - -
           - - - - - -
           - - - - - -

     For computational convenience, this representation is flatten into a
     single bit string that contains 4N = 2(n+1)(n+2) bits. As a matter of
     consistency, it is thought of as 4 blocks of N bits, following one
     another.

     The mapping between (rc) and index (i) in the flattened bit string is
     i = r * N + c  vs   r = [i / N] and c = i % N
  */

  class SlaterDeterminant
  {
  public:
    boost::dynamic_bitset<> bits;
    std::size_t N; // shell degeneracy

    void check_index(std::size_t r, std::size_t c) const;

    // Constructor
    SlaterDeterminant(std::size_t n);
    SlaterDeterminant(const boost::dynamic_bitset<>& b);
    SlaterDeterminant(const SlaterDeterminant& other)
        : bits(other.bits), N(other.N)
    { }

    // Bit position
    std::size_t index(std::size_t r, std::size_t c) const;

    // Invert bit position
    std::pair<std::size_t, std::size_t> invert_index(std::size_t i) const;

    // Occupy orbital
    void occupy(std::size_t r, std::size_t c);

    // Vacate orbital
    void vacate(std::size_t r, std::size_t c);

    // Check if an orbital is occupied
    bool is_occupied(std::size_t r, std::size_t c) const;

    // Count number of nucleons
    std::size_t occupation() const;

    // Count number of all preceding occupied states
    std::size_t preceding_occupation(std::size_t r, std::size_t c) const;

    // Count number of nucleons being hopped over
    std::size_t num_hop(std::size_t r_i, std::size_t c_i, std::size_t r_f,
                        std::size_t c_f) const;

    // Return number of single-particle states
    std::size_t num_sps() const;

    // Print the determinant
    // void print() const;
    friend std::ostream& operator<<(std::ostream& s,
                                    const SlaterDeterminant& sd)
    {
      for (std::size_t i = 0; i < sd.bits.size(); i++)
      {
        s << sd.bits[i];
        if ((i + 1) % sd.N == 0)
          s << std::endl;
      }
      return s;
    }

    // Compare two determinant
    bool operator==(const SlaterDeterminant& other) const;
  };

} // namespace UNxU4
