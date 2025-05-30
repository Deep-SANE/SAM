#pragma once

#include <boost/dynamic_bitset.hpp>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <U4basis.h>

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

    // Count number of nucleons being hopped over
    std::size_t num_hop(std::size_t r_i, std::size_t c_i, std::size_t r_f,
                        std::size_t c_f) const;

    // Return number of single-particle states
    std::size_t num_sps() const;

    // Print the determinant
    void print() const;

    // Compare two determinant
    bool operator==(const SlaterDeterminant& other) const;
  };

  /* Conversion from U(4) irrep to U(N) irrep such that they are conjugate.
     Each entry of the U(N) irrep cannot exceed 4, i.e., niN ∈ {0, 1, 2, 3, 4}.
     The algorithm is as follows:
     
      for i from 1 to N:
        if i <= n_{44} :
          n_{iN} = 4
        if n_{44} < i <= n_{34} :
          n_{iN} = 3
        if n_{34} < i <= n_{24} :
          n_{iN} = 2
        if n_{24} < i <= n_{14} :
          n_{iN} = 1
        if n_{14} < i :
          n_{iN} = 0

     To be compatible with untou3 library, a U(N) irrep will be represented by
     5 numbers, which are the numbers of 4, 3, 2, 1, 0 in the irrep.
  */

  std::vector<std::size_t> U4_to_UN(std::size_t N, const u4::irrep& p);


} // namespace UNxU4
