#pragma once

#include <boost/dynamic_bitset.hpp>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace UNxU4
{

  /* Slater Determinant for the n-th shell is represented by 4 bit strings,
     each has length of N = (n+1)(n+2)/2. 

     Each bit represents a single-particle U(N)xU(4) state and can be labeled  
     by two indices (r,c):
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
    boost::dynamic_bitset<> bits_;
    uint8_t N_; // shell degeneracy which is 231 for n=20

  public:
    // Constructor
    SlaterDeterminant(int n); // n is shell number
    SlaterDeterminant(const boost::dynamic_bitset<>& b);
    SlaterDeterminant(const SlaterDeterminant& other);

    // Getters
    const boost::dynamic_bitset<>& bits() const { return bits_; }
    size_t shell_degeneracy() const { return static_cast<size_t>(N_); }

    // Check if indices are out of range
    bool check_index(size_t r, size_t c) const;

    // Bit position
    size_t index(size_t r, size_t c) const;

    // Invert bit position
    std::pair<size_t, size_t> invert_index(size_t i) const;

    // Occupy orbital
    void occupy(size_t r, size_t c);

    // Vacate orbital
    void vacate(size_t r, size_t c);

    // Check if an orbital is occupied
    bool is_occupied(size_t r, size_t c) const;

    // Count number of nucleons
    size_t occupation() const;

    // Count number of all preceding occupied states
    size_t preceding_occupation(size_t r, size_t c) const;

    // Count number of nucleons being hopped over
    size_t num_hop(size_t r_i, size_t c_i, size_t r_f, size_t c_f) const;

    // Return number of single-particle states
    size_t num_sps() const;

    // Compare two determinant
    bool operator==(const SlaterDeterminant& other) const;

    // Print the determinant
    // void print() const;
    friend std::ostream& operator<<(std::ostream& s, const SlaterDeterminant& sd);
  };
  using SlaterDeterminants = std::vector<UNxU4::SlaterDeterminant>;

} // namespace UNxU4
