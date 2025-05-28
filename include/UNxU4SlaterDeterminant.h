#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <stdexcept>

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
  private:
    boost::dynamic_bitset<> bits;
    std::size_t N;
    std::size_t index(std::size_t r, std::size_t c) const { return r * N + c; }

  public:
    SlaterDeterminant(std::size_t n) : bits(2 * (n + 1) * (n + 2))
    {
      N = (n + 1) * (n + 2) / 2; // shell degeneracy
    }

    // Occupy orbital
    void occupy(std::size_t r, std::size_t c)
    {
      check_index(r, c);
      bits.set(index(r, c));
    }

    // Vacate orbital
    void vacate(std::size_t r, std::size_t c)
    {
      check_index(r, c);
      bits.reset(index(r, c));
    }

    // Check if an orbital is occupied
    bool is_occupied(std::size_t r, std::size_t c) const
    {
      check_index(r, c);
      return bits[index(r, c)];
    }

    // Count number of nucleons
    std::size_t occupation() const { return bits.count(); }

    // Count number of nucleons being hopped
    std::size_t num_hop(std::size_t r_i, std::size_t c_i, std::size_t r_f,
                        std::size_t c_f) const
    {
      std::size_t i_i = index(r_i, c_i), i_f = index(r_f, c_f);
      if (i_i > i_f)
        std::swap(i_i, i_f);

      std::size_t count = 0;
      for (std::size_t k = i_i + 1; k < i_f; k++)
        if (bits[k])
          count++;
      return count;
    }

    // Return number of single-particle states
    std::size_t num_sps() const { return bits.size(); }

    // Print the determinant
    void print() const
    {
      for (std::size_t i = 0; i < bits.size(); i++)
      {
        std::cout << bits[i];
        if ((i + 1) % N == 0)
          std::cout << std::endl;
      }
    }

    // Equality comparison
    bool operator==(const SlaterDeterminant& other) const
    {
      return bits == other.bits;
    }

  private:
    // Private constructor for internal operations
    SlaterDeterminant(const boost::dynamic_bitset<>& b) : bits(b) { }

    void check_index(std::size_t r, std::size_t c) const
    {
      if (r >= 4 || c >= N)
        throw std::out_of_range("Orbital index out of range");
    }
  };

} // namespace UNxU4
