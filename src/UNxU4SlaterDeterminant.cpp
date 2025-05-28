#include <algorithm>
#include <iostream>

#include <UNxU4SlaterDeterminant.h>

namespace UNxU4
{

  SlaterDeterminant::SlaterDeterminant(std::size_t n) : bits(2 * (n + 1) * (n + 2))
  {
    N = (n + 1) * (n + 2) / 2;
  }

  void SlaterDeterminant::occupy(std::size_t r, std::size_t c)
  {
    check_index(r, c);
    bits.set(index(r, c));
  }

  void SlaterDeterminant::vacate(std::size_t r, std::size_t c)
  {
    check_index(r, c);
    bits.reset(index(r, c));
  }

  bool SlaterDeterminant::is_occupied(std::size_t r, std::size_t c) const
  {
    check_index(r, c);
    return bits[index(r, c)];
  }

  std::size_t SlaterDeterminant::occupation() const
  {
    return bits.count();
  }

  std::size_t SlaterDeterminant::num_hop(std::size_t r_i, std::size_t c_i, 
                                         std::size_t r_f, std::size_t c_f) const
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

  std::size_t SlaterDeterminant::num_sps() const
  {
    return bits.size();
  }

  void SlaterDeterminant::print() const
  {
    for (std::size_t i = 0; i < bits.size(); i++)
    {
        std::cout << bits[i];
        if ((i + 1) % N == 0)
            std::cout << std::endl;
    }
  }

  bool SlaterDeterminant::operator==(const SlaterDeterminant& other) const
  {
    return bits == other.bits;
  }

  std::size_t SlaterDeterminant::index(std::size_t r, std::size_t c) const
  {
    return r * N + c;
  }

  void SlaterDeterminant::check_index(std::size_t r, std::size_t c) const
  {
    if (r >= 4 || c >= N)
        throw std::out_of_range("Orbital index out of range");
  }

} // namespace UNxU4
