#include <algorithm>
#include <iostream>
#include <vector>

#include <UNxU4SlaterDeterminant.h>

namespace UNxU4
{

  SlaterDeterminant::SlaterDeterminant(std::size_t n)
      : bits(2 * (n + 1) * (n + 2))
  {
    N = (n + 1) * (n + 2) / 2;
  }

  std::size_t SlaterDeterminant::index(std::size_t r, std::size_t c) const
  {
    return r * N + c;
  }

  std::pair<std::size_t, std::size_t>
  SlaterDeterminant::invert_index(std::size_t i) const
  {
    return std::make_pair(i / N, i % N);
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

  std::size_t SlaterDeterminant::occupation() const { return bits.count(); }

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

  std::size_t SlaterDeterminant::num_sps() const { return bits.size(); }

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

  void SlaterDeterminant::check_index(std::size_t r, std::size_t c) const
  {
    if (r >= 4 || c >= N)
      throw std::out_of_range("Orbital index out of range");
  }

  std::vector<std::size_t> U4_to_UN(std::size_t N, const u4::irrep& p)
  {
    //std::vector<std::size_t> x(N, 0);  
    std::vector<std::size_t> y(5, 0);

    for (std::size_t i = 1; i <= N; i++)
    {
      if (i <= p.n44())
      {
        y[0]++;
        // x[i] = 4;
      }
      else if (p.n44() < i && i <= p.n34())
      {
        y[1]++;
        // x[i] = 3;
      }
      else if (p.n34() < i && i <= p.n24())
      {
        y[2]++;
        // x[i] = 2;
      }
      else if (p.n24() < i && i <= p.n14())
      {
        y[3]++;
        // x[i] = 1;
      }
      else if (p.n14() < i)
      {
        y[4]++;
        // x[i] = 0;
      }
    }
    return y;
  }

} // namespace UNxU4
