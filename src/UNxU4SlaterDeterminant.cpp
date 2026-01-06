#include <algorithm>
#include <iostream>
#include <vector>

#include <UNxU4SlaterDeterminant.h>

#include <qcx_convert.h>

namespace UNxU4
{

  SlaterDeterminant::SlaterDeterminant(int n)
    : bits_(2 * (n + 1) * (n + 2)), N_(qcx_utils::itou8((n + 1) * (n + 2) / 2))
  { }

  SlaterDeterminant::SlaterDeterminant(const boost::dynamic_bitset<>& b)
    : bits_(b), N_(qcx_utils::itou8(b.size() / 4))
  { }

  SlaterDeterminant::SlaterDeterminant(const SlaterDeterminant& other)
    : bits_(other.bits_), N_(other.N_)
  { }

  bool SlaterDeterminant::check_index(size_t r, size_t c) const
  {
    if (r >= 4 || c >= shell_degeneracy()) 
      return false;
    else 
      return true;
  }

  size_t SlaterDeterminant::index(size_t r, size_t c) const
  {
    if (check_index(r, c) == true)
      return r * shell_degeneracy() + c;
    else
      throw std::out_of_range("Orbital index out of range");
  }

  std::pair<size_t, size_t> SlaterDeterminant::invert_index(size_t i) const
  {
    if (i < bits_.size())
      return std::make_pair(i / shell_degeneracy(), i % shell_degeneracy());
    else
      throw std::out_of_range("Bit index out of range");
  }

  void SlaterDeterminant::occupy(size_t r, size_t c)
  {
    bits_.set(index(r, c));
  }

  void SlaterDeterminant::vacate(size_t r, size_t c)
  {
    bits_.reset(index(r, c));
  }

  bool SlaterDeterminant::is_occupied(size_t r, size_t c) const
  {
    return bits_[index(r, c)];
  }

  size_t SlaterDeterminant::occupation() const { return bits_.count(); }

  size_t SlaterDeterminant::preceding_occupation(size_t r, size_t c) const
  {
    size_t i = index(r, c), count = 0;
    for (size_t k = 0; k < i; k++)
      if (bits_[k]) count++;
    return count;
  }

  size_t SlaterDeterminant::num_hop(size_t r_i, size_t c_i, size_t r_f, size_t c_f) const
  {
    size_t i_i = index(r_i, c_i), i_f = index(r_f, c_f);
    if (i_i > i_f) std::swap(i_i, i_f);

    size_t count = 0;
    for (size_t k = i_i + 1; k < i_f; k++)
      if (bits_[k]) count++;
    return count;
  }

  size_t SlaterDeterminant::num_sps() const { return bits_.size(); }

  bool SlaterDeterminant::operator==(const SlaterDeterminant& other) const
  {
    return bits_ == other.bits();
  }

  std::ostream& operator<<(std::ostream& s, const SlaterDeterminant& sd)
  {
    for (size_t i = 0; i < sd.bits().size(); i++)
    {
      s << sd.bits()[i];
      if ((i + 1) % sd.shell_degeneracy() == 0) s << std::endl;
    }
    return s;
  }

} // namespace UNxU4
