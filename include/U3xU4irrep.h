#pragma once

#include <istream>
#include <ostream>
#include <tuple>
#include <vector>

#include <U3irrep.h>
#include <U3decomposition.h>

#include <U4irrep.h>
#include <U4decomposition.h>

#include <qcx_readwrite.h>

#ifdef HAVE_FMT
#include <fmt/format.h>
#endif

namespace U3xU4
{
  // A class for representing U(3)xU(4) irreps
  class irrep
  {
    u3::irrep u3_;
    u4::irrep u4_;

  public:
    // constructors:

    irrep() = default;

    irrep(const u3::irrep& u3, const u4::irrep& u4) : u3_(u3), u4_(u4) { }

    irrep(int n13, int n23, int n33, int n14, int n24, int n34, int n44)
        : u3_(n13, n23, n33), u4_(n14, n24, n34, n44)
    { }

    // comparison:
    auto operator<=>(const irrep&) const = default;

    // getters:

    const auto& u3() const { return u3_; }
    const auto& u4() const { return u4_; }

    // total quanta:
    int quanta() const { return u3_.quanta(); }

    // total particles:
    int particles() const { return u4_.quanta(); }

    // print U3xU4::irrep:
    friend std::ostream& operator<<(std::ostream& s, const irrep& x)
    {
      return s << x.u3_ << x.u4_;
    }

    // write to binary file
    void writeBinary(std::ostream& s) const
    {
      qcx_utils::write<int>(s, u3_.n13());
      qcx_utils::write<int>(s, u3_.n23());
      qcx_utils::write<int>(s, u3_.n33());

      qcx_utils::write<int>(s, u4_.n14());
      qcx_utils::write<int>(s, u4_.n24());
      qcx_utils::write<int>(s, u4_.n34());
      qcx_utils::write<int>(s, u4_.n44());
    }

    // read from binary file
    void readBinary(std::istream& s)
    {
      int n13, n23, n33;
      int n14, n24, n34, n44;

      qcx_utils::read<int>(s, n13);
      qcx_utils::read<int>(s, n23);
      qcx_utils::read<int>(s, n33);

      qcx_utils::read<int>(s, n14);
      qcx_utils::read<int>(s, n24);
      qcx_utils::read<int>(s, n34);
      qcx_utils::read<int>(s, n44);

      u3_ = u3::irrep(n13, n23, n33);
      u4_ = u4::irrep(n14, n24, n34, n44);
    }
  };

  // Decomposition of coupling irreps
  // The first std::vector<int> is for rhomaxs
  // The second std::vector<int> is for etamaxs
  inline auto decomposition(const U3xU4::irrep& a, const U3xU4::irrep& b)
  {
    auto [u3s, multi3s] = u3::decomposition(a.u3(), b.u3());
    auto [u4s, multi4s] = u4::decomposition(a.u4(), b.u4());

    std::vector<U3xU4::irrep> u3u4s;
    std::vector<int> rhomaxs, etamaxs;

    auto n = u3s.size() * u4s.size();
    u3u4s.reserve(n);
    rhomaxs.reserve(n);
    etamaxs.reserve(n);

    for (int i = 0; i < u3s.size(); i++)
      for (int j = 0; j < u4s.size(); j++)
      {
        u3u4s.emplace_back(u3s[i], u4s[j]);
        rhomaxs.push_back(multi3s[i]);
        etamaxs.push_back(multi4s[j]);
      }

    return std::make_tuple(u3u4s, rhomaxs, etamaxs);
  }

} // namespace U3xU4

#ifdef HAVE_FMT

// formatter for U3xU4::irrep:

template <>
class fmt::formatter<U3xU4::irrep>
{
public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename Context>
  constexpr auto format(const U3xU4::irrep& i, Context& ctx) const
  {
    return format_to(ctx.out(), "({},{})", i.u3(), i.u4());
  }
};

#endif
