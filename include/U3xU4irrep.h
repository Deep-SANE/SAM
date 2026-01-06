#pragma once

#include <istream>
#include <ostream>
#include <tuple>
#include <utility>
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
      : u3_(n13, n23, n33), u4_(n14, n24, n34, n44) // implicit conversion check
    { }

    // comparison:
    auto operator<=>(const irrep&) const = default;

    // getters:

    const su3::irrep& su3() const { return u3_; }
    const u3::irrep& u3() const { return u3_; }
    const u4::irrep& u4() const { return u4_; }

    // SU(3) shorthands:

    int lambda() const { return u3_.lambda(); }
    int lam() const { return lambda(); }
    int mu() const { return u3_.mu(); }

    // U(3) shorthands:

    int n13() const { return u3_.n13(); }
    int n23() const { return u3_.n23(); }
    int n33() const { return u3_.n33(); }

    // SU(4) shorthands:

    int P1() const { return u4_.P1(); }
    int P2() const { return u4_.P2(); }
    int P3() const { return u4_.P3(); }

    // U(4) shorthands:

    int n14() const { return u4_.n14(); }
    int n24() const { return u4_.n24(); }
    int n34() const { return u4_.n34(); }
    int n44() const { return u4_.n44(); }

    // is vacuum?
    bool vacuum() const { return u3_.zero() && u4_.zero(); }

    // write to binary file
    void writeBinary(std::ostream& s) const
    {
      u3_.writeBinary(s);
      u4_.writeBinary(s);
    }

    // read from binary file
    void readBinary(std::istream& s)
    {
      u3_.readBinary(s);
      u4_.readBinary(s);
    }
  };
  using irreps = std::vector<U3xU4::irrep>;

  // print U3xU4::irrep:
  inline std::ostream& operator<<(std::ostream& s, const irrep& x)
  {
    return s << x.u3() << x.u4();
  }

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

    return std::make_tuple(std::move(u3u4s), 
              std::move(rhomaxs), std::move(etamaxs));
  }

  // Get multiplicities
  inline auto RhoEta(
    const U3xU4::irrep& a, const U3xU4::irrep& b, const U3xU4::irrep& c)
  {
    int rhomax = u3::Rho(a.u3(), b.u3(), c.u3());
    int etamax = u4::Eta(a.u4(), b.u4(), c.u4());
    return std::make_pair(rhomax, etamax);
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
    return format_to(ctx.out(), "{}{}", i.u3(), i.u4());
  }
};

#endif
