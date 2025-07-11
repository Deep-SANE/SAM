#pragma once

#include <ostream>
#include <tuple>
#include <vector>

#include <U3basis.h>
#include <U3decomposition.h>

#include <U4basis.h>
#include <U4decomposition.h>

namespace U3xU4
{
  // A class for representing U(3)xU(4) irreps
  class irrep
  {
  public:
    u3::irrep u3;
    u4::irrep u4;

    // constructor
    irrep(const u3::irrep& u3_, const u4::irrep& u4_) : u3(u3_), u4(u4_) { }
    irrep(int n13, int n23, int n33, int n14, int n24, int n34, int n44)
        : u3(n13, n23, n33), u4(n14, n24, n34, n44)
    { }

    // comparison operator
    bool operator==(const irrep& other) const
    {
      return (u3 == other.u3) && (u4 == other.u4);
    }

    // total quanta
    int quanta() const { return u3.quanta(); }

    // total particle
    int particles() const { return u4.quanta(); }

    // print U3xU4::irrep
    friend std::ostream& operator<<(std::ostream& s, const irrep& x)
    {
      return s << x.u3 << x.u4;
    }
  };

  // Decomposition of coupling irreps
  // The first std::vector<int> is for rhomaxs
  // The second std::vector<int> is for etamaxs
  inline std::tuple<std::vector<U3xU4::irrep>, std::vector<int>,
                    std::vector<int>>
  decomposition(const U3xU4::irrep& a, const U3xU4::irrep& b)
  {
    auto [u3s, multi3s] = u3::decomposition(a.u3, b.u3);
    auto [u4s, multi4s] = u4::decomposition(a.u4, b.u4);

    std::vector<U3xU4::irrep> cs;
    std::vector<int> rhomaxs, etamaxs;

    for (int i = 0; i < u3s.size(); i++)
      for (int j = 0; j < u4s.size(); j++)
      {
        cs.emplace_back(u3s[i], u4s[j]);
        rhomaxs.push_back(multi3s[i]);
        etamaxs.push_back(multi4s[j]);
      }

    return std::make_tuple(cs, rhomaxs, etamaxs);
  }

} // namespace U3xU4
