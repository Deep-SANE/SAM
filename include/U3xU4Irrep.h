#pragma once

#include <iostream>
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
    irrep() = default;
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

    // write to binary file
    void writeBinary(std::ostream& s) const
    {
      int n13 = u3.n13();
      int n23 = u3.n23();
      int n33 = u3.n33();

      int n14 = u4.n14();
      int n24 = u4.n24();
      int n34 = u4.n34();
      int n44 = u4.n44();

      s.write(reinterpret_cast<const char*>(&n13), sizeof(n13));
      s.write(reinterpret_cast<const char*>(&n23), sizeof(n23));
      s.write(reinterpret_cast<const char*>(&n33), sizeof(n33));

      s.write(reinterpret_cast<const char*>(&n14), sizeof(n14));
      s.write(reinterpret_cast<const char*>(&n24), sizeof(n24));
      s.write(reinterpret_cast<const char*>(&n34), sizeof(n34));
      s.write(reinterpret_cast<const char*>(&n44), sizeof(n44));
    }

    // read from binary file
    void readBinary(std::istream& s)
    {
      int n13, n23, n33;
      int n14, n24, n34, n44;

      s.read(reinterpret_cast<char*>(&n13), sizeof(n13));
      s.read(reinterpret_cast<char*>(&n23), sizeof(n23));
      s.read(reinterpret_cast<char*>(&n33), sizeof(n33));

      s.read(reinterpret_cast<char*>(&n14), sizeof(n14));
      s.read(reinterpret_cast<char*>(&n24), sizeof(n24));
      s.read(reinterpret_cast<char*>(&n34), sizeof(n34));
      s.read(reinterpret_cast<char*>(&n44), sizeof(n44));

      u3 = u3::irrep(n13, n23, n33);
      u4 = u4::irrep(n14, n24, n34, n44);
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
