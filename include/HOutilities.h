#pragma once

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <CustomHash.h>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

namespace HO
{
  class Table_nljmjmt
  {
  private:
    // key = (n, ll, jj, mmj, mmt)
    // Assume n = 20 => max(ll) = 40, max(jj) = 41, 
    // min(mmj) = -41, max(mmj) = 41, mmt = +-1 => Key can be int8: -128 to 127
    using key_type = std::tuple<int8_t, int8_t, int8_t, int8_t, int8_t>;

    // Each shell, dimension is 2(n+1)(n+2)
    // For n = 0 to 20: total number of single-particle states = 7084 
    // Thus index can be uint16
    std::unordered_map<key_type, uint16_t> state_to_index;
    std::vector<key_type> index_to_state;
    uint16_t dim;

  public:
    // Constructor
    Table_nljmjmt() : dim(0) { }
    Table_nljmjmt(int maxShell);

    // Get index from quantum numbers
    uint16_t getIndex(int8_t n, int8_t ll, int8_t jj, 
                        int8_t mmj, int8_t mmt) const;

    // Get quantum numbers from index
    key_type getLabels(uint16_t index) const;
  };

  // < na lla || b+ || nb llb >
  double bosonCreatorRME(int8_t na, int8_t lla, int8_t nb, int8_t llb);

  // < na lla jja || b+ || nb llb jjb >
  double bosonCreatorRME(int8_t na, int8_t lla, int8_t jja, int8_t nb, 
                int8_t llb, int8_t jjb, qcx_utils::SO3_Racah9& racah9);

  // < na lla || b- || nb llb >
  double bosonAnnihilatorRME(int8_t na, int8_t lla, int8_t nb, int8_t llb);

  // < na lla jja || b- || nb llb jjb >
  double bosonAnnihilatorRME(int8_t na, int8_t lla, int8_t jja, int8_t nb, 
                int8_t llb, int8_t jjb, qcx_utils::SO3_Racah9& racah9);

  struct Orbit
  {
    int8_t n,  // nodal quantum number (!!!not shell number e)
        l,     // orbital angular momentum
        jj,    // twice total angular momentum
        mmt;   // twice isospin projection: +1 for neutron, -1 for proton

    // Constructor
    Orbit() { }
    Orbit(int8_t n_, int8_t l_, int8_t jj_, int8_t mmt_);

    bool operator==(const Orbit& other) const;
    bool operator<(const Orbit& other) const;
  };

  inline std::ostream& operator<<(std::ostream& s, const Orbit& o)
  {
    return s << static_cast<int>(o.n) << "\t" 
             << static_cast<int>(o.l) << "\t" 
             << static_cast<int>(o.jj) << "\t" 
             << static_cast<int>(o.mmt);
  }

  class OrbitsMap
  {
  public:
    uint8_t emax; // e is the shell number
    uint16_t num_orbits;

    // All orbits
    std::vector<Orbit> orbits;
    std::map<Orbit, uint16_t> index_map;

    // Constructor
    OrbitsMap() : emax(0), num_orbits(0), orbits(), index_map() { }
    OrbitsMap(int emax_);

    // Get index of the orbit
    // Caution: n = (e - l)/2 is the nodal quantum number
    // whereas e is the shell number
    int getIndex(int8_t n, int8_t l, int8_t jj, int8_t mmt) const;

    // Get orbit
    Orbit getOrbit(int idx) const { return orbits[idx - 1]; }

    // Print out the list of orbits with there
    void print() const
    {
      for (size_t i = 1; i <= orbits.size(); i++)
        std::cout << i << "\t" << orbits[i - 1] << std::endl;
    }
  };

  // Two-body matrix elements (TBME) are read in from files computed with
  // NuHamil. Since formatted ascii files take much longer time to read, the
  // default option is to read from binary files (specifically .myg.bin);
  // comment out the line below to read from text files (.myg). The drawback is
  // that binary files are dependent on computer architecture, so they might
  // need generating and read back from the same computer. However, it might be
  // true that NuHamil has handled these binary files in such a way that they
  // can be moved across platforms.
#define READ_FROM_BINARY
  class TBME_scalar
  {
  private:
    // key: a b c d J
    using key_type =
        std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>;

    std::unordered_map<key_type, double> me_map;
    OrbitsMap orbit_map;
    int num_me;

    std::string name;
    int8_t hw, Nmax, valenceShell, maxShell;

  public:
    // Constructor
    TBME_scalar(); // Default constructor
    TBME_scalar(const std::string& name_, int hw_, int Nmax_,
                int valenceShell_, int maxShell_);

    // Read options during construction
    void read();
    void read_from_binary(const std::string& filename);
    void read_from_file(const std::string& filename);

    // Print out name
    void is() const
    {
      std::cout << "Two-body potential with " << name
                << " hw=" << static_cast<int>(hw)
                << " valenceShell=" << static_cast<int>(valenceShell)
                << " Nmax=" << static_cast<int>(Nmax)
                << " maxShell=" << static_cast<int>(maxShell) << std::endl;
    }

    // Total number of matrix elements
    int countMEs() const { return num_me; }

    // Get ME from channels
    double getME(uint16_t a, uint16_t b, uint16_t c, uint16_t d,
                 uint16_t J) const;

    // Get ME from quantum numbers
    double getME(
      int8_t ea, int8_t lla, int8_t jja, int8_t mmja, int8_t mmta, 
      int8_t eb, int8_t llb, int8_t jjb, int8_t mmjb, int8_t mmtb, 
      int8_t ec, int8_t llc, int8_t jjc, int8_t mmjc, int8_t mmtc, 
      int8_t ed, int8_t lld, int8_t jjd, int8_t mmjd, int8_t mmtd, 
      qcx_utils::SO3_CGC& cgcs) const;
  };
} // namespace HO
