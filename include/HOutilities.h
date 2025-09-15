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
    using key_type = std::tuple<int16_t, int16_t, int16_t, int16_t, int16_t>;

    std::unordered_map<key_type, int> state_to_index;
    std::vector<key_type> index_to_state;
    int dim;

  public:
    // Constructor
    Table_nljmjmt() : dim(0) { }
    Table_nljmjmt(int maxShell);

    // Get index from quantum numbers
    int getIndex(int n, int ll, int jj, int mmj, int mmt) const;

    // Get quantum numbers from index
    key_type getLabels(int index) const;
  };

  // < na lla || b+ || nb llb >
  double bosonCreatorRME(int na, int lla, int nb, int llb);

  // < na lla jja || b+ || nb llb jjb >
  double bosonCreatorRME(int na, int lla, int jja, int nb, int llb, int jjb,
                         qcx_utils::SO3_Racah9& racah9);

  // < na lla || b- || nb llb >
  double bosonAnnihilatorRME(int na, int lla, int nb, int llb);

  // < na lla jja || b- || nb llb jjb >
  double bosonAnnihilatorRME(int na, int lla, int jja, int nb, int llb, int jjb,
                             qcx_utils::SO3_Racah9& racah9);

  struct Orbit
  {
    int16_t n, // nodal quantum number (!!!not shell number e)
        l,     // orbital angular momentum
        jj,    // twice total angular momentum
        mmt;   // twice isospin projection: +1 for neutron, -1 for proton

    // Constructor
    Orbit() { }
    Orbit(int n_, int l_, int jj_, int mmt_);

    bool operator==(const Orbit& other) const;
    bool operator<(const Orbit& other) const;
  };

  class OrbitsMap
  {
  public:
    uint16_t emax, // e is the shell number
        num_orbits;

    // All orbits
    std::vector<Orbit> orbits;
    std::map<Orbit, int> index_map;

    // Constructor
    OrbitsMap() : emax(0), num_orbits(0), orbits(), index_map() { }
    OrbitsMap(int emax_);

    // Get index of the orbit
    // Caution: n = (e - l)/2 is the nodal quantum number
    // whereas e is the shell number
    int getIndex(int n, int l, int jj, int mmt) const;

    // Get orbit
    Orbit getOrbit(int idx) const { return orbits[idx - 1]; }
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
    // using key_type = std::tuple<uint, uint, uint, uint, uint>;
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
      std::cout << "Two-body Hamiltonian with " << name
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
    double getME(int ea, int lla, int jja, int mmja, int mmta, int eb, int llb,
                 int jjb, int mmjb, int mmtb, int ec, int llc, int jjc,
                 int mmjc, int mmtc, int ed, int lld, int jjd, int mmjd,
                 int mmtd, qcx_utils::SO3_CGC& cgcs) const;
  };
} // namespace HO
