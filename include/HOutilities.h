#pragma once

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/container_hash/hash.hpp>

#include <GlobalConstants.h>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

namespace HO
{
  /////////////// DETERMINING VALENCE SHELL ///////////////
  int valenceShell(int A);

  /////////////// DETERMINING OSCILLATOR STRENGTH ///////////////
  
  // mmt = 1 for neutron, -1 for proton
  double oscilatorLength2(int hw, int mmt); // squared
  double oscilatorLength(int hw, int mmt);

  /////////////// RME OF BOSON OPERATORS ///////////////

  // < na lla || b+ || nb llb >
  double bosonCreatorRME(int na, int lla, int nb, int llb);

  // < na lla jja || b+ || nb llb jjb >
  double bosonCreatorRME(int na, int lla, int jja, 
    int nb, int llb, int jjb);

  // < na lla || b- || nb llb >
  double bosonAnnihilatorRME(int na, int lla, int nb, int llb);

  // < na lla jja || b- || nb llb jjb >
  double bosonAnnihilatorRME(int na, int lla, int jja, 
    int nb, int llb, int jjb);

  /////////////// RME OF POSITION & MOMENTUM OPERATORS ///////////////

  // mmt = 1 for neutron, -1 for proton

  // < na lla || r || nb llb >
  double rRME(int hw, int na, int lla, int nb, int llb, int mmt);

  // < na lla jja || r || nb llb jjb >
  double rRME(int hw, int na, int lla, int jja, 
    int nb, int llb, int jjb, int mmt);

  // < na lla || p || nb llb > defined without i
  double pRME(int hw, int na, int lla, int nb, int llb, int mmt);

  // < na lla jja || p || nb llb jjb > defined without i
  double pRME(int hw, int na, int lla, int jja, 
    int nb, int llb, int jjb, int mmt);

  //////////// CLASS Table_nljmjmt /////////////

  // Assume n = 20 => max(ll) = 40, max(jj) = 41, 
  // min(mmj) = -41, max(mmj) = 41, mmt = +-1 
  // Thus these quantum numbers can be represented by int8: -128 to 127
  using qn_t = int8_t;

  class Table_nljmjmt
  {
  private:
    // key = (n, ll, jj, mmj, mmt)
    using key_type = std::tuple<qn_t, qn_t, qn_t, qn_t, qn_t>;

    // Each shell, dimension is 2(n+1)(n+2)
    // For n = 0 to 44: total number of single-particle states = 64860 
    // Thus index can be uint16 (0 to 65,535)
    std::unordered_map<key_type, uint16_t, boost::hash<key_type>> state_to_index;
    std::vector<key_type> index_to_state;

  public:
    // Constructor
    Table_nljmjmt() = default;
    Table_nljmjmt(int maxShell);

    size_t dim() const { return index_to_state.size(); }

    // Get index from quantum numbers
    uint16_t getIndex(qn_t n, qn_t ll, qn_t jj, qn_t mmj, qn_t mmt) const;

    // Get quantum numbers from index
    const key_type& getLabels(uint16_t index) const;
  };


  ////////////// STRUCT Orbit ///////////////

  struct Orbit
  {
    qn_t n_,   // nodal quantum number (!!!not shell number e)
         l_,   // orbital angular momentum
         jj_,  // twice total angular momentum
         mmt_; // twice isospin projection: +1 for neutron, -1 for proton

    // Constructor
    Orbit() : n_(0), l_(0), jj_(0), mmt_(0) {}
    Orbit(qn_t n, qn_t l, qn_t jj, qn_t mmt)
      : n_(n), l_(l), jj_(jj), mmt_(mmt)
    { }

    bool operator==(const Orbit& other) const;
    bool operator<(const Orbit& other) const;
    qn_t shell() const { return 2 * n_ + l_; }
  };

  inline std::ostream& operator<<(std::ostream& s, const Orbit& o)
  {
    return s << static_cast<int>(o.n_) << "\t" 
             << static_cast<int>(o.l_) << "\t" 
             << static_cast<int>(o.jj_) << "\t" 
             << static_cast<int>(o.mmt_);
  }


  //////////// CLASS OrbitsMap //////////////

  class OrbitsMap
  {
    qn_t emax_; // e is the shell number

    // All orbits
    std::vector<Orbit> orbits;
    std::map<Orbit, uint16_t> index_map;

  public:
    // Constructor
    OrbitsMap() : emax_(0), orbits(), index_map() { }
    OrbitsMap(int emax, std::string style); // style: "NuHamil" or "IMSRG"

    size_t num_orbits() const { return orbits.size(); }

    // Get index of the orbit
    // Caution: n = (e - l)/2 is the nodal quantum number
    // whereas e is the shell number
    uint16_t getIndex(qn_t n, qn_t l, qn_t jj, qn_t mmt) const;

    // Get orbit
    const Orbit& getOrbit(int idx) const { return orbits[idx - 1]; }

    // Print out the list of orbits with there
    void print() const
    {
      for (size_t i = 1; i <= orbits.size(); i++)
        std::cout << i << "\t" << orbits[i - 1] << std::endl;
    }
  };


  //////////// CLASS OBME_scalar /////////////
  class OBME_scalar
  {
  private:
    // key: mmta a mmtb b
    using key_type = std::tuple<int, int, int, int>;
    std::unordered_map<key_type, double, boost::hash<key_type>> me_map;

    // nlj orbit
    using nlj = std::tuple<qn_t, qn_t, qn_t>;
    std::map<nlj, int> nlj_map;

    std::string name_;
    bool custom_name;
    qn_t hw_, Nmax_, valenceShell_, maxShell_;

  public:
    // Constructor
    OBME_scalar();
    OBME_scalar(const std::string& name, 
      int hw, int Nmax, int valenceShell, int maxShell);

    // Read options during construction
    void read();
    void read_from_file_nuhamil(const std::string& filename);

    // Print out name
    void is() const;
    friend std::ostream& operator<<(std::ostream& s, const OBME_scalar& x);

    // Total number of matrix elements
    size_t num_mes() const { return me_map.size();; }

    // Get ME from quantum numbers
    double getME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta, 
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb) const;
  };

  //////////// CLASS TwBME_scalar /////////////

  // Two-body matrix elements (TBME) are read in from files computed with
  // NuHamil. Since formatted ascii files take much longer time to read, the
  // default option is to read from binary files (specifically .myg.bin);
  // comment out the line below to read from text files (.myg). The drawback is
  // that binary files are dependent on computer architecture, so they might
  // need generating and read back from the same computer. However, it might be
  // true that NuHamil has handled these binary files in such a way that they
  // can be moved across platforms.
#define READ_TWBME_FROM_BINARY
  class TwBME_scalar
  {
  private:
    // key: a b c d J
    using key_type =
        std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, qn_t>;

    std::unordered_map<key_type, double, boost::hash<key_type>> me_map;
    OrbitsMap orbit_map;

    std::string name_;
    bool custom_name;
    qn_t hw_, Nmax_, valenceShell_, maxShell_;

  public:
    // Constructor
    TwBME_scalar(); // Default constructor
    TwBME_scalar(const std::string& name, 
      int hw, int Nmax, int valenceShell, int maxShell);

    // Read options during construction
    void read();
    void read_from_binary_nuhamil(const std::string& filename);
    void read_from_binary_imsrg(const std::string& filename);
    void read_from_file_nuhamil(const std::string& filename);

    // Print out name
    void is() const;
    friend std::ostream& operator<<(std::ostream& s, const TwBME_scalar& x);

    // Total number of matrix elements
    size_t num_mes() const { return me_map.size();; }

    // Get ME from channels
    double getME(uint16_t a, uint16_t b, uint16_t c, uint16_t d,
                 qn_t J) const;

    // Get ME from quantum numbers
    double getME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta, 
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb, 
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc, 
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const;
  };
} // namespace HO
