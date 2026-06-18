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
#include <so3Coeffs/SO3_Fano.h>

#include <cstdint>
#include <utility>
#include <vector>
#include <boost/unordered/unordered_flat_map.hpp>

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

  // Memoize the m-scheme two-body matrix element. getME(qn...) is otherwise a
  // Clebsch-Gordan recoupling sum recomputed on every call, and the tensor
  // expansion requests the same m-scheme tuple repeatedly. With this defined,
  // results are cached in a shared, thread-safe table (lock-free under
  // OpenMP). Comment out the line below to disable. Trades memory (the
  // materialized m-scheme ME table) for speed.
#define UNCOUPLE_ME_CACHE

#ifdef UNCOUPLE_ME_CACHE
  // Per-thread cache of m-scheme two-body matrix elements. Each thread owns a
  // private map, so getME's hot lookup is lock-free: a shared
  // boost::concurrent_flat_map was profiled as the dominant cost (per-call read
  // lock + TLS), and since the computation is fully amortized and the table is
  // small, replicating it per thread is far cheaper than locking a shared one.
  // The key packs the four m-scheme states into two uint64 (each -> 30 bits;
  // see getME). Only one nuclear potential is active at a time, so the table
  // needs no per-operator tag.
  using TwBME_mscheme_key = std::pair<uint64_t, uint64_t>;
  using TwBME_mscheme_cache_t =
    boost::unordered_flat_map<TwBME_mscheme_key, double,
                              boost::hash<TwBME_mscheme_key>>;

  // This thread's cache (indexed by omp_get_thread_num()).
  TwBME_mscheme_cache_t& twbme_mscheme_cache();

  // Lifecycle, driven from sam::initialize / sam::finalize: initialize sizes
  // one map per thread; finalize frees them.
  void initialize_twbme_cache();
  void finalize_twbme_cache();
#endif

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

    // The actual m-scheme ME (CG recoupling sum). getME(qn...) calls this
    // directly, or through twbme_mscheme_cache() when UNCOUPLE_ME_CACHE is on.
    double computeME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const;

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

    // Release the hash table to free memory
    void clear();

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

  //////////// CLASS TwBME_general /////////////

  // General (tensor) two-body operator of rank (OpJ, OpP, OpZ), read from the
  // gzip me2j-like text written by NuHamil's write_general_operator_gzip
  // (NuHamil-public/src/TwoBody/TwoBodyLabOps.F90). Unlike TwBME_scalar the
  // entries carry no indices: the file lists matrix elements in a fixed nested
  // loop order, so the reader replicates that loop exactly (see the imsrg
  // reference ReadWrite::ReadOperator2b_Miyagi).
  //
  // "One reader, two stores": one pass over the file fills the two-body store
  // (active) and, when NN_only_ == false, the one-body store too. For now the
  // one-body block is consumed but not stored (NN_only_ == true).
  //
  // Isospin convention: mmt = +1 -> neutron (n), mmt = -1 -> proton (p), so
  // proton index = orbit_map.getIndex(n,l,jj,-1), neutron = getIndex(...,+1).
  class TwBME_general
  {
  private:
    // two-body key: a b c d Jab Jcd  (pn orbit indices; bra and ket J)
    using key2_t =
        std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, qn_t, qn_t>;
    std::unordered_map<key2_t, double, boost::hash<key2_t>> me_map;

    // one-body key: a b  (pn orbit indices); populated only if !NN_only_
    using key1_t = std::tuple<uint16_t, uint16_t>;
    std::unordered_map<key1_t, double, boost::hash<key1_t>> obme_map;

    // true: store the two-body (NN) block only, just consume the one-body block;
    // false: also store the one-body block.
    bool NN_only_ = true;

    OrbitsMap orbit_map;

    std::string name_;
    bool custom_name;
    qn_t hw_, Nmax_, valenceShell_, maxShell_;

    // operator quantum numbers, read from the file header
    qn_t JJ_, parity_, TT_;

    // The actual m-scheme two-body ME (rank-OpJ Wigner-Eckart CG sum). The
    // 20-arg getME below calls this directly, or through twbme_mscheme_cache()
    // when UNCOUPLE_ME_CACHE is on.
    double computeME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const;

  public:
    // Constructor
    TwBME_general(); // Default constructor
    TwBME_general(const std::string& name,
      int hw, int Nmax, int valenceShell, int maxShell);

    // false => also store the one-body block (call before read())
    void set_NN_only(bool b) { NN_only_ = b; }

    // Read options during construction
    void read();
    void read_from_gzip_nuhamil(const std::string& filename);

    // Release the hash tables to free memory
    void clear();

    // Print out name
    void is() const;
    friend std::ostream& operator<<(std::ostream& s, const TwBME_general& x);

    // Operator quantum numbers (from the file header)
    int JJ() const { return static_cast<int>(JJ_); }
    int parity() const { return static_cast<int>(parity_); }
    int TT() const { return static_cast<int>(TT_); }

    // Total number of stored matrix elements
    size_t num_mes() const { return me_map.size(); }
    size_t num_obmes() const { return obme_map.size(); }

    // Get coupled ME from channels (bra J = Jab, ket J = Jcd)
    double getME(uint16_t a, uint16_t b, uint16_t c, uint16_t d,
                 qn_t Jab, qn_t Jcd) const;

    // Get m-scheme ME from quantum numbers (rank-OpJ Wigner-Eckart decoupling)
    double getME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const;

    // --- one-body piece (only meaningful when read with NN_only_ == false) ---

    // Coupled (reduced) one-body ME by pn orbit indices.
    double getME(uint16_t a, uint16_t b) const;

    // m-scheme one-body ME (rank-OpJ Wigner-Eckart; reduces to the scalar
    // one-body ME when OpJ=0,OpP=+1,OpZ=0).
    double getME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb) const;
  };
} // namespace HO
