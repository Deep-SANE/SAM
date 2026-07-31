#pragma once

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include <GlobalConstants.h>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Fano.h>

#include <HOorbits.h>

namespace HO
{
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
  inline TwBME_mscheme_key pack_mscheme_2Bstate(
        qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
        qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
        qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
        qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd)
  {
    return {(uint64_t(pack_mscheme_state(ea, lla, jja, mmja, mmta)) << 32) |
                      pack_mscheme_state(eb, llb, jjb, mmjb, mmtb),
            (uint64_t(pack_mscheme_state(ec, llc, jjc, mmjc, mmtc)) << 32) |
                      pack_mscheme_state(ed, lld, jjd, mmjd, mmtd)};
  }

  using TwBME_mscheme_cache_t =
    boost::unordered_flat_map<TwBME_mscheme_key, double,
                              boost::hash<TwBME_mscheme_key>>;

  // This thread's cache (indexed by omp_get_thread_num()).
  TwBME_mscheme_cache_t& twbme_mscheme_cache();

  // Lifecycle, driven from sam::initialize / sam::finalize: initialize sizes
  // one map per thread; finalize frees them.
  void initialize_twbme_cache();
  void finalize_twbme_cache();

  // Hard memory cap for very large model spaces (e.g. SuperMike Nmax>=10): if
  // this thread's m-scheme cache has grown past `max_entries`, release it
  // (swap-to-empty, so the buckets are actually returned to the allocator, not
  // just logically cleared) and return true. A cap of 0 disables it. Trades
  // some cache reuse (recompute) for a bounded per-thread footprint; results are
  // unaffected -- the cache is a pure memo. Call between tensors in the
  // expansion loop, where each shell-combo boundary is a natural reuse edge.
  bool maybe_release_twbme_cache(std::size_t max_entries);

  // Diagnostic: per-thread m-scheme cache hit/miss counts (tells us whether the
  // getME cost is cache misses = CG recompute, or hits = probes into a big
  // table). Uncomment, rebuild, run, then call report_mscheme_cache_stats().
// #define COUNT_MSCHEME_CACHE
  void report_mscheme_cache_stats();

  // Diagnostic: intra-tensor reuse of m-scheme quads -- unique quads probed per
  // tensor vs total probes. High reuse => a small per-tensor hot cache (fits L2)
  // would turn most of the 16B DRAM probes into fast local hits. Uncomment,
  // rebuild, run; report_mscheme_reuse() prints after the loop.
// #define MEASURE_MSCHEME_REUSE
#ifdef MEASURE_MSCHEME_REUSE
  void mscheme_reuse_begin();   // reset this thread's per-tensor probe set
  void mscheme_reuse_end();     // fold it into the aggregates
  struct MSchemeReuseScope      // RAII: brackets one tensor's getStrengths
  {
    MSchemeReuseScope()  { mscheme_reuse_begin(); }
    ~MSchemeReuseScope() { mscheme_reuse_end(); }
  };
#endif
  void report_mscheme_reuse();
#endif

  class TwBME_scalar
  {
  private:
    // key: the four orbital indices (a,b,c,d) packed into one uint64, plus the
    // coupling J.
    using key_type = std::pair<uint64_t, qn_t>;

    // Pack the four orbital indices into a single 64-bit value (16 bits each).
    static uint64_t pack_abcd(uint16_t a, uint16_t b, uint16_t c, uint16_t d)
    {
      return (uint64_t(a) << 48) | (uint64_t(b) << 32) |
             (uint64_t(c) << 16) | uint64_t(d);
    }

    std::unordered_map<key_type, double, boost::hash<key_type>> me_map;
    OrbitsMap orbit_map;

    const std::unordered_set<std::string> known_potentials = {
      "N2LO_EMN500_bare", "N3LO_EMN500_bare", "N4LO_EMN500_bare",
      "N3LO_EM500_bare",  "N2LO_opt_bare",    "N2LO_sat_bare", 
      "N2LO_EMN500_srg1.0", "N3LO_EMN500_srg1.0", "N4LO_EMN500_srg1.0",
      "N3LO_EM500_srg1.0",  "N2LO_opt_srg1.0",    "N2LO_sat_srg1.0",
      "N2LO_EMN500_srg2.0", "N3LO_EMN500_srg2.0", "N4LO_EMN500_srg2.0",
      "N3LO_EM500_srg2.0",  "N2LO_opt_srg2.0",    "N2LO_sat_srg2.0",
      "N2LO_EMN500_srg4.0", "N3LO_EMN500_srg4.0", "N4LO_EMN500_srg4.0",
      "N3LO_EM500_srg4.0",  "N2LO_opt_srg4.0",    "N2LO_sat_srg4.0"
    };

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

#ifdef UNCOUPLE_ME_CACHE
    // Same as the 20-arg getME, but memoizes in the caller-supplied cache. Lets
    // the hot loop resolve the thread_local cache once and pass it in, instead
    // of re-resolving it (emulated-TLS on macOS) on every one of billions of
    // calls.
    double getME(TwBME_mscheme_cache_t& cache,
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const;

    // Two-level: check the small per-tensor `local` cache first, fall through to
    // `global` (and populate local) on a miss. getStrengths keeps `local`
    // L2/L3-resident so the ~9x intra-tensor reuse skips the 2.9GB-table DRAM
    // probe.
    double getME(TwBME_mscheme_cache_t& local, TwBME_mscheme_cache_t& global,
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const;
#endif
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
    // two-body key: the four pn orbit indices (a,b,c,d) packed into one uint64
    // (16 bits each) + the bra/ket couplings (Jab,Jcd) packed into one uint16
    // (8 bits each). Hashing two scalars beats hashing a 6-tuple.
    using key2_t = std::pair<uint64_t, uint16_t>;
    std::unordered_map<key2_t, double, boost::hash<key2_t>> me_map;

    // one-body key: the two pn orbit indices (a,b) packed into one uint32
    // (16 bits each). Populated only if !NN_only_.
    using key1_t = uint32_t;
    std::unordered_map<key1_t, double> obme_map;

    // Key packers. Orbit indices fit uint16; Jab,Jcd = 2J <= 2*emax+1 <= 89
    // fit a byte each, all disjoint.
    static uint64_t pack_abcd(uint16_t a, uint16_t b, uint16_t c, uint16_t d)
    {
      return (uint64_t(a) << 48) | (uint64_t(b) << 32)
           | (uint64_t(c) << 16) |  uint64_t(d);
    }
    static uint16_t pack_JJ(qn_t Jab, qn_t Jcd)
    {
      return static_cast<uint16_t>((uint16_t(uint8_t(Jab)) << 8)
                                   | uint16_t(uint8_t(Jcd)));
    }
    static uint32_t pack_ab(uint16_t a, uint16_t b)
    {
      return (uint32_t(a) << 16) | uint32_t(b);
    }

    // true: store the two-body (NN) block only, just consume the one-body block;
    // false: also store the one-body block.
    bool NN_only_ = true;

    OrbitsMap orbit_map;

    const std::unordered_set<std::string> known_ops = {
      "GamowTellerMinus_2B_N3LO_EM500_bare"
    };

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

#ifdef UNCOUPLE_ME_CACHE
    // Cache-passing variant (see TwBME_scalar::getME with a cache).
    double getME(TwBME_mscheme_cache_t& cache,
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const;

    // Two-level variant (see TwBME_scalar::getME with local+global).
    double getME(TwBME_mscheme_cache_t& local, TwBME_mscheme_cache_t& global,
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const;
#endif

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
