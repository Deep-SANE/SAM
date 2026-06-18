#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <boost/container/flat_map.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include <qcx_readwrite.h>

#include <HOutilities.h>
#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  /* On-disk + in-memory cache for SingleShellTensor_phys::expand().

     expand() is interaction-independent: its result depends only on the
     tensor's structure (shell n, number of ad's/at's, the Omega irreps and the
     coupling T) together with the m-scheme labels (LL, SS, TT, MMT, JJ, MMJ).
     The expansion is the expensive step in tensor decomposition, so we
     precompute it once and reuse it across operators and runs.

     Layout mirrors RME_db: one binary file per SingleShellTensor *structure*
     (named from its quantum numbers), each holding a map from the projection
     key (LL, SS, TT, MMT, JJ, MMJ) to the serialized expansion. The format
     version is encoded in the filename, so a layout change simply ignores the
     stale files.

     Only the _phys tensor is cached here; SingleShellTensor_gelfand may get an
     analogous database later. */
  class SSTDecomp_phys_db
  {
  public:
    // The value type returned by SingleShellTensor_phys::expand().
    using Expansion = std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
                                 std::vector<U3xU4::ads_phys>,     // ad's
                                 std::vector<U3xU4::ats_phys>>;    // a's

  private:
    // Bump when the on-disk Record layout changes (encoded in the filename).
    static constexpr int FORMAT_VERSION = 1;

    std::string dir;

    // Within-file key: the m-scheme projection labels packed into 64 bits.
    // (The ordering it induces need not be physically meaningful, only stable.)
    struct DecompKey
    {
      uint64_t qkey = 0;

      DecompKey() = default;
      explicit DecompKey(uint64_t k) : qkey(k) { }
      DecompKey(int LL, int SS, int TT, int MMT, int JJ, int MMJ);

      auto operator<=>(const DecompKey&) const = default;
    };

    // Serialized, interaction-independent expansion. We keep this compact form
    // in memory and rebuild the heavyweight Eigen / ad_phys / at_phys objects
    // on demand in get(). ad_phys / at_phys are stored by their Table_nljmjmt
    // index (getIndex()) -- their full observable state, since every cached
    // a^tilde is already an annihilation operator (tensor == false). The index
    // enumeration is maxShell-invariant, so records are portable across runs.
    //
    // The three arrays' row counts are stored independently: the general case
    // has n_ad_rows == n_at_rows == nrows, but a pure multi-creation tensor
    // (Tad only) leaves atsArr empty (n_at_rows == 0) and a pure
    // multi-annihilation tensor (Tat only) leaves adsArr empty.
    struct Record
    {
      uint64_t nrows = 0; // coeff rows (= coeffs.size())
      uint32_t ncols = 0; // columns per coeff row  (rho*eta*kappa*zeta)
      std::vector<double> coeffs; // nrows * ncols, row-major

      uint32_t n_ad_rows = 0;       // ad rows (= adsArr.size(): nrows or 0)
      uint16_t nad = 0;             // ad's per row
      std::vector<uint16_t> ad_idx; // n_ad_rows * nad  (table indices)

      uint32_t n_at_rows = 0;       // a rows  (= atsArr.size(): nrows or 0)
      uint16_t nat = 0;             // a's per row
      std::vector<uint16_t> at_idx; // n_at_rows * nat  (table indices)
    };

    // One binary file holding every cached projection of one tensor structure.
    struct DecompFile
    {
      std::string path;
      boost::container::flat_map<DecompKey, Record> records;
      size_t initial_size = 0;
      mutable std::mutex file_mutex_;

      explicit DecompFile(const std::string& path_);

      bool contains(const DecompKey& key) const;

      // Copy the record out under lock (empty if absent), to be unpacked
      // outside any lock -- mirrors RME_db's copy-out pattern.
      std::optional<Record> get_if_exist(const DecompKey& key) const;

      // Insert if new; return a copy of the stored record (existing wins).
      Record append_or_get(const DecompKey& key, Record&& rec);
    };

    // Top-level cache: filename -> file.
    boost::unordered_flat_map<std::string, std::unique_ptr<DecompFile>> cache_;
    mutable std::shared_mutex cache_mutex_;

    DecompFile& getFile(const std::string& file);

    // Per-tensor file name, built from the structural quantum numbers (same
    // scheme as SingleShellTensor::readBinary, plus a format-version tag).
    static std::string filename(const U3xU4::SingleShellTensor& t);

    // (de)serialization between the live Expansion and the compact Record.
    static Record pack(const Expansion& e);
    static Expansion unpack(const Record& rec,
                            const HO::Table_nljmjmt& labels);
    static void writeRecord(std::ostream& s, const DecompKey& key,
                            const Record& rec);

  public:
    SSTDecomp_phys_db();
    explicit SSTDecomp_phys_db(int maxShell);

    // Flush newly computed entries to disk and release in-memory data.
    void finalize();

    // Cached SingleShellTensor_phys::expand(): returns the stored expansion,
    // computing and caching it on a miss. Drop-in replacement for t.expand().
    Expansion get(const U3xU4::SingleShellTensor_phys& t);
  };
} // namespace U3xU4
