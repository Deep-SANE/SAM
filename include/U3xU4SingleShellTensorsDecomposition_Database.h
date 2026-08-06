#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

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
     key (LL, SS, TT, MMT, JJ, MMJ) to the serialized Expansion. The format
     version is encoded in the filename, so a layout change simply ignores the
     stale files.

     The cached value is the shared U3xU4::Expansion (flat coeffs + row-major
     operator indices); only its runtime `labels` pointer is not serialized.
     Only the _phys tensor is cached here. */
  class SSTDecomp_phys_db
  {
  private:
    // Bump when the on-disk Expansion layout changes (encoded in the filename).
    static constexpr int FORMAT_VERSION = 2;

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

    // One binary file holding every cached projection of one tensor structure.
    // Expansions are held by unique_ptr so they are pointer-stable: inserts may
    // relocate the flat_map's elements, but the pointees never move, so get_view
    // can hand out borrowed pointers into a stored Expansion.
    struct DecompFile
    {
      std::string path;
      boost::container::flat_map<DecompKey, std::unique_ptr<Expansion>> records;
      size_t initial_size = 0;
      mutable std::mutex file_mutex_;

      explicit DecompFile(const std::string& path_);

      // Stable pointer to the stored Expansion, or nullptr if absent. The
      // runtime `labels` pointer is (re)bound under the lock -- the data is
      // immutable and never erased until finalize, so it may be read lock-free
      // after the lookup. Misses are computed and inserted by the caller.
      Expansion* get_ptr(const DecompKey& key,
                         const HO::Table_nljmjmt& labels) const;
      Expansion* append_or_get_ptr(const DecompKey& key, Expansion&& e);
    };

    // Top-level cache: filename -> file.
    boost::unordered_flat_map<std::string, std::unique_ptr<DecompFile>> cache_;
    mutable std::shared_mutex cache_mutex_;

    DecompFile& getFile(const std::string& file);

    // Per-tensor file name, built from the structural quantum numbers (same
    // scheme as SingleShellTensor::readBinary, plus a format-version tag).
    static std::string filename(const U3xU4::SingleShellTensor& t);

    // Serialize one cached expansion (coeffs + flat operator lists).
    static void writeRecord(std::ostream& s, const DecompKey& key,
                            const Expansion& e);

  public:
    SSTDecomp_phys_db();
    explicit SSTDecomp_phys_db(int maxShell);

    // Flush newly computed entries to disk and release in-memory data.
    void finalize();

    // Diagnostic snapshot: (#files loaded, total #records) currently held in the
    // in-memory cache. Cheap; for the per-rank memory monitor.
    std::pair<std::size_t, std::size_t> cache_stats() const;

    // Borrow-or-own handle to an Expansion. On a cache hit get_view() borrows a
    // stable pointer into the in-memory cache (zero copy, valid for the db's
    // lifetime); the uncached single-operator case owns a freshly built one.
    // All accessors live on Expansion itself -- use via operator->.
    class ExpansionRef
    {
      std::unique_ptr<Expansion> owned_;     // set only on the uncached path
      const Expansion* p_ = nullptr;         // borrowed, or == owned_.get()

      friend class SSTDecomp_phys_db;
      explicit ExpansionRef(const Expansion* p) : p_(p) { }
      explicit ExpansionRef(std::unique_ptr<Expansion> o)
        : owned_(std::move(o)), p_(owned_.get()) { }

    public:
      ExpansionRef() = default;
      const Expansion& operator*()  const { return *p_; }
      const Expansion* operator->() const { return p_; }
      explicit operator bool() const { return p_ != nullptr; }
    };

    // Cached expand() as a borrow-or-own Expansion (see ExpansionRef).
    ExpansionRef get_view(const U3xU4::SingleShellTensor_phys& t);
  };
} // namespace U3xU4
