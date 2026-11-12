#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <U3xU4SingleShellTensorsDecomposition_Database.h>
#include <DataDirectory.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim: return evicted pages to the OS
#endif

namespace U3xU4
{
  //=========================================================================
  // DecompKey: pack the six "twice" m-scheme labels into 48 bits.
  //=========================================================================
  SSTDecomp_phys_db::DecompKey::DecompKey(int LL, int SS, int TT,
                                          int MMT, int JJ, int MMJ)
  {
    // Labels are signed (MMJ/MMT can be negative); map through int8 -> uint8 so
    // the bit pattern is reversible. The induced ordering need only be stable.
    auto b = [](int v) { return uint64_t(uint8_t(int8_t(v))); };
    qkey = (b(LL) << 40) | (b(SS) << 32) | (b(TT) << 24) |
           (b(MMT) << 16) | (b(JJ) << 8) | b(MMJ);
  }

  //=========================================================================
  // filename: structural identity of the tensor (same scheme as
  // SingleShellTensor::readBinary), plus a format-version tag.
  //=========================================================================
  std::string SSTDecomp_phys_db::filename(const U3xU4::SingleShellTensor& t)
  {
    using std::to_string;

    std::string shell = "n" + to_string(t.shell());

    std::string ads = "_" + to_string(t.num_ads()) + "ads";
    for (const auto& Omega : t.Omega_ads())
      ads += "_" + to_string(Omega.lam()) + "-" + to_string(Omega.mu()) + "-" +
             to_string(Omega.P1()) + "-" + to_string(Omega.P2()) + "-" +
             to_string(Omega.P3());

    std::string ats = "_" + to_string(t.num_ats()) + "ats";
    for (const auto& Omega : t.Omega_ats())
      ats += "_" + to_string(Omega.lam()) + "-" + to_string(Omega.mu()) + "-" +
             to_string(Omega.P1()) + "-" + to_string(Omega.P2()) + "-" +
             to_string(Omega.P3());

    std::string irr = "_" + to_string(t.rho0max()) + "-" + to_string(t.lam()) +
                      "-" + to_string(t.mu()) + "-" + to_string(t.eta0max()) +
                      "-" + to_string(t.P1()) + "-" + to_string(t.P2()) + "-" +
                      to_string(t.P3());

    return shell + ads + ats + irr + "_v" + to_string(FORMAT_VERSION) + ".bin";
  }

  //=========================================================================
  // Binary record I/O (one Expansion per key; terminated by EOF). nrows is
  // written so the flat vectors can be sized on read; the runtime labels
  // pointer is not serialized.
  //=========================================================================
  void SSTDecomp_phys_db::writeRecord(std::ostream& s, const DecompKey& key,
                                      const Expansion& e)
  {
    uint64_t nrows = e.rows();
    utils::writeBinary(s, key.qkey);
    utils::writeBinary(s, nrows);
    utils::writeBinary(s, e.coeffs.ncols);
    utils::writeBinary(s, e.nops);
    utils::writeBinary(s, e.coeffs.data);
    utils::writeBinary(s, e.op_index);
    utils::writeBinary(s, e.op_is_ad);
  }

  //=========================================================================
  // DecompFile
  //=========================================================================
  // In-memory payload size of one Expansion -- the unit the memory cap counts.
  static std::size_t expansion_bytes(const Expansion& e)
  {
    return e.coeffs.data.size() * sizeof(double)
         + e.op_index.size()   * sizeof(std::uint16_t)
         + e.op_is_ad.size()   * sizeof(std::uint8_t);
  }

  // Field-only (no qkey) Expansion I/O for the overflow store: the offset index
  // already holds the key, so only the payload is written/read. Same field order
  // as writeRecord / the DecompFile loader, minus the leading qkey.
  static void write_expansion_fields(std::ostream& s, const Expansion& e)
  {
    uint64_t nrows = e.rows();
    utils::writeBinary(s, nrows);
    utils::writeBinary(s, e.coeffs.ncols);
    utils::writeBinary(s, e.nops);
    utils::writeBinary(s, e.coeffs.data);
    utils::writeBinary(s, e.op_index);
    utils::writeBinary(s, e.op_is_ad);
  }

  static bool read_expansion_fields(std::istream& s, Expansion& e)
  {
    uint64_t nrows;
    utils::readBinary(s, nrows);
    utils::readBinary(s, e.coeffs.ncols);
    utils::readBinary(s, e.nops);
    if (!s) return false;
    e.coeffs.data.resize(nrows * e.coeffs.ncols);
    e.op_index.resize(nrows * e.nops);
    e.op_is_ad.resize(nrows * e.nops);
    utils::readBinary(s, e.coeffs.data);
    utils::readBinary(s, e.op_index);
    utils::readBinary(s, e.op_is_ad);
    return bool(s);
  }

  SSTDecomp_phys_db::DecompFile::DecompFile(const std::string& path_)
    : path(path_)
  {
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
      initial_size = 0;
      return;
    }

    while (true)
    {
      uint64_t qkey;
      utils::readBinary(file, qkey);
      if (!file) break; // clean EOF

      auto e = std::make_unique<Expansion>();
      uint64_t nrows;
      utils::readBinary(file, nrows);
      utils::readBinary(file, e->coeffs.ncols);
      utils::readBinary(file, e->nops);

      e->coeffs.data.resize(nrows * e->coeffs.ncols);
      e->op_index.resize(nrows * e->nops);
      e->op_is_ad.resize(nrows * e->nops);
      utils::readBinary(file, e->coeffs.data);
      utils::readBinary(file, e->op_index);
      utils::readBinary(file, e->op_is_ad);

      if (!file) break; // truncated / corrupt -> stop loading

      bytes += expansion_bytes(*e);
      records.emplace(DecompKey(qkey), std::move(e));
    }

    file.close();
    initial_size = records.size();
  }

  Expansion*
  SSTDecomp_phys_db::DecompFile::get_ptr(const DecompKey& key,
                                         const HO::Table_nljmjmt& labels) const
  {
    std::lock_guard lock(file_mutex_);
    auto it = records.find(key);
    if (it == records.end()) return nullptr;
    // (Re)bind the runtime labels pointer under the lock -- needed for records
    // loaded from disk; idempotent for ones computed this run.
    it->second->labels = &labels;
    return it->second.get();
  }

  Expansion*
  SSTDecomp_phys_db::DecompFile::append_or_get_ptr(const DecompKey& key,
                                                   Expansion&& e)
  {
    std::lock_guard lock(file_mutex_);
    const std::size_t nb = expansion_bytes(e); // before the move
    auto [it, inserted] =
        records.emplace(key, std::make_unique<Expansion>(std::move(e)));
    if (inserted) bytes += nb;
    return it->second.get(); // existing one wins on race
  }

  //=========================================================================
  // getFile: top-level cache with shared/exclusive locking (as RME_db).
  //=========================================================================
  SSTDecomp_phys_db::DecompFile&
  SSTDecomp_phys_db::getFile(const std::string& file)
  {
    const std::size_t stamp =
        access_clock_.fetch_add(1, std::memory_order_relaxed);
    {
      std::shared_lock lock(cache_mutex_);
      auto it = cache_.find(file);
      if (it != cache_.end())
      {
        it->second->last_access.store(stamp, std::memory_order_relaxed);
        it->second->refs.fetch_add(1, std::memory_order_relaxed);
        return *(it->second);
      }
    }

    auto path = dir + "/" + file;
    auto new_file = std::make_unique<DecompFile>(path);

    std::unique_lock lock(cache_mutex_);
    auto it = cache_.find(file);
    if (it != cache_.end())
    {
      it->second->last_access.store(stamp, std::memory_order_relaxed);
      it->second->refs.fetch_add(1, std::memory_order_relaxed);
      return *(it->second);
    }

    new_file->last_access.store(stamp, std::memory_order_relaxed);
    new_file->refs.fetch_add(1, std::memory_order_relaxed);
    auto [it_inserted, inserted] = cache_.emplace(file, std::move(new_file));
    return *(it_inserted->second);
  }

  //=========================================================================
  // Overflow store (SAM_DECOMP_OVERFLOW=1): disk-back the big uncached decomps.
  // Instead of discarding a giant after use, append it to a per-rank file and
  // record its offset, so a re-reference RELOADS it (seek) rather than
  // recomputing. Both paths hold overflow_mutex_, which serializes all overflow
  // I/O -- acceptable since the big decomps are the minority. RAM stays bounded:
  // only the offset lives in memory. Reads use a fresh ifstream (the writer is an
  // append-only ofstream, flushed per write) so read/write never share a stream.
  //=========================================================================
  void SSTDecomp_phys_db::store_overflow(const std::string& file,
                                         const DecompKey& key,
                                         const Expansion& e)
  {
    std::lock_guard lock(overflow_mutex_);
    if (!overflow_out_) return;
    auto& slot = overflow_index_[file];
    if (slot.find(key.qkey) != slot.end()) return; // already stored (race dup)

    const long long off = static_cast<long long>(overflow_out_.tellp());
    write_expansion_fields(overflow_out_, e);
    overflow_out_.flush();               // make it visible to the read stream
    if (overflow_out_) slot.emplace(key.qkey, off);
  }

  std::unique_ptr<Expansion>
  SSTDecomp_phys_db::load_overflow(const std::string& file, const DecompKey& key)
  {
    std::lock_guard lock(overflow_mutex_);
    auto fit = overflow_index_.find(file);
    if (fit == overflow_index_.end()) return nullptr;
    auto kit = fit->second.find(key.qkey);
    if (kit == fit->second.end()) return nullptr;

    std::ifstream in(overflow_path_, std::ios::binary);
    if (!in) return nullptr;
    in.seekg(kit->second);
    auto e = std::make_unique<Expansion>();
    if (!read_expansion_fields(in, *e)) return nullptr;
    return e;
  }

  //=========================================================================
  // Constructors: discover the storage directory and preload existing files.
  //=========================================================================
  namespace
  {
    std::string decomp_dir()
    {
      return sam::data_dir() + "/decomp_phys";
    }

    [[maybe_unused]] int leading_shell(const std::string& filename)
    {
      // filenames look like "n<shell>_...": parse the digits after 'n'.
      size_t pos = 1;
      int shell = 0;
      while (pos < filename.size() && std::isdigit(filename[pos]))
        shell = shell * 10 + (filename[pos++] - '0');
      return shell;
    }
  } // namespace

  SSTDecomp_phys_db::SSTDecomp_phys_db()
  {
    dir = decomp_dir();
    std::filesystem::create_directories(dir);

#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0)
      std::cerr << "Start loading tensor-decomposition files" << std::endl;
#else
    std::cout << "Start loading tensor-decomposition files" << std::endl;
#endif

    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
      if (entry.path().extension() != ".bin") continue;

      std::string fn = entry.path().filename().string();
      if (fn.empty() || fn[0] != 'n') continue;

      cache_.emplace(fn, std::make_unique<DecompFile>(entry.path().string()));
    }

#ifdef USE_MPI
    if (mpi_rank == 0)
      std::cerr << "Finish loading tensor-decomposition files" << std::endl;
#else
    std::cout << "Finish loading tensor-decomposition files" << std::endl;
#endif

    cache_.reserve(cache_.size() + 16);
  }

  // LAZY: do NOT preload the whole decomp_phys folder. The old body pulled EVERY
  // file (<= maxShell) into RAM on EVERY rank -- replicated across all ranks on a
  // node, plus a startup I/O storm (every rank scanning + reading the entire
  // folder off the shared FS). getFile() loads each file on first access
  // (thread-safe), so a rank only holds the decomposition files its own tensors
  // actually touch, and the reads spread over the compute. maxShell is kept for
  // signature compatibility; only requested (hence in-range) files ever load.
  SSTDecomp_phys_db::SSTDecomp_phys_db(int maxShell)
  {
    (void)maxShell;
    dir = decomp_dir();
    std::filesystem::create_directories(dir);

    if (const char* e = std::getenv("SAM_DECOMP_CACHE_CAP"))
      cache_cap_ = std::strtoull(e, nullptr, 10);
    if (const char* e = std::getenv("SAM_DECOMP_NOCACHE_ABOVE"))
      nocache_bytes_ = std::strtoull(e, nullptr, 10);

    // Disk-back the big uncached decomps (see store/load_overflow). Per-rank
    // scratch file, truncated fresh each run. Off unless SAM_DECOMP_OVERFLOW set.
    if (std::getenv("SAM_DECOMP_OVERFLOW"))
    {
      int r = 0;
#ifdef USE_MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &r);
#endif
      overflow_path_ = dir + "/.overflow_rank" + std::to_string(r) + ".bin";
      overflow_out_.open(overflow_path_,
                         std::ios::binary | std::ios::out | std::ios::trunc);
      overflow_on_ = static_cast<bool>(overflow_out_);
    }

#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0)
      std::cerr << "Tensor-decomposition files: lazy load on demand" << std::endl;
#else
    std::cout << "Tensor-decomposition files: lazy load on demand" << std::endl;
#endif

    cache_.reserve(64);
  }

  std::pair<std::size_t, std::size_t> SSTDecomp_phys_db::cache_stats() const
  {
    std::shared_lock lock(cache_mutex_);
    std::size_t files = cache_.size(), records = 0;
    for (const auto& [name, df] : cache_)
    {
      std::lock_guard flock(df->file_mutex_);
      records += df->records.size();
    }
    return {files, records};
  }

  std::size_t SSTDecomp_phys_db::cache_bytes() const
  {
    std::shared_lock lock(cache_mutex_);
    std::size_t b = 0;
    for (const auto& [fn, df] : cache_) b += df->bytes;
    return b;
  }

  // Drop least-recently-used files until the live cache is under cache_cap_.
  // Caller guarantees no ExpansionRef is outstanding (chunk-boundary omp single),
  // so freeing a DecompFile's records can't dangle a borrowed pointer. New
  // records in an evicted file are simply not persisted (recomputed if ever
  // re-referenced); the on-disk copy is never touched here.
  std::size_t SSTDecomp_phys_db::evict_to_cap()
  {
    if (cache_cap_ == 0) return 0;

    std::unique_lock lock(cache_mutex_);

    std::size_t total = 0;
    for (const auto& [fn, df] : cache_) total += df->bytes;
    if (total <= cache_cap_) return 0;

    // Evict least-REUSED first (low refs), tie-break least-recently-used. The
    // giant high-shell decomps are touched ~once so they go first; the small
    // low-shell ones reused throughout the run stay resident.
    std::vector<std::tuple<std::size_t, std::size_t, std::string>> order;
    order.reserve(cache_.size());
    for (const auto& [fn, df] : cache_)
      order.emplace_back(df->refs.load(std::memory_order_relaxed),
                         df->last_access.load(std::memory_order_relaxed), fn);
    std::sort(order.begin(), order.end());

    std::size_t freed = 0;
    for (const auto& entry : order)
    {
      if (total <= cache_cap_) break;
      const std::string& fn = std::get<2>(entry);
      auto it = cache_.find(fn);
      if (it == cache_.end()) continue;
      const std::size_t b = it->second->bytes;
      total -= b;
      freed += b;
      cache_.erase(it);
    }

#if defined(__GLIBC__)
    // Return the freed pages to the OS. Without this glibc retains them, so RSS
    // does NOT follow cache_bytes down -- the retained memory grows with the
    // evict/grow churn and OOMs the node even though the cache is bounded. Safe
    // here: caller (chunk-boundary omp single) has all other threads parked.
    if (freed > 0) malloc_trim(0);
#endif
    return freed;
  }

  //=========================================================================
  // finalize: flush newly computed entries to disk, then release memory.
  //=========================================================================
  void SSTDecomp_phys_db::finalize()
  {
    std::unique_lock cache_lock(cache_mutex_);

#ifdef USE_MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_rank == 0)
      std::cerr << "Start destructing tensor-decomposition database"
                << std::endl;

    for (int turn = 0; turn < mpi_size; turn++)
    {
      if (turn > 0 && mpi_rank == turn)
      {
        int token;
        MPI_Recv(&token, 1, MPI_INT, turn - 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }

      if (mpi_rank == turn)
      {
        for (auto& [fn, file_ptr] : cache_)
        {
          auto& file = *file_ptr;
          std::lock_guard file_lock(file.file_mutex_);

          if (file.records.size() == file.initial_size) continue;

          if (turn == 0)
          {
            std::ofstream out(file.path, std::ios::binary | std::ios::trunc);
            if (!out)
            {
              std::cerr << "Warning: Cannot open decomposition file for "
                           "writing: "
                        << file.path << " on rank " << mpi_rank << std::endl;
              continue;
            }
            for (const auto& [key, rec] : file.records)
              writeRecord(out, key, *rec);
          }
          else
          {
            // erase keys already on disk, then append the remainder
            std::ifstream in(file.path, std::ios::binary);
            if (in)
            {
              while (true)
              {
                uint64_t qkey;
                utils::readBinary(in, qkey);
                if (!in) break;

                uint64_t nrows;
                uint8_t ncols;
                uint8_t nops;
                utils::readBinary(in, nrows);
                utils::readBinary(in, ncols);
                utils::readBinary(in, nops);
                size_t ncoeff = static_cast<size_t>(nrows) * ncols;
                size_t nop = static_cast<size_t>(nrows) * nops;
                in.seekg(ncoeff * sizeof(double) + nop * sizeof(uint16_t) +
                         nop * sizeof(uint8_t), std::ios::cur);

                if (!in) break;
                file.records.erase(DecompKey(qkey));
              }
              in.close();
            }

            if (file.records.empty()) continue;

            std::ofstream out(file.path, std::ios::binary | std::ios::app);
            if (!out)
            {
              std::cerr << "Warning: Cannot open decomposition file for "
                           "appending: "
                        << file.path << " on rank " << mpi_rank << std::endl;
              continue;
            }
            for (const auto& [key, rec] : file.records)
              writeRecord(out, key, *rec);
          }
        }

        if (turn < mpi_size - 1)
        {
          int token = 1;
          MPI_Send(&token, 1, MPI_INT, turn + 1, 0, MPI_COMM_WORLD);
        }
      }
    }

    if (mpi_rank == 0)
      std::cerr << "Finish destructing tensor-decomposition database"
                << std::endl;
#else
    std::cout << "Start destructing tensor-decomposition database"
              << std::endl;

    for (auto& [fn, file_ptr] : cache_)
    {
      auto& file = *file_ptr;
      std::lock_guard file_lock(file.file_mutex_);

      if (file.records.size() == file.initial_size) continue;

      std::ofstream out(file.path, std::ios::binary | std::ios::trunc);
      if (!out)
      {
        std::cerr << "Warning: Cannot open decomposition file for writing: "
                  << file.path << std::endl;
        continue;
      }

      for (const auto& [key, rec] : file.records)
        writeRecord(out, key, *rec);
    }
    std::cout << "Finish destructing tensor-decomposition database"
              << std::endl;
#endif

    cache_.clear();

    // Drop the per-rank overflow scratch file (within-run only; not persisted).
    if (overflow_on_)
    {
      std::lock_guard lock(overflow_mutex_);
      overflow_out_.close();
      std::remove(overflow_path_.c_str());
      overflow_index_.clear();
      overflow_on_ = false;
    }
  }

  //=========================================================================
  // get_view: cached expand() as a borrow-or-own Expansion handle.
  //=========================================================================
  SSTDecomp_phys_db::ExpansionRef
  SSTDecomp_phys_db::get_view(const U3xU4::SingleShellTensor_phys& t)
  {
    // A single creation/annihilation operator expands trivially: one term, one
    // operator, coefficient +-1. Build that Expansion directly -- no expand()
    // machinery -- and let the handle own it. Not cached (cheap, and would only
    // pollute the on-disk cache).
    if (t.num_ads() + t.num_ats() == 1)
    {
      auto e = std::make_unique<Expansion>();
      e->coeffs.ncols = 1;
      e->nops = 1;
      e->labels = &t.labels();
      if (t.num_ads() == 1)
      {
        U3xU4::ad_phys ad(t.shell(), t.LL(), t.JJ(), t.MMJ(), t.MMT(), t.labels());
        double c = 1.0;
        e->coeffs.push_row(&c);
        e->append_op(ad.getIndex(), /*ad=*/true);
      }
      else // a single annihilation operator carries the turn-to-a~ phase
      {
        U3xU4::at_phys at(t.shell(), t.LL(), t.JJ(), t.MMJ(), t.MMT(), t.labels());
        double c = at.turnToAnnihilation();
        e->coeffs.push_row(&c);
        e->append_op(at.getIndex(), /*ad=*/false);
      }
      return ExpansionRef(std::move(e));
    }

    // A single-shell tensor with all four operators in one shell
    // (num_ads + num_ats == 4) IS a complete one-shell two-body tensor -- it
    // never recurs as a shared building block of a larger multishell tensor, so
    // it gets exactly one lookup and a cache would never be hit. These are the
    // bulk of decomp_phys (~1666 files / ~57 GB at maxShell 12), and once loaded
    // they grow the in-memory cache until it OOMs. Compute on the fly and own the
    // result; only the reusable (<= 3) cases are cached below. Bit-identical:
    // expand() is deterministic, so this equals what the cache would have held.
    if (t.num_ads() + t.num_ats() == 4)
    {
      auto e = std::make_unique<Expansion>(t.expand());
      e->labels = &t.labels();
      return ExpansionRef(std::move(e));
    }

    std::string file = filename(t);
    DecompKey key(t.LL(), t.SS(), t.TT(), t.MMT(), t.JJ(), t.MMJ());
    auto& df = getFile(file);

    // Hit: borrow the stable cached Expansion directly (zero copy).
    if (Expansion* e = df.get_ptr(key, t.labels()))
      return ExpansionRef(e);

    // Big decomps aren't kept in the cache (below). With the overflow store on
    // they were written to disk instead of discarded, so reload by seek rather
    // than recompute. Owned (a private copy; not a stable cache pointer).
    if (overflow_on_)
      if (auto e = load_overflow(file, key))
      {
        e->labels = &t.labels();
        return ExpansionRef(std::move(e));
      }

    // Miss: compute WITHOUT a global lock (expand() is thread-safe; it sets the
    // Expansion's labels). Big high-shell decomps are used ~once, so caching
    // them only floods memory and forces eviction: if the result exceeds
    // SAM_DECOMP_NOCACHE_ABOVE bytes, keep it TRANSIENT (own it, never cached).
    // Otherwise insert and borrow the stable pointer (a duplicate compute on a
    // race is harmless -- append_or_get_ptr keeps the first entry).
    Expansion ex = t.expand();
    if (nocache_bytes_ && expansion_bytes(ex) > nocache_bytes_)
    {
      // Persist to the overflow store (if on) so a re-reference reloads it
      // instead of recomputing this expensive expand().
      if (overflow_on_) store_overflow(file, key, ex);
      auto owned = std::make_unique<Expansion>(std::move(ex));
      owned->labels = &t.labels();
      return ExpansionRef(std::move(owned));
    }
    Expansion* e = df.append_or_get_ptr(key, std::move(ex));
    return ExpansionRef(e);
  }
} // namespace U3xU4
