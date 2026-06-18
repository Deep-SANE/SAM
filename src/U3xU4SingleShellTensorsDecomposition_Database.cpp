#include <cctype>
#include <mutex>
#include <utility>

#include <U3xU4SingleShellTensorsDecomposition_Database.h>

#ifdef USE_MPI
#include <mpi.h>
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
  // pack / unpack: live Expansion <-> compact Record.
  //=========================================================================
  SSTDecomp_phys_db::Record SSTDecomp_phys_db::pack(const Expansion& e)
  {
    const auto& [coeffs, adsArr, atsArr] = e;

    Record rec;

    // coeff rows
    rec.nrows = coeffs.size();
    rec.ncols = coeffs.empty()
                  ? 0u
                  : static_cast<uint32_t>(coeffs.front().size());
    rec.coeffs.resize(static_cast<size_t>(rec.nrows) * rec.ncols);
    for (uint64_t i = 0; i < rec.nrows; ++i)
    {
      assert(static_cast<uint32_t>(coeffs[i].size()) == rec.ncols);
      for (uint32_t c = 0; c < rec.ncols; ++c)
        rec.coeffs[i * rec.ncols + c] = coeffs[i](c);
    }

    // ad operators: table index = full state. adsArr is either empty (pure
    // multi-annihilation tensor) or has one row per coeff row.
    rec.n_ad_rows = static_cast<uint32_t>(adsArr.size());
    rec.nad = adsArr.empty() ? 0
                             : static_cast<uint16_t>(adsArr.front().size());
    rec.ad_idx.reserve(static_cast<size_t>(rec.n_ad_rows) * rec.nad);
    for (const auto& row : adsArr)
    {
      assert(row.size() == rec.nad);
      for (const auto& ad : row)
        rec.ad_idx.push_back(ad.getIndex());
    }

    // a operators: table index only -- every cached a^tilde is already an
    // annihilation operator (tensor == false), so the flag carries no info.
    rec.n_at_rows = static_cast<uint32_t>(atsArr.size());
    rec.nat = atsArr.empty() ? 0
                             : static_cast<uint16_t>(atsArr.front().size());
    rec.at_idx.reserve(static_cast<size_t>(rec.n_at_rows) * rec.nat);
    for (const auto& row : atsArr)
    {
      assert(row.size() == rec.nat);
      for (const auto& at : row)
      {
        assert(!at.isTensor());
        rec.at_idx.push_back(at.getIndex());
      }
    }

    return rec;
  }

  SSTDecomp_phys_db::Expansion
  SSTDecomp_phys_db::unpack(const Record& rec, const HO::Table_nljmjmt& labels)
  {
    std::vector<Eigen::RowVectorXd> coeffs;
    coeffs.reserve(rec.nrows);
    for (uint64_t i = 0; i < rec.nrows; ++i)
    {
      Eigen::RowVectorXd v(rec.ncols);
      for (uint32_t c = 0; c < rec.ncols; ++c)
        v(c) = rec.coeffs[i * rec.ncols + c];
      coeffs.push_back(std::move(v));
    }

    std::vector<U3xU4::ads_phys> adsArr;
    adsArr.reserve(rec.n_ad_rows);
    {
      size_t p = 0;
      for (uint32_t r = 0; r < rec.n_ad_rows; ++r)
      {
        U3xU4::ads_phys row;
        row.reserve(rec.nad);
        for (uint16_t k = 0; k < rec.nad; ++k)
          row.emplace_back(rec.ad_idx[p++], labels);
        adsArr.push_back(std::move(row));
      }
    }

    std::vector<U3xU4::ats_phys> atsArr;
    atsArr.reserve(rec.n_at_rows);
    {
      size_t p = 0;
      for (uint32_t r = 0; r < rec.n_at_rows; ++r)
      {
        U3xU4::ats_phys row;
        row.reserve(rec.nat);
        for (uint16_t k = 0; k < rec.nat; ++k)
          row.emplace_back(rec.at_idx[p++], labels);
        atsArr.push_back(std::move(row));
      }
    }

    return std::make_tuple(std::move(coeffs), std::move(adsArr),
                           std::move(atsArr));
  }

  //=========================================================================
  // Binary record I/O (terminated by EOF, so nrows==0 is a valid record).
  //=========================================================================
  void SSTDecomp_phys_db::writeRecord(std::ostream& s, const DecompKey& key,
                                      const Record& rec)
  {
    qcx_utils::writeBinary(s, key.qkey);
    qcx_utils::writeBinary(s, rec.nrows);
    qcx_utils::writeBinary(s, rec.ncols);
    qcx_utils::writeBinary(s, rec.n_ad_rows);
    qcx_utils::writeBinary(s, rec.nad);
    qcx_utils::writeBinary(s, rec.n_at_rows);
    qcx_utils::writeBinary(s, rec.nat);
    qcx_utils::writeBinary(s, rec.coeffs);
    qcx_utils::writeBinary(s, rec.ad_idx);
    qcx_utils::writeBinary(s, rec.at_idx);
  }

  //=========================================================================
  // DecompFile
  //=========================================================================
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
      qcx_utils::readBinary(file, qkey);
      if (!file) break; // clean EOF

      Record rec;
      qcx_utils::readBinary(file, rec.nrows);
      qcx_utils::readBinary(file, rec.ncols);
      qcx_utils::readBinary(file, rec.n_ad_rows);
      qcx_utils::readBinary(file, rec.nad);
      qcx_utils::readBinary(file, rec.n_at_rows);
      qcx_utils::readBinary(file, rec.nat);

      size_t ncoeff = static_cast<size_t>(rec.nrows) * rec.ncols;
      if (ncoeff)
      {
        rec.coeffs.resize(ncoeff);
        qcx_utils::readBinary(file, rec.coeffs);
      }

      size_t nadl = static_cast<size_t>(rec.n_ad_rows) * rec.nad;
      if (nadl)
      {
        rec.ad_idx.resize(nadl);
        qcx_utils::readBinary(file, rec.ad_idx);
      }

      size_t natl = static_cast<size_t>(rec.n_at_rows) * rec.nat;
      if (natl)
      {
        rec.at_idx.resize(natl);
        qcx_utils::readBinary(file, rec.at_idx);
      }

      if (!file) break; // truncated / corrupt -> stop loading

      records.emplace(DecompKey(qkey), std::move(rec));
    }

    file.close();
    initial_size = records.size();
  }

  bool SSTDecomp_phys_db::DecompFile::contains(const DecompKey& key) const
  {
    std::lock_guard lock(file_mutex_);
    return records.find(key) != records.end();
  }

  std::optional<SSTDecomp_phys_db::Record>
  SSTDecomp_phys_db::DecompFile::get_if_exist(const DecompKey& key) const
  {
    std::lock_guard lock(file_mutex_);
    auto it = records.find(key);
    if (it != records.end()) return it->second;
    return std::nullopt;
  }

  SSTDecomp_phys_db::Record
  SSTDecomp_phys_db::DecompFile::append_or_get(const DecompKey& key,
                                               Record&& rec)
  {
    std::lock_guard lock(file_mutex_);
    auto [it, inserted] = records.emplace(key, std::move(rec));
    return it->second; // existing one wins on race
  }

  //=========================================================================
  // getFile: top-level cache with shared/exclusive locking (as RME_db).
  //=========================================================================
  SSTDecomp_phys_db::DecompFile&
  SSTDecomp_phys_db::getFile(const std::string& file)
  {
    {
      std::shared_lock lock(cache_mutex_);
      auto it = cache_.find(file);
      if (it != cache_.end()) return *(it->second);
    }

    auto path = dir + "/" + file;
    auto new_file = std::make_unique<DecompFile>(path);

    std::unique_lock lock(cache_mutex_);
    auto it = cache_.find(file);
    if (it != cache_.end()) return *(it->second);

    auto [it_inserted, inserted] = cache_.emplace(file, std::move(new_file));
    return *(it_inserted->second);
  }

  //=========================================================================
  // Constructors: discover the storage directory and preload existing files.
  //=========================================================================
  namespace
  {
    std::string decomp_dir()
    {
#ifdef SAM_DATA_DIR
      const char* sam_data = std::getenv("SAM_DATA_DIR");
      if (!sam_data)
        throw std::runtime_error("Environment variable SAM_DATA_DIR isn't set");
      return std::string(sam_data) + "/decomp_phys";
#else
      const char* sam_dir = std::getenv("SAM_DIR");
      if (!sam_dir)
        throw std::runtime_error("Environment variable SAM_DIR isn't set");
      return std::string(sam_dir) + "/data/decomp_phys";
#endif
    }

    int leading_shell(const std::string& filename)
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

  SSTDecomp_phys_db::SSTDecomp_phys_db(int maxShell)
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

      if (leading_shell(fn) > maxShell) continue;

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
              writeRecord(out, key, rec);
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
                qcx_utils::readBinary(in, qkey);
                if (!in) break;

                Record rec;
                qcx_utils::readBinary(in, rec.nrows);
                qcx_utils::readBinary(in, rec.ncols);
                qcx_utils::readBinary(in, rec.n_ad_rows);
                qcx_utils::readBinary(in, rec.nad);
                qcx_utils::readBinary(in, rec.n_at_rows);
                qcx_utils::readBinary(in, rec.nat);
                size_t ncoeff = static_cast<size_t>(rec.nrows) * rec.ncols;
                if (ncoeff)
                  in.seekg(ncoeff * sizeof(double), std::ios::cur);
                size_t nadl = static_cast<size_t>(rec.n_ad_rows) * rec.nad;
                if (nadl)
                  in.seekg(nadl * sizeof(uint16_t), std::ios::cur);
                size_t natl = static_cast<size_t>(rec.n_at_rows) * rec.nat;
                if (natl)
                  in.seekg(natl * sizeof(uint16_t), std::ios::cur);

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
              writeRecord(out, key, rec);
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
        writeRecord(out, key, rec);
    }
    std::cout << "Finish destructing tensor-decomposition database"
              << std::endl;
#endif

    cache_.clear();
  }

  //=========================================================================
  // get: cached SingleShellTensor_phys::expand().
  //=========================================================================
  SSTDecomp_phys_db::Expansion
  SSTDecomp_phys_db::get(const U3xU4::SingleShellTensor_phys& t)
  {
    // A single creation or annihilation operator expands trivially (one-line
    // Tad/Tat base kernel) -- not worth caching. Everything else, including
    // pure multi-creation / multi-annihilation tensors (which the Record's
    // independent ad/at row counts handle), goes through the cache.
    if ((t.num_ads() == 0 and t.num_ats() == 1) or
        (t.num_ads() == 1 and t.num_ats() == 0))
      return t.expand();

    std::string file = filename(t);
    DecompKey key(t.LL(), t.SS(), t.TT(), t.MMT(), t.JJ(), t.MMJ());

    auto& df = getFile(file);

    // Fast path: a hit, served concurrently.
    if (auto rec = df.get_if_exist(key))
      return unpack(*rec, t.labels());

    // Miss: compute WITHOUT a global lock. expand() is thread-safe -- it only
    // reads the threadprivate cgc/Racah/Fano/Wigner caches set up by
    // sam::initialize_thread() -- so distinct tensors expand in parallel,
    // exactly as the original uncached setTensorStrength loop did. Only the
    // cache insert is locked (per file). A duplicate compute on a race is
    // harmless: append_or_get keeps the first entry and our copy is identical.
    auto e = t.expand();
    df.append_or_get(key, pack(e));
    return e;
  }
} // namespace U3xU4
