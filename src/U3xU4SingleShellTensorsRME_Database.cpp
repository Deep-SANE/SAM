#include <U3xU4SingleShellTensorsRME_Database.h>

namespace U3xU4
{
  RME_db::RMEKey::RMEKey(const U3xU4::GammaElement& f,
                         const U3xU4::GammaElement& i)
  {
    // Pack SU(3) labels into 32-bit key
    std::array<qcx_utils::UInt8, 4> su3key_safe = {f.lam(), f.mu(), 
                                                   i.lam(), i.mu()};
    su3key = (uint32_t(su3key_safe[0]) << 24) |
             (uint32_t(su3key_safe[1]) << 16) |
             (uint32_t(su3key_safe[2]) << 8) | uint32_t(su3key_safe[3]);

    // Pack SU(4) labels into 64-bit key
    std::array<qcx_utils::UInt8, 6> su4key_safe = {f.P1(), f.P2(), f.P3(),
                                                   i.P1(), i.P2(), i.P3()};
    su4key =
        (uint64_t(su4key_safe[0]) << 56) | (uint64_t(su4key_safe[1]) << 48) |
        (uint64_t(su4key_safe[2]) << 40) | (uint64_t(su4key_safe[3]) << 32) |
        (uint64_t(su4key_safe[4]) << 24) | (uint64_t(su4key_safe[5]) << 16);
  }

  RME_db::RMEFile::RMEFile(const std::string& path_)
    : path(path_)
  {
    // Open file if exists
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
      initial_size = 0;
      return;
    }

    // Load all RMEs into memory
    // std::cout << "Start loading RMEFile" << std::endl;
    file.seekg(0);
    while (true)
    {
      uint32_t su3;
      qcx_utils::readBinary(file, su3);

      uint64_t su4;
      qcx_utils::readBinary(file, su4);

      uint64_t num;
      qcx_utils::readBinary(file, num);
      if (num == 0) break;

      std::vector<double> rmes(num);
      qcx_utils::readBinary(file, rmes);

      rmes_map.emplace(RMEKey(su3, su4), std::move(rmes));

      if (!file) break;
    }
    // std::cout << "Finish loading RMEFile" << std::endl;

    // Finalize
    file.close();
    initial_size = rmes_map.size();
  }

  bool RME_db::RMEFile::contains(const RMEKey& key) const
  {
    std::lock_guard lock(file_mutex_);
    return rmes_map.find(key) != rmes_map.end();
  }

  std::vector<double> 
  RME_db::RMEFile::get_if_exist(const RMEKey& key) const
  {
    std::lock_guard lock(file_mutex_);

    auto it = rmes_map.find(key);
    if (it != rmes_map.end()) return it->second;

    return {};
  }

  // add new key & rmes to map
  std::vector<double> RME_db::RMEFile::append_or_get(
    const RMEKey& key, const std::vector<double>& rmes)
  {
    std::lock_guard lock(file_mutex_);
    
    auto [it, inserted] = rmes_map.emplace(key, rmes);

    return it->second;   // if already existed, returns old one
  }

  RME_db::RMEFile& RME_db::getFile(const std::string& file)
  {
    // Fast read lock
    {
      std::shared_lock lock(cache_mutex_);
      auto it = cache_.find(file);
      if (it != cache_.end()) return *(it->second);
    }

    // Create file OUTSIDE lock
    auto path = dir + "/" + file;
    auto new_file = std::make_unique<RMEFile>(path);

    // Now exclusive lock to insert
    std::unique_lock lock(cache_mutex_);

    // if some thread inserted meanwhile
    auto it = cache_.find(file);
    if (it != cache_.end()) return *(it->second);  

    auto [it_inserted, inserted] =
      cache_.emplace(file, std::move(new_file));

    return *(it_inserted->second);
  }

  RME_db::RME_db()
  {
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");

    dir = std::string(sam_dir) + "/data/rmes";
    std::filesystem::create_directories(dir);

    // Preload all existing files
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
      if (entry.path().extension() != ".bin") continue;

      std::string filename = entry.path().filename().string();
      if (filename.empty() || filename[0] != 'n') continue;

      std::string path = entry.path().string();

      cache_.emplace(filename, std::make_unique<RMEFile>(path));
    }

    // Reserve to avoid rehash under concurrency
    cache_.reserve(cache_.size() + 16);
  }

  RME_db::RME_db(int maxShell)
  {
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");

    dir = std::string(sam_dir) + "/data/rmes";
    std::filesystem::create_directories(dir);

    // Preload all existing files
    // std::cout << "Start loading existing files" << std::endl;
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
      if (entry.path().extension() != ".bin") continue;

      std::string filename = entry.path().filename().string();
      if (filename.empty() || filename[0] != 'n') continue;

      std::string path = entry.path().string();

      size_t pos = 1;
      int shell = 0;

      while (pos < filename.size())
      {
        if (std::isdigit(filename[pos]))
        {
          shell = shell * 10 + (filename[pos] - '0');
          pos++;
        }
        else
          break;
      }

      if (shell > maxShell) continue;

      cache_.emplace(filename, std::make_unique<RMEFile>(path));
    }
    // std::cout << "Finish loading existing files" << std::endl;

    // Reserve to avoid rehash under concurrency
    cache_.reserve(cache_.size() + 16);
  }

  RME_db::~RME_db()
  {
    std::shared_lock cache_lock(cache_mutex_);

    for (auto& [filename, file_ptr] : cache_)
    {
      auto& file = *file_ptr;

      std::lock_guard file_lock(file.file_mutex_);

      if (file.rmes_map.size() == file.initial_size) continue;

      std::ofstream out(file.path, std::ios::binary | std::ios::trunc);
      if (!out)
      {
        std::cerr << "Warning: Cannot open RME file for append: " 
                  << file.path << std::endl;
        continue;
      }

      for (const auto& [key, rmes] : file.rmes_map)
      {
        qcx_utils::writeBinary(out, key.su3key);
        qcx_utils::writeBinary(out, key.su4key);

        uint64_t num = rmes.size();
        qcx_utils::writeBinary(out, num);
        qcx_utils::writeBinary(out, rmes);
      }
    }
  }

  sam::RME RME_db::get(const U3xU4::SingleShellTensor& t,
                       const U3xU4::GammaElement& f,
                       const U3xU4::GammaElement& i,
                       const U3xU4::Table_gelfand& labels)
  {
    // std::cout << "Inside RME_db" << std::endl;

    // (1) Check if this RME exists
    if (t.RME_exist(f, i) == false)
      return {};

    size_t ai_max = i.alphamax();
    size_t af_max = f.alphamax();

    // (2) If this is an identity tensor
    if (t.identity() && i == f)
    {
      assert(t.Omega_ads().empty());
      assert(t.Omega_ats().empty());
      assert(t.T() == U3xU4::irrep(0, 0, 0, 0, 0, 0, 0));
      assert(t.rho0max() == 1);
      assert(t.eta0max() == 1);

      // Now calculate RMEs
      std::vector<double> rmes;
      for (int af = 0; af < af_max; af++)
        for (int ai = 0; ai < ai_max; ai++)
          rmes.push_back((ai == af) ? 1.0 : 0.0);

      return {std::move(rmes), af_max, ai_max, 1, 1, 1};
    }

    // (3) If this is not an identity tensor

    int A_i = i.particles(), A_f = f.particles(), n = t.shell();

    // (3.1) Special case: <1||a+||0>

    if (A_f == 1 && A_i == 0)
    {
      assert(t.num_ads() == 1);
      assert(t.num_ats() == 0);
      assert(af_max == 1);
      assert(ai_max == 1);

      return {{1.0}, 1, 1, 1, 1, 1};
    }

    // (3.2) Special case: <0||a||1>

    if (A_f == 0 && A_i == 1)
    {
      assert(t.num_ads() == 0);
      assert(t.num_ats() == 1);
      assert(af_max == 1);
      assert(ai_max == 1);

      return {{std::sqrt(2.0 * (n + 1.0) * (n + 2.0))}, 1, 1, 1, 1, 1};
    }

    // (3.3) Special case: <2||a+ a+||0>

    if (A_f == 2 && A_i == 0)
    {
      assert(t.num_ads() == 2);
      assert(t.num_ats() == 0);
      assert(af_max == 1);
      assert(ai_max == 1);

      return {{SQRT2}, 1, 1, 1, 1, 1};
    }

    // (3.4) Special case: <0||a a||2>

    if (A_f == 0 && A_i == 2)
    {
      assert(t.num_ads() == 0);
      assert(t.num_ats() == 2);
      assert(af_max == 1);
      assert(ai_max == 1);

      return {{-SQRT2 * std::sqrt(i.dim())}, 1, 1, 1, 1, 1};
    }

    // (3.5) General case
    RMEKey key(f, i);
    auto& file = t.filename();
    auto& rme_file = getFile(file);

    size_t t_amax = t.rho0max() * t.eta0max();
    auto [rhomax, etamax] = U3xU4::RhoEta(i, t.T(), f);

    // First check under lock (inside get_if_exist)
    {
      std::vector<double> rmes = rme_file.get_if_exist(key);

      if (!rmes.empty())
      {
        // std::cout << "Found" << std::endl;
        return {std::move(rmes),
                af_max,
                ai_max,
                t_amax,
                static_cast<size_t>(rhomax),
                static_cast<size_t>(etamax)};
      }
    }
    
    // Compute outside lock
    // std::cout << "Compute" << std::endl;
    auto rmes = t.RME(f, i, labels);
    
    // Insert (second check inside append_or_get)
    auto final = rme_file.append_or_get(key, rmes.data());

    // std::cout << "Return computed" << std::endl;
    return {std::move(final),
            af_max,
            ai_max,
            t_amax,
            static_cast<size_t>(rhomax),
            static_cast<size_t>(etamax)};
  }
} // namespace U3xU4
