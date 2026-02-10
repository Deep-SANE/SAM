#include <U3xU4SingleShellTensorsRME_Database.h>

namespace U3xU4
{
  std::string RME_db::filename(const U3xU4::SingleShellTensor& t) const
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

    return dir + "/" + shell + ads + ats + irr + ".bin";
  }

  RME_db::RMEKey::RMEKey(const U3xU4::GammaElement& f,
                         const U3xU4::GammaElement& i)
  {
    // Pack SU(3) labels into 32-bit key
    std::array<qcx_utils::UInt8, 4> su3key_safe = {f.lam(), f.mu(), i.lam(),
                                                   i.mu()};
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

  RME_db::RMEFile::RMEFile(const std::string& path)
  {
    // Ensure file exists
    if (!std::filesystem::exists(path))
      std::ofstream(path, std::ios::binary).close();

    file.open(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file)
      throw std::runtime_error("Cannot open RME file: " + path);

    // Load all RMEs into memory
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

    file.clear();                 // clear EOF
    file.seekp(0, std::ios::end); // ready for appending
  }

  bool RME_db::RMEFile::contains(const RMEKey& key) const
  {
    return rmes_map.find(key) != rmes_map.end();
  }

  const std::vector<double>& RME_db::RMEFile::get(const RMEKey& key) const
  {
    return rmes_map.at(key);
  }

  // add new key & rmes to map
  void RME_db::RMEFile::append(const RMEKey& key,
                               const std::vector<double>& rmes)
  {
    if (contains(key))
      return;

    qcx_utils::writeBinary(file, key.su3key);
    qcx_utils::writeBinary(file, key.su4key);

    uint64_t num = rmes.size();
    qcx_utils::writeBinary(file, num);
    for (double v : rmes)
      qcx_utils::writeBinary(file, v);

    file.flush();
    rmes_map.emplace(key, rmes);
  }

  RME_db::RMEFile& RME_db::getFile(const std::string& file)
  {
    auto it = cache_.find(file);
    if (it == cache_.end())
    {
      auto [new_it, inserted] = cache_.emplace(file, RMEFile(file));
      return new_it->second;
    }
    return it->second;
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
    auto key = RMEKey(f, i);
    auto file = filename(t);
    auto& rme_file = getFile(file);

    size_t t_amax = t.rho0max() * t.eta0max();
    auto [rhomax, etamax] = U3xU4::RhoEta(i, t.T(), f);

    if (rme_file.contains(key))
    {
      // std::cout << "Found" << std::endl;
      std::vector<double> rmes = rme_file.get(key);

      return {std::move(rmes),
              af_max,
              ai_max,
              t_amax,
              static_cast<size_t>(rhomax),
              static_cast<size_t>(etamax)};
    }
    else
    {
      // std::cout << "Not Found" << std::endl;
      std::vector<double> rmes = t.RME(f, i, labels).get();
      rme_file.append(key, rmes);

      return {std::move(rmes),
              af_max,
              ai_max,
              t_amax,
              static_cast<size_t>(rhomax),
              static_cast<size_t>(etamax)};
    }
  }
} // namespace U3xU4
