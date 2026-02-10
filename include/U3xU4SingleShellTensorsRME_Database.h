#pragma once

#include <filesystem>
#include <fstream>
#include <string>

#include <qcx_Cache.h>
#include <qcx_readwrite.h>

#include <RME.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  class RME_db
  {
  private:
    std::string dir;

    // Get filename for a tensor
    std::string filename(const U3xU4::SingleShellTensor& t) const;

    // Key for RME
    struct RMEKey
    {
      uint32_t su3key;
      uint64_t su4key;

      RMEKey() = default;

      RMEKey(uint32_t su3key_, uint64_t su4key_) 
        : su3key(su3key_), su4key(su4key_) { }

      RMEKey(const U3xU4::GammaElement& f, const U3xU4::GammaElement& i);

      auto operator<=>(const RMEKey&) const = default;
    };

    // Cached file streams
    struct RMEFile
    {
      std::fstream file;

      // map key -> rmes
      std::map<RMEKey, std::vector<double>> rmes_map;

      // constructor
      RMEFile(const std::string& path);

      // check if key exists
      bool contains(const RMEKey& key) const;

      // get rmes from key
      const std::vector<double>& get(const RMEKey& key) const;

      // add new key and rmes
      void append(const RMEKey& key, const std::vector<double>& rmes);
    };

    // Hash table of file streams
    std::unordered_map<std::string, RMEFile> cache_;
    RMEFile& getFile(const std::string& file);

  public:
  	RME_db()
    {
      const char* sam_dir = std::getenv("SAM_DIR");
      if (!sam_dir)
        throw std::runtime_error("Environment variable SAM_DIR not set");

      dir = std::string(sam_dir) + "/data/rmes";
      std::filesystem::create_directories(dir);
    }

  	sam::RME get(const U3xU4::SingleShellTensor& t,
                 const U3xU4::GammaElement& f, 
                 const U3xU4::GammaElement& i,
                 const U3xU4::Table_gelfand& labels);
  };
} // namespace U3xU4
