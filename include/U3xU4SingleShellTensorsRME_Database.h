#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <shared_mutex>
#include <string>

#include <boost/container/flat_map.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

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
    // std::string filename(const U3xU4::SingleShellTensor& t) const;

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

    // File for a U3xU4::SingleShellTensor
    struct RMEFile
    {
      std::string path;

      // map key -> rmes
      boost::container::flat_map<RMEKey, std::vector<double>> rmes_map;
      size_t initial_size;

      mutable std::mutex file_mutex_;

      // constructor
      explicit RMEFile(const std::string& path_);

      // check if key exists
      bool contains(const RMEKey& key) const;

      // get rmes from key
      std::vector<double> get_if_exist(const RMEKey& key) const;

      // add new key and rmes
      std::vector<double> append_or_get(
        const RMEKey& key, const std::vector<double>& rmes);
    };

    // Hash table of files (top-level cache)
    boost::unordered_flat_map<std::string, std::unique_ptr<RMEFile>> cache_;

    mutable std::shared_mutex cache_mutex_;

    RMEFile& getFile(const std::string& file);

  public:
  	RME_db();
    RME_db(int maxShell);
    ~RME_db();

  	sam::RME get(const U3xU4::SingleShellTensor& t,
                 const U3xU4::GammaElement& f, 
                 const U3xU4::GammaElement& i,
                 const U3xU4::Table_gelfand& labels);
  };
} // namespace U3xU4
