#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sam
{
  // root of SAM's data tree: operator expansions, rmes, hws, decompositions
  // and interaction files all live under this directory.
  //
  // resolution order:
  //   1. the SAM_DATA_DIR environment variable (run-time override, e.g. a
  //      per-job scratch directory on a cluster);
  //   2. the path chosen at `make config`, built in as SAM_DEFAULT_DATA_DIR.
  //
  // `make config` builds in the real (resolved) path, never the SAM/data
  // link, so compute nodes do not need to go through the home directory.
  inline std::string data_dir()
  {
    if (const char* env = std::getenv("SAM_DATA_DIR"); env && *env)
      return std::string(env);
#ifdef SAM_DEFAULT_DATA_DIR
    return std::string(SAM_DEFAULT_DATA_DIR);
#else
    throw std::runtime_error(
      "Sorry, the SAM data directory is unknown. Run 'make config' and "
      "rebuild, or set the SAM_DATA_DIR environment variable.");
#endif
  }
}
