#pragma once

#include <utils.h>
#include <u3lib.h>
#include <u4lib.h>

#include <DataDirectory.h>
#include <HOutilities.h>

namespace sam
{
  // u4::initialize cascades down to u3 and the utils foundation, so we only
  // init the top of the chain here (+ SAM-specific caches).
  inline void initialize(int* argc, char*** argv)
  {
    u4::initialize(argc, argv);   // cascades: u4 -> u3 -> utils
#ifdef UNCOUPLE_ME_CACHE
    HO::initialize_twbme_cache();
#endif
  }

  inline void finalize()
  {
#ifdef UNCOUPLE_ME_CACHE
    HO::finalize_twbme_cache();
#endif
    u4::finalize();   // cascades: u4 -> u3 -> utils
  }

  inline void initialize_thread()
  {
    u4::initialize_thread();   // cascades: u4 -> u3 -> utils
  }

  inline void finalize_thread()
  {
    u4::finalize_thread();   // cascades: u4 -> u3 -> utils
  }

  inline void clear() 
  {
    u4::clear();
    u3::clear();
  }
} // namespace sam
