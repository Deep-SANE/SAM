#pragma once

#include <qcx_utils.h>
#include <u3lib.h>
#include <u4lib.h>

namespace sam
{
  inline void initialize(int* argc, char*** argv)
  {
    qcx_utils::initialize(argc, argv);
  	u3::initialize();
  	u4::initialize();
  }

  inline void finalize() 
  {
    u4::finalize();
    u3::finalize();  	
  	qcx_utils::finalize();
  }

  inline void initialize_thread() 
  {
    qcx_utils::initialize_thread();
    u3::initialize_thread();
    u4::initialize_thread();
  }

  inline void finalize_thread()
  {
    u4::finalize_thread();
    u3::finalize_thread();
    qcx_utils::finalize_thread();
  }
} // namespace sam
