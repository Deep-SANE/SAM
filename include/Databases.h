#ifndef SAM_DATABASIS_H
#define SAM_DATABASIS_H

#include <so3Coeffs/SO3_Racah.h>
#include <so3Coeffs/SO3_CGC.h>
#include <U3SO3xU3SO3.h>
#include <U3U2xU3U2.h>
#include <U4SU2SU2xU4SU2SU2.h>
#include <U4U3xU4U3.h>

namespace sam
{
  struct Databases
  {
    qcx_utils::SO3_CGC cgcs;
    qcx_utils::SO3_Racah racah;
    qcx_utils::SO3_Racah9 racah9;
    u3::WigU3U2_db wigU3U2db;
    u3::WigU3SO3_db wigU3SO3db;
    u4::WigU4U3_db wigU4U3db;
    u4::WigU4SU2SU2_db wigU4SU2SU2db;

    Databases()
        : wigU3U2db(cgcs),
          wigU3SO3db(cgcs, wigU3U2db),
          wigU4U3db(cgcs, wigU3U2db),
          wigU4SU2SU2db(cgcs, wigU3U2db, wigU4U3db)
    { }

    Databases(const Databases&) = delete;
    Databases(Databases&&) = delete;

    Databases& operator=(const Databases&) = delete;
    Databases& operator=(Databases&&) = delete;
  };
} // namespace sam

#endif
