#pragma once

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <U3basis.h>
#include <U4basis.h>

#include "UNxU4SlaterDeterminant.h"

namespace U3xU4
{
  // find all SDs in the expansion of the HWS
  std::vector<UNxU4::SlaterDeterminant> expand_to_SD(const u3::irrep& b,
                                                     const u4::irrep& p);

  // find the expansion of the HWS
  std::pair<std::vector<UNxU4::SlaterDeterminant>, Eigen::MatrixXd>
  HWS_to_SD(const u3::irrep& b, const u4::irrep& p);

} // namespace U3xU4