#pragma once

#include <filesystem>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <U3xU4GammaOmega.h>
#include <U3xU4irrep.h>

#include "UNxU4SlaterDeterminant.h"

namespace U3xU4
{
  // find all SDs in the expansion of the HWS
  UNxU4::SlaterDeterminants expand_to_SD(const U3xU4::GammaElement& G, 
    const u3::gelfands& sps3, const u4::gelfands& sps4);

  // find the expansion of the HWS
  std::pair<UNxU4::SlaterDeterminants, Eigen::MatrixXd>
  HWS_to_SD(const U3xU4::GammaElement& G,
    const u3::gelfands& sps3, const u4::gelfands& sps4);

  // find the HWS in the database
  std::filesystem::path get_HWS_file(const U3xU4::GammaElement& G, 
    const u3::gelfands& sps3, const u4::gelfands& sps4);

  std::pair<UNxU4::SlaterDeterminants, Eigen::MatrixXd>
  read_HWS_data(const std::filesystem::path& path);

} // namespace U3xU4
