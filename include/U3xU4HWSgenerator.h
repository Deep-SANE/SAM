#pragma once

#include <filesystem>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <U3xU4Irrep.h>

#include "UNxU4SlaterDeterminant.h"

namespace U3xU4
{
  // find all SDs in the expansion of the HWS
  std::vector<UNxU4::SlaterDeterminant>
  expand_to_SD(std::size_t A, std::size_t n, std::size_t N,
               const U3xU4::irrep& x, const std::vector<u3::gelfand>& sps3,
               const std::vector<u4::gelfand>& sps4);

  // find the expansion of the HWS
  std::pair<std::vector<UNxU4::SlaterDeterminant>, Eigen::MatrixXd>
  HWS_to_SD(int A, int n, const U3xU4::irrep& x,
            const std::vector<u3::gelfand>& sps3,
            const std::vector<u4::gelfand>& sps4);

  // find the HWS in the database
  std::filesystem::path get_HWS_file(int A, int n, const U3xU4::irrep& x,
                                     const std::vector<u3::gelfand>& sps3,
                                     const std::vector<u4::gelfand>& sps4);

  std::pair<std::vector<UNxU4::SlaterDeterminant>,
            std::vector<Eigen::RowVectorXd>>
  read_HWS_data(const std::filesystem::path& path);

} // namespace U3xU4