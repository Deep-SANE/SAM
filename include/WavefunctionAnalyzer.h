#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <eigensolver.h>
#include <qcx_convert.h>

#include <U3irrep.h>

#include <U4irrep.h>

#include <U3xU4ManyNucleonBasis.h>

namespace sam
{
  inline std::vector<U3xU4::hw_subspace> U3_wavefunction_analyzer(
    const U3xU4::ManyNucleonBasis& basis, const Eigen::MatrixXd& evecs, 
    int num_states)
  {
    assert(num_states <= evecs.cols());

    auto irreps = basis.irreps();

    std::vector<U3xU4::hw_subspace> hws;

    int quanta_0hw = 0, offset = 0;

    for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
    {
      auto [gamma, omega] = irreps[irrep_id];
      auto offset_next = (irrep_id == irreps.size() - 1) 
                       ? basis.dim() : irreps.offset(irrep_id + 1);

      auto irr = (omega.empty()) ? gamma.back().u3() : omega.back().u3();
      irr.normalize();

      int quanta = 0;
      for (const auto& g : gamma)
      {
        if (irrep_id == 0) quanta_0hw += g.quanta();
        quanta += g.quanta();
      }
      int hw = quanta - quanta_0hw;

      std::vector<double> prob(num_states, 0.0);
      for (size_t col = 0; col < num_states; col++)
        for (size_t row = offset; row < offset_next; row++)
          prob[col] += (evecs(row, col) * evecs(row, col) * 100);

      bool exist = false;

      if (!hws.empty())
      {
        for (auto& subspace : hws)
        {
          if (subspace.hw == hw)
          {
            exist = true;

            bool exist2 = false;
            for (size_t id = 0; id < subspace.u3s.size(); id++)
            {
              if (irr == subspace.u3s[id])
              {
                exist2 = true;
                for (size_t col = 0; col < num_states; col++)
                  subspace.probs[id][col] += prob[col];

                break;
              }
            }

            if (!exist2)
            {
              subspace.u3s.push_back(irr);
              subspace.probs.push_back(prob);
            }
          }
        }
      }

      // if this subspace hasn't been added
      if (!exist)
      {
        U3xU4::hw_subspace x(hw, {irr}, {}, {}, {}, {prob});
        hws.push_back(std::move(x));
      }

      offset = offset_next;
    }

    for (auto& hw : hws) hw.sort_u3_by_C2_decreasing();

    return hws;
  }

  inline std::vector<U3xU4::hw_subspace> U4_wavefunction_analyzer(
    const U3xU4::ManyNucleonBasis& basis, const Eigen::MatrixXd& evecs, 
    int num_states)
  {
    assert(num_states <= evecs.cols());

    auto irreps = basis.irreps();

    std::vector<U3xU4::hw_subspace> hws;

    int quanta_0hw = 0, offset = 0;

    for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
    {
      auto [gamma, omega] = irreps[irrep_id];
      auto offset_next = (irrep_id == irreps.size() - 1) 
                       ? basis.dim() : irreps.offset(irrep_id + 1);

      auto irr = (omega.empty()) ? gamma.back().u4() : omega.back().u4();
      irr.normalize();

      int quanta = 0;
      for (const auto& g : gamma)
      {
        if (irrep_id == 0) quanta_0hw += g.quanta();
        quanta += g.quanta();
      }
      int hw = quanta - quanta_0hw;

      std::vector<double> prob(num_states, 0.0);
      for (size_t col = 0; col < num_states; col++)
        for (size_t row = offset; row < offset_next; row++)
          prob[col] += (evecs(row, col) * evecs(row, col) * 100);

      bool exist = false;

      if (!hws.empty())
      {
        for (auto& subspace : hws)
        {
          if (subspace.hw == hw)
          {
            exist = true;

            bool exist2 = false;
            for (size_t id = 0; id < subspace.u4s.size(); id++)
            {
              if (irr == subspace.u4s[id])
              {
                exist2 = true;
                for (size_t col = 0; col < num_states; col++)
                  subspace.probs[id][col] += prob[col];

                break;
              }
            }

            if (!exist2)
            {
              subspace.u4s.push_back(irr);
              subspace.probs.push_back(prob);
            }
          }
        }
      }

      // if this subspace hasn't been added
      if (!exist)
      {
        U3xU4::hw_subspace x(hw, {}, {irr}, {}, {}, {prob});
        hws.push_back(std::move(x));
      }

      offset = offset_next;
    }

    for (auto& hw : hws) hw.sort_u4_by_C2_increasing();

    return hws;
  }

} // namespace sam

/*
  inline std::pair<std::vector<double>, std::vector<U3xU4::hw_subspace>>
  U3xU4_wavefunction_analyzer(const U3xU4::ManyNucleonBasis& basis,
                              const std::string& matrix_file, int num_states)
  {
    std::vector<U3xU4::hw_subspace> hws;

    auto irreps = basis.irreps();
    auto labels = basis.decomposition();
    auto dim = basis.dim();

    auto H = qcx_utils::readBinary(matrix_file);
    assert(dim == H.cols());

    auto [evals, evecs] = qcx_utils::eigensolver_symmetric(H, H.rows());
    std::vector<double> evals_extracted(evals.begin(),
                                        evals.begin() + num_states);

    int quanta_0hw = 0, offset = 0;

    for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
    {
      auto [gamma, omega] = irreps[irrep_id];
      auto offset_next =
          (irrep_id == irreps.size() - 1) ? dim : irreps.offset(irrep_id + 1);

      auto last_irrep =
          (omega.empty()) ? gamma.back().u3u4() : omega.back().u3u4();
      u3::irrep u3(last_irrep.n13(), last_irrep.n23(), last_irrep.n33());
      u3.normalize();
      u4::irrep u4(last_irrep.n14(), last_irrep.n24(), last_irrep.n34(),
                   last_irrep.n44());
      u4.normalize();

      if (irrep_id == 0)
        for (const auto& g : gamma)
          quanta_0hw += g.quanta();

      int quanta = 0;
      for (const auto& g : gamma)
        quanta += g.quanta();
      int hw = quanta - quanta_0hw;

      std::vector<double> prob(num_states, 0.0);
      for (size_t col = 0; col < num_states; col++)
        for (size_t row = offset; row < offset_next; row++)
          prob[col] += (evecs(row, col) * evecs(row, col) * 100);

      bool exist = false;

      if (!hws.empty())
      {
        for (auto& subspace : hws)
        {
          if (subspace.hw == hw)
          {
            exist = true;

            bool exist2 = false;
            for (size_t id = 0; id < subspace.u3s.size(); id++)
            {
              if (u3 == subspace.u3s[id] && u4 == subspace.u4s[id])
              {
                exist2 = true;
                for (size_t col = 0; col < num_states; col++)
                  subspace.probs[id][col] += prob[col];

                break;
              }
            }

            if (!exist2)
            {
              subspace.u3s.push_back(u3);
              subspace.u4s.push_back(u4);
              subspace.probs.push_back(prob);
            }
          }
        }
      }

      // if this subspace hasn't been added
      if (!exist)
      {
        U3xU4::hw_subspace x(hw, {u3}, {u4}, {prob});
        hws.push_back(x);
      }

      offset = offset_next;
    }

    for (auto& hw : hws)
      hw.sort_u3_by_C2_decreasing();

    return std::make_pair(evals_extracted, hws);
  }

  inline std::pair<std::vector<double>, std::vector<U3xU4::hw_subspace>>
  U3xU4_wavefunction_analyzer_write(const U3xU4::ManyNucleonBasis& basis,
                              const std::string& matrix_file, int num_states)
  {
    std::vector<U3xU4::hw_subspace> hws;

    auto irreps = basis.irreps();
    auto labels = basis.decomposition();
    auto dim = basis.dim();

    auto H = qcx_utils::readBinary(matrix_file);
    assert(dim == H.cols());

    auto [evals, evecs] = qcx_utils::eigensolver_symmetric(H, H.rows());
    std::vector<double> evals_extracted(evals.begin(),
                                        evals.begin() + num_states);

    int quanta_0hw = 0, offset = 0;

    for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
    {
      auto [gamma, omega] = irreps[irrep_id];
      auto offset_next =
          (irrep_id == irreps.size() - 1) ? dim : irreps.offset(irrep_id + 1);

      auto last_irrep =
          (omega.empty()) ? gamma.back().u3u4() : omega.back().u3u4();
      u3::irrep u3(last_irrep.n13(), last_irrep.n23(), last_irrep.n33());
      u3.normalize();
      u4::irrep u4(last_irrep.n14(), last_irrep.n24(), last_irrep.n34(),
                   last_irrep.n44());
      u4.normalize();

      if (irrep_id == 0)
        for (const auto& g : gamma)
          quanta_0hw += g.quanta();

      int quanta = 0;
      for (const auto& g : gamma)
        quanta += g.quanta();
      int hw = quanta - quanta_0hw;

      std::vector<double> prob(num_states, 0.0);
      for (size_t col = 0; col < num_states; col++)
        for (size_t row = offset; row < offset_next; row++)
          prob[col] += (evecs(row, col) * evecs(row, col) * 100);

      bool exist = false;

      if (!hws.empty())
      {
        for (auto& subspace : hws)
        {
          if (subspace.hw == hw)
          {
            exist = true;

            bool exist2 = false;
            for (size_t id = 0; id < subspace.u3s.size(); id++)
            {
              if (u3 == subspace.u3s[id] && u4 == subspace.u4s[id])
              {
                exist2 = true;
                for (size_t col = 0; col < num_states; col++)
                  subspace.probs[id][col] += prob[col];

                break;
              }
            }

            if (!exist2)
            {
              subspace.u3s.push_back(u3);
              subspace.u4s.push_back(u4);
              subspace.probs.push_back(prob);
            }
          }
        }
      }

      // if this subspace hasn't been added
      if (!exist)
      {
        U3xU4::hw_subspace x(hw, {u3}, {u4}, {prob});
        hws.push_back(x);
      }

      offset = offset_next;
    }

    std::string filename = matrix_file + "_U3xU4_statistics.dat";
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "Cannot open file for writing.\n";
      return {};
    }
    file << num_states << "\n";
    for (auto& hw : hws)
    {
      hw.sort_u3_by_C2_decreasing();
      hw.print_to_file(file);
    }

    return std::make_pair(evals_extracted, hws);
  }
*/
