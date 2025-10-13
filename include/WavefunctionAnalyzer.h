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

namespace U3xU4
{
  struct hw_subspace
  {
    uint8_t hw;

    std::vector<u3::irrep> u3s;
    std::vector<u4::irrep> u4s;

    std::vector<std::vector<double>> probs;

    hw_subspace() = default;
    hw_subspace(int hw_, const std::vector<u3::irrep>& u3s_,
                const std::vector<u4::irrep>& u4s_,
                const std::vector<std::vector<double>>& probs_)
        : hw(qcx_utils::itou8(hw_)), u3s(u3s_), u4s(u4s_), probs(probs_)
    {}

    // apply permutation `perm` to vector `vec` 
    // (perm[i] = new position of old element i)
    template <typename T>
    static void apply_permutation_by_indices(std::vector<T>& vec, 
    							const std::vector<size_t>& indices)
    {
      std::vector<T> tmp;
      tmp.reserve(vec.size());
      for (size_t idx : indices)
        tmp.push_back(vec[idx]);     
      vec = std::move(tmp);
  	}

    // sort u3s in decreasing order of C2
    void sort_u3_by_C2_decreasing()
    {
      if (u3s.empty()) return;

      // build index array [0,1,2,...]
      std::vector<size_t> indices;
      for (size_t i = 0; i < u3s.size(); i++)
        indices.push_back(i);

      // sort indices by u3s[*].C2() decreasing
      std::stable_sort(indices.begin(), indices.end(),
        [&](size_t i, size_t j) { return u3s[i].C2() > u3s[j].C2(); });

      // apply permutation consistently
      apply_permutation_by_indices(u3s, indices);
      apply_permutation_by_indices(probs, indices);

      if (!u4s.empty())
        apply_permutation_by_indices(u4s, indices);
    }

    // sort u4s in increasing order of C2
    void sort_u4_by_C2_increasing()
    {
      if (u4s.empty()) return;

      // build index array [0,1,2,...]
      std::vector<size_t> indices;
      for (size_t i = 0; i < u4s.size(); i++)
        indices.push_back(i);

      // sort indices by u4s[*].C2() increasing
      std::stable_sort(indices.begin(), indices.end(),
        [&](size_t i, size_t j) { return u4s[i].C2() < u4s[j].C2(); });

      // apply permutation consistently
      apply_permutation_by_indices(u4s, indices);
      apply_permutation_by_indices(probs, indices);

      if (!u3s.empty())
        apply_permutation_by_indices(u3s, indices);
    }

    // print to terminal
    friend std::ostream& operator<<(std::ostream& os, const hw_subspace& s)
    {
      if (s.u3s.empty())
      {
        os << "subspace hw = " << static_cast<int>(s.hw) << "\n"
           << "u4::irrep,C2 ===> probabilities \n";
        for (size_t i = 0; i < s.u4s.size(); i++)
        {
          os << s.u4s[i] << s.u4s[i].C2() << " ===> ";
          for (const auto& p : s.probs[i])
            os << p << "%\t";
          os << "\n";
        }
      }
      else if (s.u4s.empty())
      {
        os << "subspace hw = " << static_cast<int>(s.hw) << "\n"
           << "u3::irrep,C2 ===> probabilities \n";
        for (size_t i = 0; i < s.u3s.size(); i++)
        {
          os << s.u3s[i] << s.u3s[i].C2() << " ===> ";
          for (const auto& p : s.probs[i])
            os << p << "%\t";
          os << "\n";
        }
      }
      else
      {
        os << "subspace hw = " << static_cast<int>(s.hw) << "\n"
           << "u3::irrep,C2 x u4::irrep,C2 ===> probabilities \n";
        for (size_t i = 0; i < s.u3s.size(); i++)
        {
          os << s.u3s[i] << s.u3s[i].C2() << " x " << s.u4s[i] << s.u4s[i].C2()
             << " ===> ";
          for (const auto& p : s.probs[i])
            os << p << "%\t";
          os << "\n";
        }
      }
      return os;
    }

    // print to file
    void print_to_file(std::ostream& os) const
    {
      os << static_cast<int>(hw) << "\n"
         << static_cast<int>(u3s.size()) << "\n"
         << static_cast<int>(u4s.size()) << "\n";

      if (u3s.empty())
      {
        for (size_t i = 0; i < u4s.size(); i++)
        {
          os << u4s[i].n14() << "\t" << u4s[i].n24() << "\t" << u4s[i].n34()
             << "\t" << u4s[i].n44() << "\t" << u4s[i].C2() << "\n";

          for (const auto& p : probs[i])
            os << p << "\t";
          os << "\n";
        }
      }
      else if (u4s.empty())
      {
        for (size_t i = 0; i < u3s.size(); i++)
        {
          os << u3s[i].n13() << "\t" << u3s[i].n23() << "\t" << u3s[i].n33()
             << "\t" << u3s[i].C2() << "\n";

          for (const auto& p : probs[i])
            os << p << "\t";
          os << "\n";
        }
      }
      else
      {
        for (size_t i = 0; i < u3s.size(); i++)
        {
          os << u3s[i].n13() << "\t" << u3s[i].n23() << "\t" << u3s[i].n33()
             << "\t" << u3s[i].C2() << "\n"
             << u4s[i].n14() << "\t" << u4s[i].n24() << "\t" << u4s[i].n34()
             << "\t" << u4s[i].n44() << "\t" << u4s[i].C2() << "\n";

          for (const auto& p : probs[i])
            os << p << "\t";
          os << "\n";
        }
      }
      return;
    }
  };
} // namespace U3xU4

namespace sam
{
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

  inline std::pair<std::vector<double>, std::vector<U3xU4::hw_subspace>>
  U3_wavefunction_analyzer(const U3xU4::ManyNucleonBasis& basis,
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
              if (u3 == subspace.u3s[id])
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
              subspace.probs.push_back(prob);
            }
          }
        }
      }

      // if this subspace hasn't been added
      if (!exist)
      {
        U3xU4::hw_subspace x(hw, {u3}, {}, {prob});
        hws.push_back(x);
      }

      offset = offset_next;
    }

    for (auto& hw : hws)
      hw.sort_u3_by_C2_decreasing();

    return std::make_pair(evals_extracted, hws);
  }

  inline std::pair<std::vector<double>, std::vector<U3xU4::hw_subspace>>
  U3_wavefunction_analyzer_write(const U3xU4::ManyNucleonBasis& basis,
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
              if (u3 == subspace.u3s[id])
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
              subspace.probs.push_back(prob);
            }
          }
        }
      }

      // if this subspace hasn't been added
      if (!exist)
      {
        U3xU4::hw_subspace x(hw, {u3}, {}, {prob});
        hws.push_back(x);
      }

      offset = offset_next;
    }

    std::string filename = matrix_file + "_U3_statistics.dat";
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

  inline std::pair<std::vector<double>, std::vector<U3xU4::hw_subspace>>
  U4_wavefunction_analyzer(const U3xU4::ManyNucleonBasis& basis,
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
            for (size_t id = 0; id < subspace.u4s.size(); id++)
            {
              if (u4 == subspace.u4s[id])
              {
                exist2 = true;
                for (size_t col = 0; col < num_states; col++)
                  subspace.probs[id][col] += prob[col];

                break;
              }
            }

            if (!exist2)
            {
              subspace.u4s.push_back(u4);
              subspace.probs.push_back(prob);
            }
          }
        }
      }

      // if this subspace hasn't been added
      if (!exist)
      {
        U3xU4::hw_subspace x(hw, {}, {u4}, {prob});
        hws.push_back(x);
      }

      offset = offset_next;
    }

    for (auto& hw : hws)
      hw.sort_u4_by_C2_increasing();

    return std::make_pair(evals_extracted, hws);
  }

  inline std::pair<std::vector<double>, std::vector<U3xU4::hw_subspace>>
  U4_wavefunction_analyzer_write(const U3xU4::ManyNucleonBasis& basis,
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
            for (size_t id = 0; id < subspace.u4s.size(); id++)
            {
              if (u4 == subspace.u4s[id])
              {
                exist2 = true;
                for (size_t col = 0; col < num_states; col++)
                  subspace.probs[id][col] += prob[col];

                break;
              }
            }

            if (!exist2)
            {
              subspace.u4s.push_back(u4);
              subspace.probs.push_back(prob);
            }
          }
        }
      }

      // if this subspace hasn't been added
      if (!exist)
      {
        U3xU4::hw_subspace x(hw, {}, {u4}, {prob});
        hws.push_back(x);
      }

      offset = offset_next;
    }

    std::string filename = matrix_file + "_U4_statistics.dat";
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "Cannot open file for writing.\n";
      return {};
    }
    file << num_states << "\n";
    for (auto& hw : hws)
    {
      hw.sort_u4_by_C2_increasing();
      hw.print_to_file(file);
    }

    return std::make_pair(evals_extracted, hws);
  }
} // namespace sam
