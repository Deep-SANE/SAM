#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <TwoBodyOperatorJT.h>

namespace sam
{
  struct strength_stats
  {
    double min;
    double max;
    double mean;
    double stdev;

    strength_stats() = default;
    strength_stats(double min_, double max_, double mean_, double stdev_)
      : min(min_), max(max_), mean(mean_), stdev(stdev_) {}

    friend std::ostream& operator<<(std::ostream& os, const strength_stats& s)
    {
      os << "min: " << s.min << ", "
         << "max: " << s.max << ", "
         << "mean: " << s.mean << ", "
         << "stdev: " << s.stdev;
      return os;
    }
  };

  inline strength_stats compute_stats(const std::vector<double>& vs)
  {
    if (vs.empty())
      return {};

    double sum = 0.0, sum_sq = 0.0, min = vs[0], max = vs[0];
    for (const auto& v : vs)
    {
      sum += v;
      if (min > v)
        min = v;
      if (max < v)
        max = v;
    }

    double mean = sum / vs.size();

    for (const auto& v : vs)
      sum_sq += (v - mean) * (v - mean);

    return {min, max, mean, std::sqrt(sum_sq / (vs.size() - 1.0))};
  }

  inline std::vector<std::pair<u3::irrep, strength_stats>>
  U3_tensor_analyzer_write(const std::string& name)
  {
    int hw = -1, valenceShell = 0, Nmax = 6, maxShell = 6;
    HO::Table_nljmjmt labels(maxShell);

    JT::TwoBodyOperator opt(name, labels, hw, Nmax, valenceShell, maxShell);

    // read from binary
    auto tensors = opt.readBinary();

    std::vector<std::vector<double>> strengths_U3;
    std::vector<u3::irrep> U3s;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      auto U3 = tensor.getLastIrrep().u3();

      auto it3 = std::find(U3s.begin(), U3s.end(), U3);
      if (it3 != U3s.end())
      {
        size_t index = std::distance(U3s.begin(), it3);
        strengths_U3[index].insert(strengths_U3[index].end(),
                                   tensor.coeffs.begin(), tensor.coeffs.end());
      }
      else
      {
        U3s.push_back(U3);
        strengths_U3.push_back(tensor.coeffs);
      }
    }

    std::vector<std::pair<u3::irrep, strength_stats>> tensors_stats;
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");

    std::string base_dir = std::string(sam_dir) + "/data/OperatorJT";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name + "_U3.dat";
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "Cannot open file for writing.\n";
      return {};
    }

    for (size_t id = 0; id < U3s.size(); id++)
    {
      auto stats = compute_stats(strengths_U3[id]);
      tensors_stats.emplace_back(U3s[id], stats);

      file << U3s[id].n13() << "\n"
           << U3s[id].n23() << "\n"
           << U3s[id].n33() << "\n"
           << stats.min << "\n"
           << stats.max << "\n"
           << stats.mean << "\n"
           << stats.stdev << "\n"
           << strengths_U3[id].size() << std::endl;

      for (const auto& c : strengths_U3[id])
        file << c << std::endl;
      file << "\n";
    }
    file.close();

    return tensors_stats;
  }

  inline std::vector<std::pair<u3::irrep, strength_stats>>
  U3_tensor_analyzer(const std::string& name)
  {
    int hw = -1, valenceShell = 0, Nmax = 6, maxShell = 6;
    HO::Table_nljmjmt labels(maxShell);

    JT::TwoBodyOperator opt(name, labels, hw, Nmax, valenceShell, maxShell);

    // read from binary
    auto tensors = opt.readBinary();

    std::vector<std::vector<double>> strengths_U3;
    std::vector<u3::irrep> U3s;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      auto U3 = tensor.getLastIrrep().u3();

      auto it3 = std::find(U3s.begin(), U3s.end(), U3);
      if (it3 != U3s.end())
      {
        size_t index = std::distance(U3s.begin(), it3);
        strengths_U3[index].insert(strengths_U3[index].end(),
                                   tensor.coeffs.begin(), tensor.coeffs.end());
      }
      else
      {
        U3s.push_back(U3);
        strengths_U3.push_back(tensor.coeffs);
      }
    }

    std::vector<std::pair<u3::irrep, strength_stats>> tensors_stats;

    for (size_t id = 0; id < U3s.size(); id++)
    {
      auto stats = compute_stats(strengths_U3[id]);
      tensors_stats.emplace_back(U3s[id], stats);
    }

    return tensors_stats;
  }

  inline std::vector<std::pair<u4::irrep, strength_stats>>
  U4_tensor_analyzer_write(const std::string& name)
  {
    int hw = -1, valenceShell = 0, Nmax = 6, maxShell = 6;
    HO::Table_nljmjmt labels(maxShell);

    JT::TwoBodyOperator opt(name, labels, hw, Nmax, valenceShell, maxShell);

    // read from binary
    auto tensors = opt.readBinary();

    std::vector<std::vector<double>> strengths_U4;
    std::vector<u4::irrep> U4s;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      auto U4 = tensor.getLastIrrep().u4();

      auto it4 = std::find(U4s.begin(), U4s.end(), U4);
      if (it4 != U4s.end())
      {
        size_t index = std::distance(U4s.begin(), it4);
        strengths_U4[index].insert(strengths_U4[index].end(),
                                   tensor.coeffs.begin(), tensor.coeffs.end());
      }
      else
      {
        U4s.push_back(U4);
        strengths_U4.push_back(tensor.coeffs);
      }
    }

    std::vector<std::pair<u4::irrep, strength_stats>> tensors_stats;
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");

    std::string base_dir = std::string(sam_dir) + "/data/OperatorJT";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name + "_U4.dat";
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "Cannot open file for writing.\n";
      return {};
    }

    for (size_t id = 0; id < U4s.size(); id++)
    {
      auto stats = compute_stats(strengths_U4[id]);
      tensors_stats.emplace_back(U4s[id], stats);

      file << U4s[id].n14() << "\n"
           << U4s[id].n24() << "\n"
           << U4s[id].n34() << "\n"
           << U4s[id].n44() << "\n"
           << stats.min << "\n"
           << stats.max << "\n"
           << stats.mean << "\n"
           << stats.stdev << "\n"
           << strengths_U4[id].size() << std::endl;

      for (const auto& c : strengths_U4[id])
        file << c << std::endl;
      file << "\n";
    }
    file.close();

    return tensors_stats;
  }

  inline std::vector<std::pair<u4::irrep, strength_stats>>
  U4_tensor_analyzer(const std::string& name)
  {
    int hw = -1, valenceShell = 0, Nmax = 6, maxShell = 6;
    HO::Table_nljmjmt labels(maxShell);

    JT::TwoBodyOperator opt(name, labels, hw, Nmax, valenceShell, maxShell);

    // read from binary
    auto tensors = opt.readBinary();

    std::vector<std::vector<double>> strengths_U4;
    std::vector<u4::irrep> U4s;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      auto U4 = tensor.getLastIrrep().u4();

      auto it4 = std::find(U4s.begin(), U4s.end(), U4);
      if (it4 != U4s.end())
      {
        size_t index = std::distance(U4s.begin(), it4);
        strengths_U4[index].insert(strengths_U4[index].end(),
                                   tensor.coeffs.begin(), tensor.coeffs.end());
      }
      else
      {
        U4s.push_back(U4);
        strengths_U4.push_back(tensor.coeffs);
      }
    }

    std::vector<std::pair<u4::irrep, strength_stats>> tensors_stats;

    for (size_t id = 0; id < U4s.size(); id++)
    {
      auto stats = compute_stats(strengths_U4[id]);
      tensors_stats.emplace_back(U4s[id], stats);
    }

    return tensors_stats;
  }

} // namespace sam
