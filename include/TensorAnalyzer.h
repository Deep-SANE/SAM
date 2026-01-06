#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <U3xU4MultiShellTensors.h>

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
  U3_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors, 
                                      const std::string& filename)
  {
    std::vector<std::vector<double>> strengths_U3;
    std::vector<u3::irrep> U3s;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      auto U3 = tensor.getLastIrrep().u3();

      std::vector<double> coeffs;
      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
        for (const auto& s : strengths) coeffs.push_back(s);

      auto it = std::find(U3s.begin(), U3s.end(), U3);
      if (it != U3s.end())
      {
        size_t index = std::distance(U3s.begin(), it);
        strengths_U3[index].insert(strengths_U3[index].end(),
                                   coeffs.begin(), coeffs.end());
      }
      else
      {
        U3s.push_back(std::move(U3));
        strengths_U3.push_back(std::move(coeffs));
      }
    }
    
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "Cannot open file for writing.\n";
      return {};
    }

    std::vector<std::pair<u3::irrep, strength_stats>> tensors_stats;

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
  U3_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors)
  {
    std::vector<std::vector<double>> strengths_U3;
    std::vector<u3::irrep> U3s;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      auto U3 = tensor.getLastIrrep().u3();

      std::vector<double> coeffs;
      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
        for (const auto& s : strengths) coeffs.push_back(s);

      auto it = std::find(U3s.begin(), U3s.end(), U3);
      if (it != U3s.end())
      {
        size_t index = std::distance(U3s.begin(), it);
        strengths_U3[index].insert(strengths_U3[index].end(),
                                   coeffs.begin(), coeffs.end());
      }
      else
      {
        U3s.push_back(std::move(U3));
        strengths_U3.push_back(std::move(coeffs));
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
  U4_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors, 
                                      const std::string& filename)
  {
    std::vector<std::vector<double>> strengths_U4;
    std::vector<u4::irrep> U4s;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      auto U4 = tensor.getLastIrrep().u4();

      std::vector<double> coeffs;
      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
        for (const auto& s : strengths) coeffs.push_back(s);

      auto it = std::find(U4s.begin(), U4s.end(), U4);
      if (it != U4s.end())
      {
        size_t index = std::distance(U4s.begin(), it);
        strengths_U4[index].insert(strengths_U4[index].end(),
                                   coeffs.begin(), coeffs.end());
      }
      else
      {
        U4s.push_back(std::move(U4));
        strengths_U4.push_back(std::move(coeffs));
      }
    }
    
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "Cannot open file for writing.\n";
      return {};
    }

    std::vector<std::pair<u4::irrep, strength_stats>> tensors_stats;

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
  U4_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors)
  {
    std::vector<std::vector<double>> strengths_U4;
    std::vector<u4::irrep> U4s;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      auto U4 = tensor.getLastIrrep().u4();

      std::vector<double> coeffs;
      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
        for (const auto& s : strengths) coeffs.push_back(s);

      auto it = std::find(U4s.begin(), U4s.end(), U4);
      if (it != U4s.end())
      {
        size_t index = std::distance(U4s.begin(), it);
        strengths_U4[index].insert(strengths_U4[index].end(),
                                   coeffs.begin(), coeffs.end());
      }
      else
      {
        U4s.push_back(std::move(U4));
        strengths_U4.push_back(std::move(coeffs));
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

  inline std::vector<std::pair<int, strength_stats>> 
  LL_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors, 
                                      const std::string& filename)
  {
    std::vector<std::vector<double>> strengths_LL;
    std::vector<int> LLs;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
      {
        auto it = std::find(LLs.begin(), LLs.end(), LL);
        if (it != LLs.end())
        {
          size_t index = std::distance(LLs.begin(), it);
          strengths_LL[index].insert(strengths_LL[index].end(),
                                     strengths.begin(), strengths.end());
        }
        else
        {
          LLs.push_back(LL);
          strengths_LL.push_back(std::move(strengths));
        } 
      }
    }
    
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "Cannot open file for writing.\n";
      return {};
    }

    std::vector<std::pair<int, strength_stats>> tensors_stats;

    for (size_t id = 0; id < LLs.size(); id++)
    {
      auto stats = compute_stats(strengths_LL[id]);
      tensors_stats.emplace_back(LLs[id], stats);

      file << LLs[id] << "\n"
           << stats.min << "\n"
           << stats.max << "\n"
           << stats.mean << "\n"
           << stats.stdev << "\n"
           << strengths_LL[id].size() << std::endl;

      for (const auto& c : strengths_LL[id])
        file << c << std::endl;
      file << "\n";
    }
    file.close();

    return tensors_stats;
  }

  inline std::vector<std::pair<int, strength_stats>> 
  LL_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors)
  {
    std::vector<std::vector<double>> strengths_LL;
    std::vector<int> LLs;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
      {
        auto it = std::find(LLs.begin(), LLs.end(), LL);
        if (it != LLs.end())
        {
          size_t index = std::distance(LLs.begin(), it);
          strengths_LL[index].insert(strengths_LL[index].end(),
                                     strengths.begin(), strengths.end());
        }
        else
        {
          LLs.push_back(LL);
          strengths_LL.push_back(std::move(strengths));
        } 
      }
    }

    std::vector<std::pair<int, strength_stats>> tensors_stats;

    for (size_t id = 0; id < LLs.size(); id++)
    {
      auto stats = compute_stats(strengths_LL[id]);
      tensors_stats.emplace_back(LLs[id], stats);
    }

    return tensors_stats;
  }

  inline std::vector<std::pair<int, strength_stats>> 
  SS_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors, 
                                      const std::string& filename)
  {
    std::vector<std::vector<double>> strengths_SS;
    std::vector<int> SSs;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
      {
        auto it = std::find(SSs.begin(), SSs.end(), SS);
        if (it != SSs.end())
        {
          size_t index = std::distance(SSs.begin(), it);
          strengths_SS[index].insert(strengths_SS[index].end(),
                                     strengths.begin(), strengths.end());
        }
        else
        {
          SSs.push_back(SS);
          strengths_SS.push_back(std::move(strengths));
        } 
      }
    }
    
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "Cannot open file for writing.\n";
      return {};
    }

    std::vector<std::pair<int, strength_stats>> tensors_stats;

    for (size_t id = 0; id < SSs.size(); id++)
    {
      auto stats = compute_stats(strengths_SS[id]);
      tensors_stats.emplace_back(SSs[id], stats);

      file << SSs[id] << "\n"
           << stats.min << "\n"
           << stats.max << "\n"
           << stats.mean << "\n"
           << stats.stdev << "\n"
           << strengths_SS[id].size() << std::endl;

      for (const auto& c : strengths_SS[id])
        file << c << std::endl;
      file << "\n";
    }
    file.close();

    return tensors_stats;
  }

  inline std::vector<std::pair<int, strength_stats>> 
  SS_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors)
  {
    std::vector<std::vector<double>> strengths_SS;
    std::vector<int> SSs;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
      {
        auto it = std::find(SSs.begin(), SSs.end(), SS);
        if (it != SSs.end())
        {
          size_t index = std::distance(SSs.begin(), it);
          strengths_SS[index].insert(strengths_SS[index].end(),
                                     strengths.begin(), strengths.end());
        }
        else
        {
          SSs.push_back(SS);
          strengths_SS.push_back(std::move(strengths));
        } 
      }
    }

    std::vector<std::pair<int, strength_stats>> tensors_stats;

    for (size_t id = 0; id < SSs.size(); id++)
    {
      auto stats = compute_stats(strengths_SS[id]);
      tensors_stats.emplace_back(SSs[id], stats);
    }

    return tensors_stats;
  }

  inline std::vector<std::pair<int, strength_stats>> 
  TT_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors, 
                                      const std::string& filename)
  {
    std::vector<std::vector<double>> strengths_TT;
    std::vector<int> TTs;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
      {
        auto it = std::find(TTs.begin(), TTs.end(), TT);
        if (it != TTs.end())
        {
          size_t index = std::distance(TTs.begin(), it);
          strengths_TT[index].insert(strengths_TT[index].end(),
                                     strengths.begin(), strengths.end());
        }
        else
        {
          TTs.push_back(TT);
          strengths_TT.push_back(std::move(strengths));
        } 
      }
    }
    
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "Cannot open file for writing.\n";
      return {};
    }

    std::vector<std::pair<int, strength_stats>> tensors_stats;

    for (size_t id = 0; id < TTs.size(); id++)
    {
      auto stats = compute_stats(strengths_TT[id]);
      tensors_stats.emplace_back(TTs[id], stats);

      file << TTs[id] << "\n"
           << stats.min << "\n"
           << stats.max << "\n"
           << stats.mean << "\n"
           << stats.stdev << "\n"
           << strengths_TT[id].size() << std::endl;

      for (const auto& c : strengths_TT[id])
        file << c << std::endl;
      file << "\n";
    }
    file.close();

    return tensors_stats;
  }

  inline std::vector<std::pair<int, strength_stats>> 
  TT_tensor_analyzer(const U3xU4::MultiShellTensors_phys& tensors)
  {
    std::vector<std::vector<double>> strengths_TT;
    std::vector<int> TTs;

    for (const auto& tensor : tensors)
    {
      // std::cout << tensor << std::endl;

      for (const auto& [k, LL, z, SS, TT, strengths] : tensor.LLSSTTstrengths())
      {
        auto it = std::find(TTs.begin(), TTs.end(), TT);
        if (it != TTs.end())
        {
          size_t index = std::distance(TTs.begin(), it);
          strengths_TT[index].insert(strengths_TT[index].end(),
                                     strengths.begin(), strengths.end());
        }
        else
        {
          TTs.push_back(TT);
          strengths_TT.push_back(std::move(strengths));
        } 
      }
    }

    std::vector<std::pair<int, strength_stats>> tensors_stats;

    for (size_t id = 0; id < TTs.size(); id++)
    {
      auto stats = compute_stats(strengths_TT[id]);
      tensors_stats.emplace_back(TTs[id], stats);
    }

    return tensors_stats;
  }

} // namespace sam
