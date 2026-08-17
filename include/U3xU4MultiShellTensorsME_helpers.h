#pragma once

// Shared machinery for the MatrixElements_* builders (split across
// U3xU4MultiShellTensorsME.cpp [CSR] and ..._ME_Eigen.cpp [Eigen]):
// memory diagnostics, the block-local coupling-coefficient caches, the
// terminal-irrep reordering, and the assembly-timing macros.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <boost/container_hash/hash.hpp>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <U3xU4MultiShellTensors.h>

// Memory diagnostics
#define DIAGNOSE_MEMORY
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#include <sys/resource.h>

inline double rss_gb()
{
#if defined(__linux__)
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0)
      return std::stoul(line.substr(6)) / (1024.0 * 1024.0);
#elif defined(__APPLE__)
  mach_task_basic_info info;
  mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                (task_info_t)&info, &size) == KERN_SUCCESS)
    return info.resident_size / (1024.0 * 1024.0 * 1024.0);
#endif
  return 0.0;
}

// Peak (high-water) RSS since process start. The instantaneous rss_gb() can
// both miss transient peaks inside a phase and over-report freed-but-cached
// allocator pages; this is the number that decides OOM.
inline double rss_peak_gb()
{
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0)
#if defined(__APPLE__)
    return ru.ru_maxrss / (1024.0 * 1024.0 * 1024.0);  // bytes on macOS
#else
    return ru.ru_maxrss / (1024.0 * 1024.0);           // kilobytes on Linux
#endif
  return 0.0;
}

#ifdef USE_MPI
inline void print_rss(const std::string& label, int mpi_rank, int mpi_size)
{
  double rss = rss_gb();
  double peak = rss_peak_gb();
  double rss_min, rss_max, rss_sum, peak_max;
  MPI_Reduce(&rss, &rss_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&rss, &rss_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&rss, &rss_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&peak, &peak_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (mpi_rank == 0)
    std::cerr << "[MEM] " << label
              << "  min=" << rss_min
              << "  avg=" << rss_sum / mpi_size
              << "  max=" << rss_max
              << "  peak=" << peak_max << " GB" << std::endl;
}
#else
inline void print_rss(const std::string& label, int = 0, int = 1)
{
  std::cerr << "[MEM] " << label << "  rss=" << rss_gb()
            << "  peak=" << rss_peak_gb() << " GB" << std::endl;
}
#endif

// Attribute part of the RSS to the (per-rank) RME_db cache -- pass its live GB
// and (files, records) from RME_db::cache_bytes()/cache_stats(). With lazy
// loading this should read ~0 right after RMEdb construction and grow to its
// peak by the end of the OMP assembly block; the max across ranks is what a
// node must hold x ranks-per-node.
#ifdef USE_MPI
inline void print_rme_mem(const std::string& label, double rme_gb,
                          std::size_t files, std::size_t recs,
                          int mpi_rank, int /*mpi_size*/)
{
  double rme_max = rme_gb, rme_sum = rme_gb;
  MPI_Reduce(&rme_gb, &rme_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&rme_gb, &rme_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (mpi_rank == 0)
    std::cerr << "[MEM] " << label << "  RME cache: max=" << rme_max
              << " GB  total=" << rme_sum << " GB across ranks  (rank0 "
              << files << " files / " << recs << " recs)" << std::endl;
}
#else
inline void print_rme_mem(const std::string& label, double rme_gb,
                          std::size_t files, std::size_t recs, int = 0, int = 1)
{
  std::cerr << "[MEM] " << label << "  RME cache: " << rme_gb << " GB  ("
            << files << " files / " << recs << " recs)" << std::endl;
}
#endif

#define EXPLOIT_SYMMETRIC

  // Uncomment to print a coarse split of the matrix-element assembly time
  // ([assembly] compute+merge vs teardown+finalize) in MatrixElements_CSR,
  // plus a per-phase breakdown (unify / rme / coeff-lookups / accum).
// #define TIME_ASSEMBLY

#ifdef TIME_ASSEMBLY
  namespace {
    struct AsmProf { double unify = 0, rme = 0, work = 0; };
    AsmProf g_asm_prof[256];
    // RAII accumulator: adds elapsed time to `acc` on scope exit, so it also
    // captures the early `continue`s in the coefficient/label loops.
    struct AsmScope {
      double& acc;
      std::chrono::high_resolution_clock::time_point t0;
      explicit AsmScope(double& a)
        : acc(a), t0(std::chrono::high_resolution_clock::now()) {}
      ~AsmScope() {
        acc += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
      }
    };
  }
  #ifdef USE_OPENMP
    #define ASM_TID (omp_get_thread_num())
  #else
    #define ASM_TID 0
  #endif
  #define ASM_TIC(v) auto v = std::chrono::high_resolution_clock::now()
  #define ASM_TOC(field, v) g_asm_prof[ASM_TID].field +=                    \
      std::chrono::duration<double>(                                        \
          std::chrono::high_resolution_clock::now() - (v)).count()
#else
  #define ASM_TIC(v)
  #define ASM_TOC(field, v)
#endif

namespace U3xU4
{
  // --- Block-local coupling-coefficient reuse (basis/tensor reordering) ------
  // The compute loop iterates connected pairs SORTED by terminal (bra_irr,
  // ket_irr). Within such a block the coupling coefficients cgcU3/cgcU4 depend
  // only on (terminal irreps, tensor_irr, subgroup labels) -- all fixed by the
  // block -- so they are computed once (paying the irrep/physicalvector setup,
  // DB visit, and matrix copy a single time) and reused by reference across
  // every instance and label pair in the block. `g_asm_block` marks the current
  // block (set per iteration in the sorted loop); when it changes, each thread
  // drops the previous block's table. Keyed by the byte-sized labels, as in
  // u3lib/u4lib. Non-sorted callers leave g_asm_block untouched -> the table
  // simply persists (still correct).
  inline thread_local uint32_t g_asm_block = 0xFFFFFFFFu;

  template <class DB>
  const auto& cached_U3SO3(DB& db, int a, int b, int c, int d, int e,
                           int f, int g, int h, int i)
  {
    using Key = std::array<uint8_t, 9>;
    using Val = decltype(db.get_U3SO3(a, b, c, d, e, f, g, h, i));
    thread_local std::unordered_map<Key, Val, boost::hash<Key>> cache;
    thread_local uint32_t cur = 0xFFFFFFFFu;
    if (g_asm_block != cur) { cache.clear(); cur = g_asm_block; }

    Key k{ (uint8_t)a, (uint8_t)b, (uint8_t)c, (uint8_t)d, (uint8_t)e,
           (uint8_t)f, (uint8_t)g, (uint8_t)h, (uint8_t)i };
    auto it = cache.find(k);
    if (it != cache.end()) return it->second;
    return cache.emplace(k, db.get_U3SO3(a, b, c, d, e, f, g, h, i))
        .first->second;
  }

  template <class DB>
  const auto& cached_U4SU2SU2(DB& db, int a, int b, int c, int d, int e,
                              int f, int g, int h, int i, int j,
                              int k, int l, int m, int n, int o)
  {
    using Key = std::array<uint8_t, 15>;
    using Val = decltype(db.get_U4SU2SU2(a, b, c, d, e, f, g, h, i, j,
                                         k, l, m, n, o));
    thread_local std::unordered_map<Key, Val, boost::hash<Key>> cache;
    thread_local uint32_t cur = 0xFFFFFFFFu;
    if (g_asm_block != cur) { cache.clear(); cur = g_asm_block; }

    Key key{ (uint8_t)a, (uint8_t)b, (uint8_t)c, (uint8_t)d, (uint8_t)e,
             (uint8_t)f, (uint8_t)g, (uint8_t)h, (uint8_t)i, (uint8_t)j,
             (uint8_t)k, (uint8_t)l, (uint8_t)m, (uint8_t)n, (uint8_t)o };
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key,
        db.get_U4SU2SU2(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o))
        .first->second;
  }

  // Sort the work list into contiguous terminal-irrep blocks and order the
  // tensors within each connected set by terminal irrep (basis + tensor
  // reordering, cf. lsu3shell). Returns block_id per connected pair; the
  // compute loop publishes it to g_asm_block so the block-local coupling-
  // coefficient tables reuse within a block. Reorders connected_irreps and
  // connected_tensors in place; only affects locality, not results.
  template <class IrrepsDB, class TensorType>
  std::vector<uint32_t> reorder_into_blocks(
      std::vector<std::tuple<size_t, size_t, size_t>>& connected_irreps,
      std::vector<std::vector<size_t>>& connected_tensors,
      const IrrepsDB& bras_irreps,   // bra basis (== kets_irreps when square)
      const IrrepsDB& kets_irreps,   // ket basis
      const std::vector<TensorType>& tensors)
  {
    // terminal (lam, mu, P1, P2, P3) of each irrep, per basis
    auto make_term = [](const IrrepsDB& db)
    {
      std::vector<std::array<int16_t, 5>> term(db.size());
      for (size_t id = 0; id < term.size(); id++)
      {
        auto ir = db[id];
        const U3xU4::irrep& t = ir.omega.empty() ? ir.gamma[0].u3u4()
                                                 : ir.omega.back().u3u4();
        term[id] = { (int16_t)t.lam(), (int16_t)t.mu(),
                     (int16_t)t.P1(), (int16_t)t.P2(), (int16_t)t.P3() };
      }
      return term;
    };
    const auto term_bra = make_term(bras_irreps);
    const auto term_ket = make_term(kets_irreps);

    // tensor reordering (ShareGammas has no single terminal irrep -> skip)
    if constexpr (requires (const TensorType& t) { t.getLastIrrep(); })
    {
      for (auto& t2 : connected_tensors)
        std::sort(t2.begin(), t2.end(), [&](size_t a, size_t b)
        {
          const auto& ia = tensors[a].getLastIrrep();
          const auto& ib = tensors[b].getLastIrrep();
          return std::array<int, 5>{ia.lam(), ia.mu(), ia.P1(), ia.P2(), ia.P3()}
               < std::array<int, 5>{ib.lam(), ib.mu(), ib.P1(), ib.P2(), ib.P3()};
        });
    }

    // sort connected pairs by (bra_irr, ket_irr)
    std::sort(connected_irreps.begin(), connected_irreps.end(),
      [&](const auto& x, const auto& y)
      {
        const auto& bx = term_bra[std::get<0>(x)]; const auto& kx = term_ket[std::get<1>(x)];
        const auto& by = term_bra[std::get<0>(y)]; const auto& ky = term_ket[std::get<1>(y)];
        if (bx != by) return bx < by;
        return kx < ky;
      });

    std::vector<uint32_t> block_id_of_ci(connected_irreps.size());
    uint32_t blk = 0;
    for (size_t ci = 0; ci < connected_irreps.size(); ci++)
    {
      if (ci > 0)
      {
        const auto& a = connected_irreps[ci];
        const auto& p = connected_irreps[ci - 1];
        if (term_bra[std::get<0>(a)] != term_bra[std::get<0>(p)] ||
            term_ket[std::get<1>(a)] != term_ket[std::get<1>(p)])
          blk++;
      }
      block_id_of_ci[ci] = blk;
    }
    return block_id_of_ci;
  }

  // Helper function to map distribution to indices of gamma elements
  inline std::vector<int>
    map_by_positive_order(const std::span<const uint16_t>& v)
  {
    std::vector<int> res;
    // std::cout << "map_by_positive_order: " << v.size() << std::endl;
    res.reserve(v.size());

    int counter = 0;
    for (const auto& x : v)
    {
      if (x == 0) res.push_back(-1);
      else res.push_back(counter++);
    }
    return res;
  }
} // namespace U3xU4
