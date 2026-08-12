#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <omp.h>
#include <sys/resource.h>
#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim: return freed pages (m-scheme + decomp) to OS
#endif

#include <qcx_convert.h>
#include <qcx_print.h>
using qcx_utils::print;
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>
#include <so3Coeffs/SO3_Fano.h>

#include <sam.h>
#include <u3lib.h>   // u3::clear(): wipe the coupling-coefficient caches

#include <TwoBodyOperatorJT.h>

namespace JT
{
  // Opt-in per-thread m-scheme cache cap (entries), from SAM_MSCHEME_CACHE_CAP.
  // Unset/0 => uncapped (default, bit-for-bit the old behavior). Set it on large
  // cold expansions (SuperMike Nmax>=10) to bound per-thread memory; see
  // HO::maybe_release_twbme_cache.
  static std::size_t mscheme_cache_cap()
  {
    const char* e = std::getenv("SAM_MSCHEME_CACHE_CAP");
    return e ? std::strtoull(e, nullptr, 10) : 0;
  }

  // Return freed heap pages to the OS. maybe_release_twbme_cache frees the
  // per-thread m-scheme maps inside the parallel region but CANNOT trim there (a
  // global malloc_trim would serialize every thread on the malloc lock); and the
  // decomp cache's trim only fires when its cap is actually exceeded, which
  // rarely happens once SAM_DECOMP_NOCACHE_ABOVE keeps the cache small. So the
  // sawtooth of freed m-scheme buckets otherwise sits in glibc's arena and RSS
  // creeps. Call this ONCE per chunk at the omp single (all threads parked -> no
  // lock contention; ~once per flush_chunk tensors -> negligible cost).
  static inline void trim_heap()
  {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
  }

  // --- per-rank memory monitor helpers (SAM_EXPAND_MEMLOG) -----------------
  // Peak (high-water) RSS since process start -- the number that decides OOM.
  static double rss_peak_gb()
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
#if defined(__APPLE__)
      return ru.ru_maxrss / (1024.0 * 1024.0 * 1024.0);  // bytes on macOS
#else
      return ru.ru_maxrss / (1024.0 * 1024.0);           // KB on Linux
#endif
    return 0.0;
  }
  // Current resident set (Linux /proc; peak suffices elsewhere).
  static double rss_now_gb()
  {
#if defined(__linux__)
    long total = 0, resident = 0;
    if (FILE* f = std::fopen("/proc/self/statm", "r"))
    {
      if (std::fscanf(f, "%ld %ld", &total, &resident) != 2) resident = 0;
      std::fclose(f);
    }
    return resident * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0 * 1024.0);
#else
    return 0.0;
#endif
  }

#define RESTRICT_WITH_NMAX
#define RESTRICT_JJ // only used in expand_old()
  U3xU4::TwoBodyMultiShellTensors 
    TwoBodyOperator::expand_old()
  {
    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->read();
      else
      {
        assert(TBME_general);
        TBME_general->read();
        JJ_ = static_cast<HO::qn_t>(TBME_general->JJ());
        TT_ = static_cast<HO::qn_t>(TBME_general->TT());
        parity_ = static_cast<HO::qn_t>(TBME_general->parity());
      }
    }

    U3xU4::TwoBodyMultiShellTensors tensors;

#ifdef RESTRICT_WITH_NMAX
    auto [all_na_nb_nc_nd, all_inverse, all_phase] =
        GenerateAllCombinationsHOShells(Nmax_, valenceShell_, maxShell_);
#else
    auto all_na_nb_nc_nd = GenerateAllCombinationsHOShells(maxShell_);
#endif

    // Shared cache of single-shell tensor decompositions (thread-safe)
    U3xU4::SSTDecomp_phys_db decompdb(maxShell_);

    for (size_t i = 0; i < all_na_nb_nc_nd.size(); i++)
    {
/*      
      for (auto sorted_n : all_na_nb_nc_nd[i])
        std::cout << sorted_n << " ";
      std::cout << "\n phases = ";
      for (auto phase : all_phase[i])
        std::cout << phase << " ";
      std::cout << "\n";
*/      

      // auto groups_ns = group_same_shell(sorted_ns);
      auto groups_ns = group_same_shell(all_na_nb_nc_nd[i]);
      std::vector<U3xU4::SingleShellTensors> Gammas_lists;
      for (const auto& g : groups_ns)
      {
        auto Gammas = U3xU4::GenerateSingleShellTensors(
            std::get<0>(g), std::get<1>(g), std::get<2>(g));
        Gammas_lists.push_back(std::move(Gammas));
      }

      std::vector<U3xU4::CouplingPath> all_results;
      U3xU4::GenerateMultiShellTensors(
        Gammas_lists, 0, {}, {}, {}, {}, all_results);

      for (const auto& [Gammas, Omegas] : all_results)
      {
        
        if (Gammas.size() > 1)
        {
          for (const auto& Gamma : Gammas)
          {
            // std::cout << Gamma << "\t";

            // Assert that if there are more than 1 single-shell tensor,
            // they must be all free of multiplicity.
            assert(Gamma.rho0max() == 1 && Gamma.eta0max() == 1);
          }
          //std::cout << std::endl;
        }

        U3xU4::irreps coupled_irreps;
        int Rhomax, Etamax;
        for (const auto& Omega : Omegas)
        {
          //U3xU4::irrep irr(Omega.u3(), Omega.u4());
          auto irr = Omega.u3u4();
          auto rhomax = Omega.rhomax(); 
          auto etamax = Omega.etamax();
          
          // std::cout << Omega << "\t";
          
          coupled_irreps.push_back(irr);
          Rhomax = rhomax;
          Etamax = etamax;
        }
        // std::cout << std::endl;

        // decompose into L
        u3::physicalvectors pv3s;
        // decompose into 2S and 2T
        u4::physicalvectors pv4s;

        if (coupled_irreps.empty()) // Gammas has one element only
        {
          pv3s = u3::physicalBasis(Gammas[0].u3());
          pv4s = u4::physicalBasis(Gammas[0].u4());
        }
        else
        {
          pv3s = u3::physicalBasis(coupled_irreps.back().u3());
          pv4s = u4::physicalBasis(coupled_irreps.back().u4());
        }

        for (const auto& pv3 : pv3s)
        {
          int kappa0 = pv3.kappa(), LL0 = 2 * pv3.L();

          for (const auto& pv4 : pv4s)
          {
            int zeta0 = pv4.zeta(), SS0 = pv4.SS(), TT0 = pv4.TT();

            // if the operator has definite isospin assignment
            if (TT() >= 0 && TT() != TT0) continue;

#ifdef RESTRICT_JJ
            // couple LL and SS to JJ
            if (JJ() < std::fabs(LL0 - SS0) || JJ() > (LL0 + SS0))
              continue;

            U3xU4::TwoBodyMultiShellTensor tensor(
              Gammas, coupled_irreps, 
              coupled_irreps.empty() ? Gammas[0].rho0max() : Rhomax,
              coupled_irreps.empty() ? Gammas[0].eta0max() : Etamax, 
              kappa0, LL0, zeta0, SS0, TT0, MMT_, JJ_, MMJ_, {}, HO_labels_);

            bool ignore =
                setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);

            //std::cout << (ignore ? "Ignore" : "Accept") << tensor << std::endl;

            if (ignore == false)
            {
              // std::cout << tensor << std::endl;
              tensors.push_back(std::move(tensor));
            }
#else
            for (int JJ0 = abs(LL0 - SS0); JJ0 <= LL0 + SS0; JJ0 += 2)
            {
              U3xU4::TwoBodyMultiShellTensor tensor(
                Gammas, coupled_irreps, 
                coupled_irreps.empty() ? Gammas[0].rho0max() : Rhomax,
                coupled_irreps.empty() ? Gammas[0].eta0max() : Etamax, 
                kappa0, LL0, zeta0, SS0, TT0, MMT_, JJ0, MMJ_, {}, HO_labels_);

              bool ignore =
                setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);

              //std::cout << (ignore ? "Ignore" : "Accept") << tensor << std::endl;

              if (ignore == false)
              {
                // std::cout << tensor << std::endl;
                tensors.push_back(std::move(tensor));
              }
            }
#endif
          }
        }
      }
    }

    // Finalize
    decompdb.finalize();
    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->clear();
      else TBME_general->clear();
    }


    return tensors;
  }

  U3xU4::MultiShellTensors_phys
    TwoBodyOperator::expand_new()
  {
    if (valenceShell_ > 0) throw std::runtime_error("Set valenceShell to 0 for reusability");

    // if (Nmax_ != maxShell_) throw std::runtime_error("Set Nmax and maxShell equal");

    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->read();
      else
      {
        assert(TBME_general);
        TBME_general->read();
        JJ_ = static_cast<HO::qn_t>(TBME_general->JJ());
        TT_ = static_cast<HO::qn_t>(TBME_general->TT());
        parity_ = static_cast<HO::qn_t>(TBME_general->parity());
      }
    }

#ifdef RESTRICT_WITH_NMAX
    auto [all_na_nb_nc_nd, all_inverse, all_phase] =
        GenerateAllCombinationsHOShells(Nmax_, valenceShell_, maxShell_);
#else
    auto [all_na_nb_nc_nd, all_inverse, all_phase] = 
        GenerateAllCombinationsHOShells(maxShell_);
#endif

// std::cerr << all_na_nb_nc_nd.size() << std::endl;

    // (I) Generate the entire list of tensors

    // U3xU4::MultiShellTensors_phys tensors;
    std::vector<std::pair<size_t, U3xU4::MultiShellTensor_phys>> tensors;

    for (size_t i = 0; i < all_na_nb_nc_nd.size(); i++)
    {
      // auto groups_ns = group_same_shell(sorted_ns);
      auto groups_ns = group_same_shell(all_na_nb_nc_nd[i]);
      std::vector<U3xU4::SingleShellTensors> Gammas_lists;
      for (const auto& g : groups_ns)
      {
        auto Gammas = U3xU4::GenerateSingleShellTensors(
            std::get<0>(g), std::get<1>(g), std::get<2>(g));
        Gammas_lists.push_back(std::move(Gammas));
      }

      std::vector<U3xU4::CouplingPath> all_results;
      U3xU4::GenerateMultiShellTensors(
        Gammas_lists, 0, {}, {}, {}, {}, all_results);

// std::cerr << i << "/" << all_na_nb_nc_nd.size() << " : " << all_results.size() << std::endl;

      // for (const auto& [Gammas, Omegas] : all_results)
      for (size_t j = 0; j < all_results.size(); j++)
      {
        const auto& Gammas = all_results[j].first;
        const auto& Omegas = all_results[j].second;

        U3xU4::MultiShellTensor_phys tensor(Gammas, Omegas, 
                                            MMT_, JJ_, MMJ_, HO_labels_);

        // selection rules
        if (U3selection(tensor.getLastIrrep().u3()) == false) continue;
        if (U4selection(tensor.getLastIrrep().u4()) == false) continue;
        if (tensor.shift_quanta() < 0 and symmetric_) continue;

        // setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);

        // if (!tensor.LLSSTTstrengths().empty())
          tensors.emplace_back(i, std::move(tensor));
      }
    }

    // (II) Run tensor expansion via OpenMP
    std::cerr << "Processing " << tensors.size() << " tensors" << std::endl;

    // Shared cache of single-shell tensor decompositions (thread-safe)
    U3xU4::SSTDecomp_phys_db decompdb(maxShell_);

    const std::size_t cache_cap = mscheme_cache_cap();

#pragma omp parallel
{
  sam::initialize_thread();

  // U3xU4::MultiShellTensors_phys tensors_tl;

#pragma omp for schedule(dynamic,1)
    for (size_t k = 0; k < tensors.size(); ++k)
    {
      auto& [i, tensor] = tensors[k];
      setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);
      HO::maybe_release_twbme_cache(cache_cap);

      if (k % 1000 == 0)
      {
    #pragma omp critical
        std::cerr << "Thread " << omp_get_thread_num()
                  << ": " << k << "/" << tensors.size() << std::endl;
      }
    }
/*
#pragma omp critical
  {
    tensors.reserve(tensors.size() + tensors_tl.size());

    for (auto& t : tensors_tl)
      tensors.push_back(std::move(t));
  }
*/
  sam::finalize_thread();
}

    // (III) Finalize

    // (a) release the TBME hash table
    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->clear();
      else TBME_general->clear();
    }

    // (b) release shell-combo arrays
    { decltype(all_na_nb_nc_nd)().swap(all_na_nb_nc_nd); }
    { decltype(all_inverse)().swap(all_inverse); }
    { decltype(all_phase)().swap(all_phase); }

    // (c) release u3/u4 databases
    sam::clear();

    // (d) flush newly computed decompositions to disk
    decompdb.finalize();

    // diagnostic: getStrengths me-phase vs accum-phase + getME counts
    // (no-op unless TIME_GETSTRENGTHS is defined)
    U3xU4::report_getStrengths_timing();
    // diagnostic: m-scheme cache hit/miss (no-op unless COUNT_MSCHEME_CACHE)
    HO::report_mscheme_cache_stats();
    // diagnostic: intra-tensor quad reuse (no-op unless MEASURE_MSCHEME_REUSE)
    HO::report_mscheme_reuse();

    // (e) compact tensors_rank: keep only entries owned by this rank
    U3xU4::MultiShellTensors_phys tensors_final;
    tensors_final.reserve(tensors.size());
    for (auto& [i, t] : tensors)
    {
      if (!t.LLSSTTstrengths().empty()) tensors_final.push_back(std::move(t));
      // else std::cout << "Ignore: " << t << std::endl;
    }

    return tensors_final;
  }

  U3xU4::MultiShellTensors_phys 
    TwoBodyOperator::expand_new_MPI(const std::string& filename)
  {
    if (valenceShell_ > 0) throw std::runtime_error("Set valenceShell to 0 for reusability");

    // if (Nmax_ != maxShell_) throw std::runtime_error("Set Nmax and maxShell equal");

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->read();
      else
      {
        assert(TBME_general);
        TBME_general->read();
        JJ_ = static_cast<HO::qn_t>(TBME_general->JJ());
        TT_ = static_cast<HO::qn_t>(TBME_general->TT());
        parity_ = static_cast<HO::qn_t>(TBME_general->parity());
      }
    }

#ifdef RESTRICT_WITH_NMAX
    auto [all_na_nb_nc_nd, all_inverse, all_phase] =
        GenerateAllCombinationsHOShells(Nmax_, valenceShell_, maxShell_);
#else
    auto [all_na_nb_nc_nd, all_inverse, all_phase] = 
        GenerateAllCombinationsHOShells(maxShell_);
#endif

// std::cerr << all_na_nb_nc_nd.size() << std::endl;

    // (I) Generate the entire list of tensors --- identical for all ranks

    // Under the default "strided" partition, apply ownership at BUILD time so
    // each rank stores only its share of tensor STRUCTURES. The full 398788-
    // tensor list is ~38 GB/rank at Nmax=12, and replicating it on every rank
    // (it never distributes) is what OOM'd the multi-node run. g_tensor is the
    // global tensor index -- the enumeration is identical on all ranks, so the
    // strided split is consistent. "weighted" still needs the whole list, so
    // under it every rank stores all.
    const bool weighted_partition =
      [] { const char* m = std::getenv("SAM_EXPAND_PARTITION");
           return m != nullptr && std::string(m) == "weighted"; }();

    std::vector<std::pair<size_t, U3xU4::MultiShellTensor_phys>> tensors_rank;
    size_t g_tensor = 0; // global index over all selection-passing tensors

    for (size_t i = 0; i < all_na_nb_nc_nd.size(); i++)
    {
      // auto groups_ns = group_same_shell(sorted_ns);
      auto groups_ns = group_same_shell(all_na_nb_nc_nd[i]);
      std::vector<U3xU4::SingleShellTensors> Gammas_lists;
      for (const auto& g : groups_ns)
      {
        auto Gammas = U3xU4::GenerateSingleShellTensors(
            std::get<0>(g), std::get<1>(g), std::get<2>(g));
        Gammas_lists.push_back(std::move(Gammas));
      }

      std::vector<U3xU4::CouplingPath> all_results;
      U3xU4::GenerateMultiShellTensors(
        Gammas_lists, 0, {}, {}, {}, {}, all_results);

// std::cerr << i << "/" << all_na_nb_nc_nd.size() << " : " << all_results.size() << std::endl;

      // for (const auto& [Gammas, Omegas] : all_results)
      for (size_t j = 0; j < all_results.size(); j++)
      {
        const auto& Gammas = all_results[j].first;
        const auto& Omegas = all_results[j].second;

        U3xU4::MultiShellTensor_phys tensor(Gammas, Omegas, 
                                            MMT_, JJ_, MMJ_, HO_labels_);

        // selection rules
        if (U3selection(tensor.getLastIrrep().u3()) == false) continue;
        if (U4selection(tensor.getLastIrrep().u4()) == false) continue;
        if (tensor.shift_quanta() < 0 and symmetric_) continue;

        // Ownership at BUILD time: strided stores only this rank's share so the
        // full structure list is never materialized; weighted stores all.
        const bool owned = weighted_partition ||
                           (g_tensor % (size_t)mpi_size == (size_t)mpi_rank);
        ++g_tensor;
        if (owned) tensors_rank.emplace_back(i, std::move(tensor));
      }
    }

    // (II) Assign tensors to this rank, then expand via OpenMP over the
    //      assignment. Strategy chosen by env SAM_EXPAND_PARTITION:
    //
    //   "strided" (DEFAULT, known-good): round-robin it = mpi_rank, += mpi_size.
    //     Balances tensor COUNT and finely interleaves cheap/expensive combos so
    //     every rank gets an even cost mix (~99% CPU historically). Downside: it
    //     scatters shell-combos across ranks, so per-rank m-scheme caches bloat
    //     toward the full table (~2x memory skew) -- fine on a big-RAM node with
    //     --mem=0. Reproduces the original (rank-interleaved) output order.
    //
    //   "weighted": equal-weight CONTIGUOUS blocks via a shell-degeneracy proxy
    //     Q = (na+1)(na+2)...(nd+1)(nd+2), charged once per combo. Localizes each
    //     combo's cache to one rank and yields canonical (global-index) output.
    //     BUT the proxy mispredicts COMPUTE for expand_exist -- high-shell combos
    //     are cheap (many filter to empty strengths) -- so blocks end up badly
    //     compute-imbalanced (one rank drains early and idles). DO NOT use until
    //     the weight reflects surviving-strength work, not shell size. Kept for
    //     that future fix.
    //
    // (A single rank x N threads is a third option with perfect balance, but on
    //  a multi-socket node the one address space serializes page faults from N
    //  threads growing uncapped caches and loses NUMA locality -> pathologically
    //  slow. Prefer several ranks: separate address spaces, NUMA-local.)
    std::vector<size_t> my_indices;
    if (weighted_partition)
    {
      // weighted stored the FULL list above; cut it into equal-weight
      // contiguous blocks (broken cost proxy -- see note above; kept only for
      // the future weight fix).
      std::vector<size_t> block_begin(mpi_size + 1, tensors_rank.size());
      block_begin[0] = 0;

      // shell-degeneracy proxy: #m-scheme s.p. states in HO shell n
      auto shell_deg = [](int n) -> double { return double((n + 1) * (n + 2)); };

      std::unordered_map<size_t, size_t> combo_count;
      for (const auto& [i, t] : tensors_rank) combo_count[i]++;

      std::vector<double> w(tensors_rank.size(), 0.0);
      double total = 0.0;
      for (size_t it = 0; it < tensors_rank.size(); ++it)
      {
        const auto& s = all_na_nb_nc_nd[tensors_rank[it].first];
        double Q = shell_deg(s[0]) * shell_deg(s[1])
                 * shell_deg(s[2]) * shell_deg(s[3]);
        w[it] = Q / double(combo_count[tensors_rank[it].first]);
        total += w[it];
      }

      const double target = (mpi_size > 0) ? total / double(mpi_size) : total;
      double acc = 0.0;
      int r = 1;
      for (size_t it = 0; it < tensors_rank.size() && r < mpi_size; )
      {
        size_t i = tensors_rank[it].first;
        while (it < tensors_rank.size() && tensors_rank[it].first == i)
          { acc += w[it]; ++it; }
        if (acc >= target * r) block_begin[r++] = it;
      }
      for (; r <= mpi_size; ++r) block_begin[r] = tensors_rank.size();

      for (size_t it = block_begin[mpi_rank]; it < block_begin[mpi_rank + 1]; ++it)
        my_indices.push_back(it);
    }
    else // strided: ownership already applied at build time -> own all stored
    {
      my_indices.resize(tensors_rank.size());
      for (size_t it = 0; it < tensors_rank.size(); ++it) my_indices[it] = it;
    }

    // Memory-cost proxy for one tensor (shell-degeneracy product). Used both to
    // front-load heavy work (SAM_EXPAND_HEAVY_FIRST) and to gate concurrent heavy
    // computes (SAM_EXPAND_HEAVY_SEM). Imperfect -- it mis-ranks the true decomp
    // size in detail (the real giants can sit mid-run, not first) -- but monotone
    // enough to separate the memory-heavy tail from the light bulk.
    auto heaviness = [&](size_t it) {
      const auto& s = all_na_nb_nc_nd[tensors_rank[it].first];
      auto d = [](int n) { return double((n + 1) * (n + 2)); };
      return d(s[0]) * d(s[1]) * d(s[2]) * d(s[3]);
    };

    // Fail-fast tuning (SAM_EXPAND_HEAVY_FIRST): reorder this rank's work so the
    // memory-heaviest tensors are processed FIRST, so a config that will OOM
    // fails cheaply instead of hours in. Sorts the INDEX list (no tensor moves);
    // changes output order -> re-validate, physics unchanged.
    if (std::getenv("SAM_EXPAND_HEAVY_FIRST"))
      std::stable_sort(my_indices.begin(), my_indices.end(),
        [&](size_t a, size_t b) { return heaviness(a) > heaviness(b); });

    const size_t my_count = my_indices.size();
    std::cerr << "Rank " << mpi_rank << ": processing " << my_count
              << " / " << g_tensor << " tensors" << std::endl;

    // Shared cache of single-shell tensor decompositions (thread-safe)
    U3xU4::SSTDecomp_phys_db decompdb(maxShell_);

    const std::size_t cache_cap = mscheme_cache_cap();

    // (III) Stream the expansion: compute this rank's tensors in chunks, and
    //       after each chunk flush the just-computed tensors to the rank file and
    //       free them. Peak RAM per rank = caches + one chunk -- independent of
    //       total strength size, the per-tensor size imbalance, node count, and
    //       Nmax, instead of holding ALL owned strengths until the end. Output is
    //       byte-identical to the non-streaming path (same tensors, same order):
    //       a placeholder count up front, patched via seekp at the end, exactly
    //       like the rank-0 merge below.
    const size_t flush_chunk = [] {
      const char* e = std::getenv("SAM_EXPAND_FLUSH_CHUNK");
      size_t c = e ? std::strtoull(e, nullptr, 10) : 0;
      return c ? c : 2000;
    }();

    // opt-in per-rank memory monitor (SAM_EXPAND_MEMLOG=1): prints RSS + cache
    // attribution at each chunk boundary, non-collective (no cross-rank sync).
    const bool memlog = std::getenv("SAM_EXPAND_MEMLOG") != nullptr;
    size_t ms_entries_acc = 0; // shared: per-chunk sum of thread m-scheme sizes

    // Coeff-cache probe/fix: clear the u3lib coupling caches every N chunks to
    // measure their footprint (RSS freed) and bound their growth. 0/unset = off.
    const size_t coeff_clear_n = [] {
      const char* e = std::getenv("SAM_EXPAND_COEFF_CLEAR");
      return e ? std::strtoull(e, nullptr, 10) : (size_t)0;
    }();

    std::string rank_filename = filename + ".rank_" + std::to_string(mpi_rank);
    std::cerr << "[expand_new] Rank " << mpi_rank << ": opening rank file "
              << rank_filename << std::endl;
    std::ofstream rf(rank_filename, std::ios::binary);
    if (!rf) throw std::runtime_error("Failed to open rank file: " + rank_filename);
    const auto count_pos = rf.tellp();     // rank file starts with the count
    size_t n_written = 0;
    qcx_utils::writeBinary(rf, n_written); // placeholder, patched below

#pragma omp parallel
{
  sam::initialize_thread();

    for (size_t base = 0; base < my_indices.size(); base += flush_chunk)
    {
      const size_t stop = std::min(base + flush_chunk, my_indices.size());

      // compute this chunk in parallel
#pragma omp for schedule(dynamic,1)
      for (size_t k = base; k < stop; ++k)
      {
        const size_t it = my_indices[k];
        auto& [i, tensor] = tensors_rank[it];

        setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);
        HO::maybe_release_twbme_cache(cache_cap);

        if (k % 1000 == 0)
        {
#pragma omp critical
          std::cerr << "Rank " << mpi_rank << " [thread " << omp_get_thread_num()
                    << "]: " << k << " / " << my_count << std::endl;
        }
      } // implicit barrier: chunk fully computed before flush

      // memory monitor: each thread adds its m-scheme cache size before the
      // single thread reads the total (memlog is uniform across threads, so the
      // barrier is reached by all).
      if (memlog)
      {
#pragma omp atomic
        ms_entries_acc += HO::twbme_mscheme_cache().size();
#pragma omp barrier
      }

      // one thread streams this chunk to disk and frees it; the rest wait
#pragma omp single
      {
        for (size_t k = base; k < stop; ++k)
        {
          auto tensor = std::move(tensors_rank[my_indices[k]].second);
          if (!tensor.LLSSTTstrengths().empty())
          {
            tensor.writeBinary(rf);
            ++n_written;
          }
          // tensor destructs here: chunk's strengths freed before the next chunk
        }

        // bound the decomp cache: evict LRU files over SAM_DECOMP_CACHE_CAP.
        // Safe here -- all threads are parked at the barrier, so no borrowed
        // ExpansionRef is live.
        decompdb.evict_to_cap();

        // Return this chunk's freed pages (the m-scheme maps released during the
        // parallel for, plus anything evict_to_cap dropped) to the OS. Threads
        // are parked here, so the malloc lock is uncontended.
        trim_heap();

        // Coeff-cache probe/fix (SAM_EXPAND_COEFF_CLEAR=N): every N chunks wipe
        // the u3lib coupling-coefficient caches (uncapped, shared per rank, never
        // released otherwise) and record the RSS they freed -- a direct footprint
        // measurement and a periodic-clear knob. Safe here (all threads parked).
        // -1 = no clear this chunk.
        double coeff_freed_gb = -1.0;
        if (coeff_clear_n && ((base / flush_chunk) % coeff_clear_n == 0))
        {
          const double r0 = rss_now_gb();
          u3::clear();
          trim_heap();
          coeff_freed_gb = r0 - rss_now_gb();
        }

        if (memlog)
        {
          auto [dfiles, drecs] = decompdb.cache_stats();
          std::cerr << "[MEM] Rank " << mpi_rank << " @ " << stop << " / "
                    << my_indices.size() << ": rss=" << rss_now_gb()
                    << " peak=" << rss_peak_gb() << " GB  mscheme="
                    << ms_entries_acc << " entries  decomp=" << dfiles
                    << " files / " << drecs << " recs / "
                    << decompdb.cache_bytes() / 1e9 << " GB  coeff_freed="
                    << coeff_freed_gb << " GB" << std::endl;
          ms_entries_acc = 0;
        }
      } // implicit barrier: flush done before the next chunk computes
    }

  sam::finalize_thread();
}

    // patch the real count into the placeholder slot at the head of the file
    if (!rf) throw std::runtime_error("Write failed on rank file: " + rank_filename);
    rf.seekp(count_pos);
    qcx_utils::writeBinary(rf, n_written);
    rf.close();
    std::cerr << "[expand_new] Rank " << mpi_rank << ": rank file closed OK ("
              << n_written << " tensors)" << std::endl;

    // tensors_rank now holds only moved-from shells
    tensors_rank.clear();
    tensors_rank.shrink_to_fit();

    // (IV) Free everything no longer needed. This runs AFTER the streaming loop
    //      because the compute used TBME / all_inverse / all_phase / decompdb.
    // (a) release the TBME hash table
    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->clear();
      else TBME_general->clear();
    }
    // (b) release shell-combo arrays
    { decltype(all_na_nb_nc_nd)().swap(all_na_nb_nc_nd); }
    { decltype(all_inverse)().swap(all_inverse); }
    { decltype(all_phase)().swap(all_phase); }
    // (c) release u3/u4 databases
    sam::clear();
    // (d) flush newly computed decompositions to disk (MPI turn-based)
    decompdb.finalize();

    std::cerr << "[expand_new] Rank " << mpi_rank << ": tensors freed, entering barrier" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD); // wait for all rank files to be written
    std::cerr << "[expand_new] Rank " << mpi_rank << ": past write barrier" << std::endl;

    // (V) Rank 0 does a streaming merge: raw byte copy, O(1) tensor memory.
    if (mpi_rank == 0)
    {
      // Write header with placeholder count, stream rank files, then seek back to update count.
      // Single pass — no pre-counting loop needed.
      std::ofstream out(filename, std::ios::binary);
      if (!out) throw std::runtime_error("Failed to open output file: " + filename);
      qcx_utils::writeBinary(out, maxShell_);
      auto count_pos = out.tellp();          // save position of n_tensors field
      size_t placeholder = 0;
      qcx_utils::writeBinary(out, placeholder);
      std::cerr << "[expand_new] Rank 0: header written, streaming rank files" << std::endl;

      size_t total_tensors = 0;
      for (int r = 0; r < mpi_size; ++r)
      {
        std::string rf_name = filename + ".rank_" + std::to_string(r);
        std::cerr << "[expand_new] Rank 0: streaming rank file " << r << std::endl;
        std::ifstream rf(rf_name, std::ios::binary);
        size_t n; qcx_utils::readBinary(rf, n); // read per-rank count
        std::cerr << "[expand_new] Rank 0: rank " << r << " has " << n << " tensors" << std::endl;
        total_tensors += n;
        if (n > 0)  // rdbuf() sets failbit if 0 bytes extracted — skip for empty ranks
        {
          out << rf.rdbuf();                     // raw byte copy — no deserialization
          if (!out) throw std::runtime_error("Write failed after streaming rank file " + rf_name);
        }
        rf.close();
        std::remove(rf_name.c_str());
        std::cerr << "[expand_new] Rank 0: rank file " << r << " streamed and removed" << std::endl;
      }

      // Seek back and overwrite the placeholder with the real count
      out.seekp(count_pos);
      qcx_utils::writeBinary(out, total_tensors);
      out.close();
      std::cerr << "[expand_new] Rank 0: merged " << total_tensors
                << " tensors into " << filename << std::endl;
    }

    std::cerr << "[expand_new] Rank " << mpi_rank << ": entering final barrier" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD); // wait for merge before all ranks do final read
    std::cerr << "[expand_new] Rank " << mpi_rank << ": past final barrier, starting final read" << std::endl;

    // (VI) Final read and return on all ranks -- UNLESS write-only. For a
    //      cache-generation run the file is already banked by the merge, and
    //      reloading the full expansion on every rank OOMs at large Nmax; set
    //      SAM_EXPAND_WRITE_ONLY=1 to skip it and return empty.
    if (std::getenv("SAM_EXPAND_WRITE_ONLY"))
    {
      std::cerr << "[expand_new] Rank " << mpi_rank
                << ": WRITE_ONLY -- skipping final read-back" << std::endl;
      return U3xU4::MultiShellTensors_phys{};
    }

    std::ifstream s(filename, std::ios::binary);
    if (!s) throw std::runtime_error("Failed to open for final read: " + filename);

    HO::qn_t maxShell;
    qcx_utils::readBinary(s, maxShell);

    // Read number of tensors
    size_t n_tensors;
    qcx_utils::readBinary(s, n_tensors);
    std::cerr << "[expand_new] Rank " << mpi_rank << ": final read expects "
              << n_tensors << " tensors" << std::endl;

    U3xU4::MultiShellTensors_phys tensors;
    tensors.reserve(n_tensors);

    for (size_t i = 0; i < n_tensors; ++i)
    {
      U3xU4::MultiShellTensor_phys tensor(HO_labels_);
      tensor.readBinary(s);
      tensors.push_back(std::move(tensor));
      if ((i + 1) % 1000 == 0 || i + 1 == n_tensors)
        std::cerr << "[expand_new] Rank " << mpi_rank << ": read tensor "
                  << (i + 1) << " / " << n_tensors << std::endl;
    }

    std::cerr << "[expand_new] Rank " << mpi_rank << ": final read done" << std::endl;
    return tensors;
  }

  void TwoBodyOperator::expand_exist(int Nmax_old, int maxShell_old,
                              U3xU4::MultiShellTensors_phys& tensors)
  {
    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->read();
      else
      {
        assert(TBME_general);
        TBME_general->read();
        JJ_ = static_cast<HO::qn_t>(TBME_general->JJ());
        TT_ = static_cast<HO::qn_t>(TBME_general->TT());
        parity_ = static_cast<HO::qn_t>(TBME_general->parity());
      }
    }

    // Shared cache of single-shell tensor decompositions (thread-safe),
    // created once for the whole run and finalized after the loop.
    U3xU4::SSTDecomp_phys_db decompdb(maxShell_);

#ifdef RESTRICT_WITH_NMAX
    auto [all_na_nb_nc_nd, all_inverse, all_phase] =
        GenerateAllCombinationsHOShells(Nmax_, valenceShell_, maxShell_,
                                        Nmax_old, maxShell_old);
#else
    auto [all_na_nb_nc_nd, all_inverse, all_phase] = 
        GenerateAllCombinationsHOShells(maxShell_);
#endif

std::cerr << all_na_nb_nc_nd.size() << std::endl;

    for (size_t i = 0; i < all_na_nb_nc_nd.size(); i++)
    {
      // auto groups_ns = group_same_shell(sorted_ns);
      auto groups_ns = group_same_shell(all_na_nb_nc_nd[i]);
      std::vector<U3xU4::SingleShellTensors> Gammas_lists;
      for (const auto& g : groups_ns)
      {
        auto Gammas = U3xU4::GenerateSingleShellTensors(
            std::get<0>(g), std::get<1>(g), std::get<2>(g));
        Gammas_lists.push_back(std::move(Gammas));
      }

      std::vector<U3xU4::CouplingPath> all_results;
      U3xU4::GenerateMultiShellTensors(
        Gammas_lists, 0, {}, {}, {}, {}, all_results);

std::cerr << i << "/" << all_na_nb_nc_nd.size() << " : " << all_results.size() << std::endl;

      // for (const auto& [Gammas, Omegas] : all_results)

#pragma omp parallel
{
  sam::initialize_thread();

  U3xU4::MultiShellTensors_phys tensors_tl;

#pragma omp for schedule(dynamic,1)
      for (size_t j = 0; j < all_results.size(); j++)
      {
        const auto& Gammas = all_results[j].first;
        const auto& Omegas = all_results[j].second;

        U3xU4::MultiShellTensor_phys tensor(Gammas, Omegas, 
                                            MMT_, JJ_, MMJ_, HO_labels_);

        // selection rules
        if (U3selection(tensor.getLastIrrep().u3()) == false) continue;
        if (U4selection(tensor.getLastIrrep().u4()) == false) continue;
        if (tensor.shift_quanta() < 0 and symmetric_) continue;

        setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);

        if (!tensor.LLSSTTstrengths().empty())
          tensors_tl.push_back(std::move(tensor));
          // tensors.push_back(std::move(tensor));
      }

#pragma omp critical
  {
    tensors.reserve(tensors.size() + tensors_tl.size());

    for (auto& t : tensors_tl)
      tensors.push_back(std::move(t));
  }

  sam::finalize_thread();
}
    }

    // flush newly computed decompositions to disk
    decompdb.finalize();

    // clear hash tables
    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->clear();
      else TBME_general->clear();
    }
  }

  void TwoBodyOperator::expand_exist_MPI(const std::string& filename,
                                      int Nmax_old, int maxShell_old)
  {
    if (valenceShell_ > 0) throw std::runtime_error("Set valenceShell to 0 for reusability");

    // if (Nmax_ != maxShell_) throw std::runtime_error("Set Nmax and maxShell equal");

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->read();
      else
      {
        assert(TBME_general);
        TBME_general->read();
        JJ_ = static_cast<HO::qn_t>(TBME_general->JJ());
        TT_ = static_cast<HO::qn_t>(TBME_general->TT());
        parity_ = static_cast<HO::qn_t>(TBME_general->parity());
      }
    }

#ifdef RESTRICT_WITH_NMAX
    auto [all_na_nb_nc_nd, all_inverse, all_phase] =
        GenerateAllCombinationsHOShells(Nmax_, valenceShell_, maxShell_,
                                        Nmax_old, maxShell_old);
#else
    auto [all_na_nb_nc_nd, all_inverse, all_phase] = 
                      GenerateAllCombinationsHOShells(maxShell_);
#endif

// std::cerr << all_na_nb_nc_nd.size() << std::endl;

    // (I) Generate the entire list of tensors --- identical for all ranks

    // Under the default "strided" partition, apply ownership at BUILD time so
    // each rank stores only its share of tensor STRUCTURES. The full 398788-
    // tensor list is ~38 GB/rank at Nmax=12, and replicating it on every rank
    // (it never distributes) is what OOM'd the multi-node run. g_tensor is the
    // global tensor index -- the enumeration is identical on all ranks, so the
    // strided split is consistent. "weighted" still needs the whole list, so
    // under it every rank stores all.
    const bool weighted_partition =
      [] { const char* m = std::getenv("SAM_EXPAND_PARTITION");
           return m != nullptr && std::string(m) == "weighted"; }();

    std::vector<std::pair<size_t, U3xU4::MultiShellTensor_phys>> tensors_rank;
    size_t g_tensor = 0; // global index over all selection-passing tensors

    for (size_t i = 0; i < all_na_nb_nc_nd.size(); i++)
    {
      // auto groups_ns = group_same_shell(sorted_ns);
      auto groups_ns = group_same_shell(all_na_nb_nc_nd[i]);
      std::vector<U3xU4::SingleShellTensors> Gammas_lists;
      for (const auto& g : groups_ns)
      {
        auto Gammas = U3xU4::GenerateSingleShellTensors(
            std::get<0>(g), std::get<1>(g), std::get<2>(g));
        Gammas_lists.push_back(std::move(Gammas));
      }

      std::vector<U3xU4::CouplingPath> all_results;
      U3xU4::GenerateMultiShellTensors(
        Gammas_lists, 0, {}, {}, {}, {}, all_results);

// std::cerr << i << "/" << all_na_nb_nc_nd.size() << " : " << all_results.size() << std::endl;

      // for (const auto& [Gammas, Omegas] : all_results)
      for (size_t j = 0; j < all_results.size(); j++)
      {
        const auto& Gammas = all_results[j].first;
        const auto& Omegas = all_results[j].second;

        U3xU4::MultiShellTensor_phys tensor(Gammas, Omegas, 
                                            MMT_, JJ_, MMJ_, HO_labels_);

        // selection rules
        if (U3selection(tensor.getLastIrrep().u3()) == false) continue;
        if (U4selection(tensor.getLastIrrep().u4()) == false) continue;
        if (tensor.shift_quanta() < 0 and symmetric_) continue;

        // Ownership at BUILD time: strided stores only this rank's share so the
        // full structure list is never materialized; weighted stores all.
        const bool owned = weighted_partition ||
                           (g_tensor % (size_t)mpi_size == (size_t)mpi_rank);
        ++g_tensor;
        if (owned) tensors_rank.emplace_back(i, std::move(tensor));
      }
    }

    // (II) Assign tensors to this rank, then expand via OpenMP over the
    //      assignment. Strategy chosen by env SAM_EXPAND_PARTITION:
    //
    //   "strided" (DEFAULT, known-good): round-robin it = mpi_rank, += mpi_size.
    //     Balances tensor COUNT and finely interleaves cheap/expensive combos so
    //     every rank gets an even cost mix (~99% CPU historically). Downside: it
    //     scatters shell-combos across ranks, so per-rank m-scheme caches bloat
    //     toward the full table (~2x memory skew) -- fine on a big-RAM node with
    //     --mem=0. Reproduces the original (rank-interleaved) output order.
    //
    //   "weighted": equal-weight CONTIGUOUS blocks via a shell-degeneracy proxy
    //     Q = (na+1)(na+2)...(nd+1)(nd+2), charged once per combo. Localizes each
    //     combo's cache to one rank and yields canonical (global-index) output.
    //     BUT the proxy mispredicts COMPUTE for expand_exist -- high-shell combos
    //     are cheap (many filter to empty strengths) -- so blocks end up badly
    //     compute-imbalanced (one rank drains early and idles). DO NOT use until
    //     the weight reflects surviving-strength work, not shell size. Kept for
    //     that future fix.
    //
    // (A single rank x N threads is a third option with perfect balance, but on
    //  a multi-socket node the one address space serializes page faults from N
    //  threads growing uncapped caches and loses NUMA locality -> pathologically
    //  slow. Prefer several ranks: separate address spaces, NUMA-local.)
    std::vector<size_t> my_indices;
    if (weighted_partition)
    {
      // weighted stored the FULL list above; cut it into equal-weight
      // contiguous blocks (broken cost proxy -- see note above; kept only for
      // the future weight fix).
      std::vector<size_t> block_begin(mpi_size + 1, tensors_rank.size());
      block_begin[0] = 0;

      // shell-degeneracy proxy: #m-scheme s.p. states in HO shell n
      auto shell_deg = [](int n) -> double { return double((n + 1) * (n + 2)); };

      std::unordered_map<size_t, size_t> combo_count;
      for (const auto& [i, t] : tensors_rank) combo_count[i]++;

      std::vector<double> w(tensors_rank.size(), 0.0);
      double total = 0.0;
      for (size_t it = 0; it < tensors_rank.size(); ++it)
      {
        const auto& s = all_na_nb_nc_nd[tensors_rank[it].first];
        double Q = shell_deg(s[0]) * shell_deg(s[1])
                 * shell_deg(s[2]) * shell_deg(s[3]);
        w[it] = Q / double(combo_count[tensors_rank[it].first]);
        total += w[it];
      }

      const double target = (mpi_size > 0) ? total / double(mpi_size) : total;
      double acc = 0.0;
      int r = 1;
      for (size_t it = 0; it < tensors_rank.size() && r < mpi_size; )
      {
        size_t i = tensors_rank[it].first;
        while (it < tensors_rank.size() && tensors_rank[it].first == i)
          { acc += w[it]; ++it; }
        if (acc >= target * r) block_begin[r++] = it;
      }
      for (; r <= mpi_size; ++r) block_begin[r] = tensors_rank.size();

      for (size_t it = block_begin[mpi_rank]; it < block_begin[mpi_rank + 1]; ++it)
        my_indices.push_back(it);
    }
    else // strided: ownership already applied at build time -> own all stored
    {
      my_indices.resize(tensors_rank.size());
      for (size_t it = 0; it < tensors_rank.size(); ++it) my_indices[it] = it;
    }

    // Memory-cost proxy for one tensor (shell-degeneracy product). Used both to
    // front-load heavy work (SAM_EXPAND_HEAVY_FIRST) and to gate concurrent heavy
    // computes (SAM_EXPAND_HEAVY_SEM). Imperfect -- it mis-ranks the true decomp
    // size in detail (the real giants can sit mid-run, not first) -- but monotone
    // enough to separate the memory-heavy tail from the light bulk.
    auto heaviness = [&](size_t it) {
      const auto& s = all_na_nb_nc_nd[tensors_rank[it].first];
      auto d = [](int n) { return double((n + 1) * (n + 2)); };
      return d(s[0]) * d(s[1]) * d(s[2]) * d(s[3]);
    };

    // Fail-fast tuning (SAM_EXPAND_HEAVY_FIRST): reorder this rank's work so the
    // memory-heaviest tensors are processed FIRST, so a config that will OOM
    // fails cheaply instead of hours in. Sorts the INDEX list (no tensor moves);
    // changes output order -> re-validate, physics unchanged.
    if (std::getenv("SAM_EXPAND_HEAVY_FIRST"))
      std::stable_sort(my_indices.begin(), my_indices.end(),
        [&](size_t a, size_t b) { return heaviness(a) > heaviness(b); });

    const size_t my_count = my_indices.size();
    std::cerr << "Rank " << mpi_rank << ": processing " << my_count
              << " / " << g_tensor << " tensors" << std::endl;

    // Shared cache of single-shell tensor decompositions (thread-safe)
    U3xU4::SSTDecomp_phys_db decompdb(maxShell_);

    const std::size_t cache_cap = mscheme_cache_cap();

    // (III) Stream the expansion in chunks: compute a chunk, then flush its
    //       tensors to the rank file and free them. Peak RAM per rank = caches +
    //       one chunk (see expand_new_MPI for the full rationale). Byte-identical
    //       output to the non-streaming path; count is a placeholder patched via
    //       seekp at the end.
    const size_t flush_chunk = [] {
      const char* e = std::getenv("SAM_EXPAND_FLUSH_CHUNK");
      size_t c = e ? std::strtoull(e, nullptr, 10) : 0;
      return c ? c : 2000;
    }();

    // opt-in per-rank memory monitor (SAM_EXPAND_MEMLOG=1): prints RSS + cache
    // attribution at each chunk boundary, non-collective (no cross-rank sync).
    const bool memlog = std::getenv("SAM_EXPAND_MEMLOG") != nullptr;
    size_t ms_entries_acc = 0; // shared: per-chunk sum of thread m-scheme sizes

    // Coeff-cache probe/fix: clear the u3lib coupling caches every N chunks to
    // measure their footprint (RSS freed) and bound their growth. 0/unset = off.
    const size_t coeff_clear_n = [] {
      const char* e = std::getenv("SAM_EXPAND_COEFF_CLEAR");
      return e ? std::strtoull(e, nullptr, 10) : (size_t)0;
    }();

    std::string rank_filename = filename + ".rank_" + std::to_string(mpi_rank);
    std::cerr << "[expand_exist] Rank " << mpi_rank << ": opening rank file "
              << rank_filename << std::endl;
    std::ofstream rf(rank_filename, std::ios::binary);
    if (!rf) throw std::runtime_error("Failed to open rank file: " + rank_filename);
    const auto count_pos = rf.tellp();     // rank file starts with the count
    size_t n_written = 0;
    qcx_utils::writeBinary(rf, n_written); // placeholder, patched below

#pragma omp parallel
{
  sam::initialize_thread();

    for (size_t base = 0; base < my_indices.size(); base += flush_chunk)
    {
      const size_t stop = std::min(base + flush_chunk, my_indices.size());

      // compute this chunk in parallel
#pragma omp for schedule(dynamic,1)
      for (size_t k = base; k < stop; ++k)
      {
        const size_t it = my_indices[k];
        auto& [i, tensor] = tensors_rank[it];

        setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);
        HO::maybe_release_twbme_cache(cache_cap);

        if (k % 1000 == 0)
        {
#pragma omp critical
          std::cerr << "Rank " << mpi_rank << " [thread " << omp_get_thread_num()
                    << "]: " << k << " / " << my_count << std::endl;
        }
      } // implicit barrier: chunk fully computed before flush

      // memory monitor: each thread adds its m-scheme cache size before the
      // single thread reads the total (memlog is uniform across threads, so the
      // barrier is reached by all).
      if (memlog)
      {
#pragma omp atomic
        ms_entries_acc += HO::twbme_mscheme_cache().size();
#pragma omp barrier
      }

      // one thread streams this chunk to disk and frees it; the rest wait
#pragma omp single
      {
        for (size_t k = base; k < stop; ++k)
        {
          auto tensor = std::move(tensors_rank[my_indices[k]].second);
          if (!tensor.LLSSTTstrengths().empty())
          {
            tensor.writeBinary(rf);
            ++n_written;
          }
          // tensor destructs here: chunk's strengths freed before the next chunk
        }

        // bound the decomp cache: evict LRU files over SAM_DECOMP_CACHE_CAP.
        // Safe here -- all threads are parked at the barrier, so no borrowed
        // ExpansionRef is live.
        decompdb.evict_to_cap();

        // Return this chunk's freed pages (the m-scheme maps released during the
        // parallel for, plus anything evict_to_cap dropped) to the OS. Threads
        // are parked here, so the malloc lock is uncontended.
        trim_heap();

        // Coeff-cache probe/fix (SAM_EXPAND_COEFF_CLEAR=N): every N chunks wipe
        // the u3lib coupling-coefficient caches (uncapped, shared per rank, never
        // released otherwise) and record the RSS they freed -- a direct footprint
        // measurement and a periodic-clear knob. Safe here (all threads parked).
        // -1 = no clear this chunk.
        double coeff_freed_gb = -1.0;
        if (coeff_clear_n && ((base / flush_chunk) % coeff_clear_n == 0))
        {
          const double r0 = rss_now_gb();
          u3::clear();
          trim_heap();
          coeff_freed_gb = r0 - rss_now_gb();
        }

        if (memlog)
        {
          auto [dfiles, drecs] = decompdb.cache_stats();
          std::cerr << "[MEM] Rank " << mpi_rank << " @ " << stop << " / "
                    << my_indices.size() << ": rss=" << rss_now_gb()
                    << " peak=" << rss_peak_gb() << " GB  mscheme="
                    << ms_entries_acc << " entries  decomp=" << dfiles
                    << " files / " << drecs << " recs / "
                    << decompdb.cache_bytes() / 1e9 << " GB  coeff_freed="
                    << coeff_freed_gb << " GB" << std::endl;
          ms_entries_acc = 0;
        }
      } // implicit barrier: flush done before the next chunk computes
    }

  sam::finalize_thread();
}

    // patch the real count into the placeholder slot at the head of the file
    if (!rf) throw std::runtime_error("Write failed on rank file: " + rank_filename);
    rf.seekp(count_pos);
    qcx_utils::writeBinary(rf, n_written);
    rf.close();
    std::cerr << "[expand_exist] Rank " << mpi_rank << ": rank file closed OK ("
              << n_written << " tensors)" << std::endl;

    // tensors_rank now holds only moved-from shells
    tensors_rank.clear();
    tensors_rank.shrink_to_fit();

    // (IV) Free everything no longer needed. Runs AFTER the streaming loop
    //      because the compute used TBME / all_inverse / all_phase / decompdb.
    // (a) release the TBME hash table
    if (on_the_fly == false)
    {
      if (TBME_scalar) TBME_scalar->clear();
      else TBME_general->clear();
    }
    // (b) release shell-combo arrays
    { decltype(all_na_nb_nc_nd)().swap(all_na_nb_nc_nd); }
    { decltype(all_inverse)().swap(all_inverse); }
    { decltype(all_phase)().swap(all_phase); }
    // (c) release u3/u4 databases
    sam::clear();
    // (d) flush newly computed decompositions to disk
    decompdb.finalize();

    std::cerr << "[expand_exist] Rank " << mpi_rank << ": tensors freed, entering barrier" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD); // wait for all rank files to be written
    std::cerr << "[expand_exist] Rank " << mpi_rank << ": past write barrier" << std::endl;

    // (V) Rank 0 appends all rank tensor bytes to the existing file.
    //     Raw byte streaming — no tensor deserialization, O(1) memory.
    if (mpi_rank == 0)
    {
      // Open existing file: read header, seek to end, stream-append all rank files
      std::cerr << "[expand_exist] Rank 0: opening existing file " << filename << std::endl;
      std::fstream f(filename, std::ios::in | std::ios::out | std::ios::binary);
      if (!f) throw std::runtime_error("Failed to open: " + filename);
      auto maxShell_pos = f.tellg();       // byte 0: position of maxShell field
      HO::qn_t maxShell;
      qcx_utils::readBinary(f, maxShell);
      auto count_pos = f.tellg();          // right after maxShell: position of n_tensors field
      size_t n_tensors_prev;
      qcx_utils::readBinary(f, n_tensors_prev);
      std::cerr << "[expand_exist] Rank 0: existing file has " << n_tensors_prev
                << " tensors (maxShell " << (int)maxShell << " -> " << (int)maxShell_ 
                << "), seeking to end" << std::endl;

      f.seekp(0, std::ios::end);
      size_t n_new = 0;
      for (int r = 0; r < mpi_size; ++r)
      {
        std::string rf_name = filename + ".rank_" + std::to_string(r);
        std::cerr << "[expand_exist] Rank 0: appending rank file " << r << std::endl;
        std::ifstream rf(rf_name, std::ios::binary);
        size_t n; qcx_utils::readBinary(rf, n); // read per-rank count
        std::cerr << "[expand_exist] Rank 0: rank " << r << " has " << n << " tensors" << std::endl;
        n_new += n;
        if (n > 0)  // rdbuf() sets failbit if 0 bytes extracted — skip for empty ranks
        {
          f << rf.rdbuf();                       // raw byte append — no deserialization
          if (!f) throw std::runtime_error("Write failed after appending rank file " + rf_name);
        }
        rf.close();
        std::remove(rf_name.c_str());
        std::cerr << "[expand_exist] Rank 0: rank file " << r << " appended and removed" << std::endl;
      }

      size_t total = n_tensors_prev + n_new;
      std::cerr << "[expand_exist] Rank 0: updating header (maxShell=" << maxShell_
                << ", n_tensors=" << total << ")" << std::endl;
      f.seekp(maxShell_pos);
      qcx_utils::writeBinary(f, maxShell_);  // update maxShell to new (larger) value
      qcx_utils::writeBinary(f, total);      // update tensor count (lands at count_pos)
      f.close();

      std::cerr << "[expand_exist] Rank 0: appended " << n_new
                << " tensors, total = " << total << std::endl;
    }

    std::cerr << "[expand_exist] Rank " << mpi_rank << ": entering final barrier" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::cerr << "[expand_exist] Rank " << mpi_rank << ": past final barrier" << std::endl;

/*
    // Final read and return on all ranks -- UNLESS write-only (cache-gen run):
    // the file is already banked by the merge, and reloading the full expansion
    // on every rank OOMs at large Nmax. SAM_EXPAND_WRITE_ONLY=1 skips it.
    if (std::getenv("SAM_EXPAND_WRITE_ONLY"))
    {
      std::cerr << "[expand_exist] Rank " << mpi_rank
                << ": WRITE_ONLY -- skipping final read-back" << std::endl;
      return U3xU4::MultiShellTensors_phys{};
    }

    std::ifstream s(filename, std::ios::binary);
    if (!s) throw std::runtime_error("Failed to open: " + filename);

    HO::qn_t maxShell;
    qcx_utils::readBinary(s, maxShell);

    // Read number of tensors
    size_t n_tensors;
    qcx_utils::readBinary(s, n_tensors);

    U3xU4::MultiShellTensors_phys tensors;
    tensors.reserve(n_tensors);

    for (size_t i = 0; i < n_tensors; ++i)
    { 
      U3xU4::MultiShellTensor_phys tensor(HO_labels_);
      tensor.readBinary(s);

      tensors.push_back(std::move(tensor));
    }

    return tensors;
*/
  }

  void TwoBodyOperator::scaleStrengths(U3xU4::MultiShellTensors_phys& tensors) const
  {
    if (!on_the_fly) return;
    if (name_ == "Kinetic")
      for (auto& t : tensors) t.enhanceStrengths(static_cast<double>(hw_) / 20.0);
    else if (name_ == "Rrel")
      for (auto& t : tensors) t.enhanceStrengths(20.0 / static_cast<double>(hw_));
  }

  U3xU4::MultiShellTensors_phys
    TwoBodyOperator::expand()
  {
    std::string base_dir = dirname() + "/Expansion";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_ + "_hw";
    if (on_the_fly) filename += "20";
    else filename += std::to_string(static_cast<int>(hw_));

    // if this file exists, returns immediately
    if (std::filesystem::exists(filename))
    {
#ifdef USE_MPI
      return readBinary_MPI(filename); // rescale internally
#else
      return readBinary(filename); // rescale internally
#endif
    }
    else
    {
      auto requested_hw = hw_;
      // only change hw to 20 if the matrix elements are computed on the fly
      if (on_the_fly) hw_ = 20;

#ifdef USE_MPI
      auto tensors = expand_new_MPI(filename);
#else
      auto tensors = expand_new();
      writeBinary(filename, tensors); // persist hw=20
#endif

      hw_ = requested_hw;
      scaleStrengths(tensors); // scale to requested hw
      
      return tensors;
    }
  }

} // namespace JT
