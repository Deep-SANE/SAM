#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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

#include <convert.h>
#include <print.h>
using utils::print;
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>
#include <so3Coeffs/SO3_Fano.h>

#include <sam.h>
#include <u3lib.h>   // u3::clear(): wipe the coupling-coefficient caches

#include <TwoBodyOperatorJT.h>
#include <ExpansionFiles.h>

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

  // Combined kinematic selection shared by all four expand paths (declared in
  // the header). Returns false for a candidate tensor whose matrix element is
  // known a priori to vanish, so it can be skipped before the expensive
  // strength computation.
  bool TwoBodyOperator::passes_selection(
      const U3xU4::MultiShellTensor_phys& tensor) const
  {
    if (!U3selection(tensor.getLastIrrep().u3())) return false;
    if (!U4selection(tensor.getLastIrrep().u4())) return false;
    if (tensor.shift_quanta() < 0 && symmetric_) return false;
    const auto u3 = tensor.getLastIrrep().u3();

    // Central 2-body kinematic rule: for a genuine scalar potential the terminal
    // SU(3) asymmetry equals the net HO quanta shift (lam - mu == shift_quanta),
    // else the matrix element vanishes (diff == -3*h2; scalar forces require
    // h2 == 0). Gated to recognized scalar potentials ONLY: it has real
    // false-negatives on normal-ordered 2B+3B (genuine 3-body content gives
    // diff in {0,+-3,+-6,+-9}) and on non-scalar operators. On scalar potentials
    // the only kept tensors it drops carry |strength| < ~1e-4 numerical noise.
    if (known_potentials.count(name_))
    {
      if (u3.lam() - u3.mu() != tensor.shift_quanta()) return false;

      // Spin-isospin exchange (SU(4) 6 vs 10/10bar) must match between the
      // creation pair and the annihilation pair. The SU(4) exchange type is the
      // parity of S+T (6 == S+T odd, 10/10bar == S+T even), and a genuine 2-body
      // scalar interaction conserves S+T parity, so it cannot connect a symmetric
      // spin-isospin pair to an antisymmetric one. Like the lam-mu==shift rule
      // this is a 2-body-structure rule: validated 0 false-negatives on all
      // scalar NN potentials up to Nmax12, but NOT applied to NO2B, whose 3-body
      // content breaks it (318 real false-negatives on Be7/Li7).
      int cre_sign = 0, ann_sign = 0;
      for (const auto& G : tensor.Gammas())
      {
        const auto& p = G.u4();
        int s = 0;
        if (p.P1() == 0 && p.P2() == 1 && p.P3() == 0) s = -1;          // 6
        else if ((p.P1() == 2 && p.P2() == 0 && p.P3() == 0) ||          // 10
                 (p.P1() == 0 && p.P2() == 0 && p.P3() == 2)) s = +1;    // 10bar
        if (s == 0) continue;
        if (G.num_ads() == 2 && G.num_ats() == 0) cre_sign = s;
        else if (G.num_ads() == 0 && G.num_ats() == 2) ann_sign = s;
      }
      if (cre_sign != 0 && ann_sign != 0 && cre_sign != ann_sign) return false;
    }

    // Pauli + SU(4)-singlet rules apply to any JJ==0 nuclear scalar interaction:
    // recognized scalar potentials AND NO2B-truncated operators (the else-branch
    // in the constructor: read from TBME, not a known general operator, JJ==0).
    // They are antisymmetry / supermultiplet selection rules independent of the
    // potential and of 3-body content; validated 0 false-negatives on N2LO_opt/
    // sat (up to Nmax12) and on NO2B (Be7, Li7). Helper ops (L.L, L.S, Kinetic,
    // ... : on_the_fly) and non-scalar general ops (GamowTeller, JJ!=0) are
    // excluded, not having been validated.
    const bool scalar_nuclear =
        !on_the_fly && JJ() == 0 && known_general_ops.count(name_) == 0;
    if (scalar_nuclear)
    {
      // SU(4)-singlet parity rule: when the terminal SU(4) irrep is the singlet
      // (spin-isospin scalar channel, P1=P2=P3=0), the spatial SU(3) irrep must
      // lie in the space-symmetric sector, i.e. lam and mu both even; an odd
      // (lam,mu) forces compensating spin-isospin exchange the singlet cannot
      // carry. Catches the bulk of the density-chain residual the lam-mu==shift
      // rule leaves (that rule is vacuous for number-conserving density tensors,
      // shift_quanta==0).
      {
        const auto p = tensor.getLastIrrep().u4();
        if (p.P1() == 0 && p.P2() == 0 && p.P3() == 0 &&
            (u3.lam() % 2 != 0 || u3.mu() % 2 != 0))
          return false;
      }

      // Per-pair Pauli antisymmetrization rule: a genuine same-shell two-particle
      // factor (a+a+, num_ads==2) or two-hole factor (aa, num_ats==2) must have
      // its spatial SU(3) exchange symmetry opposite to its SU(4) exchange
      // symmetry (overall antisymmetric). The SU(3) exchange sign is (-1)^mu for
      // a creation pair in (n,0)x(n,0) and (-1)^lam for an annihilation pair in
      // (0,n)x(0,n); the SU(4) sign is +1 for the symmetric 10/10bar and -1 for
      // the antisymmetric 6.
      for (const auto& G : tensor.Gammas())
      {
        const bool cre = (G.num_ads() == 2 && G.num_ats() == 0);
        const bool ann = (G.num_ads() == 0 && G.num_ats() == 2);
        if (!cre && !ann) continue; // density/single factors carry no Pauli rule
        const auto& p = G.u4();
        int su4_sign = 0;
        if (p.P1() == 0 && p.P2() == 1 && p.P3() == 0) su4_sign = -1;   // 6
        else if ((p.P1() == 2 && p.P2() == 0 && p.P3() == 0) ||          // 10
                 (p.P1() == 0 && p.P2() == 0 && p.P3() == 2)) su4_sign = +1; // 10bar
        if (su4_sign == 0) continue;
        const int q = cre ? G.u3().mu() : G.u3().lam();
        const int spatial_sign = (q % 2 == 0) ? +1 : -1;
        if (spatial_sign * su4_sign != -1) return false;
      }
    }
    return true;
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
        if (!passes_selection(tensor)) continue;

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

    // (e) compact: keep only tensors whose matrix elements survived (non-empty
    //     LLSSTT strengths). A tensor that passed the kinematic selection but
    //     produced empty strengths is dropped HERE -- i.e. the real ~half
    //     rejection is post-ME, not kinematic.
    //
    //     Discovery mode (env SAM_EXPAND_DISCOVERY): emit the terminal-irrep
    //     labels of every tensor tagged KEEP / IGNORE, so a post-ME selection
    //     pattern can be searched for. Intended for small-Nmax runs.
    //       grep '\[DISCOVERY\] IGNORE' <log>   -> the vanishing tensors
    const bool discovery = (std::getenv("SAM_EXPAND_DISCOVERY") != nullptr);
    U3xU4::MultiShellTensors_phys tensors_final;
    tensors_final.reserve(tensors.size());
    for (auto& [i, t] : tensors)
    {
      const bool kept = !t.LLSSTTstrengths().empty();
      if (discovery)
        std::cout << "[DISCOVERY] " << (kept ? "KEEP  " : "IGNORE")
                  << " u3=" << t.getLastIrrep().u3()
                  << " u4=" << t.getLastIrrep().u4()
                  << " shift_q=" << t.shift_quanta()
                  << " nLSTT=" << t.LLSSTTstrengths().size()
                  << " | " << t << std::endl;
      if (kept) tensors_final.push_back(std::move(t));
    }

    return tensors_final;
  }

  U3xU4::MultiShellTensors_phys 
    TwoBodyOperator::expand_new_MPI(const std::string& filename)
  {
    sam::require_writable(filename);

    if (valenceShell_ > 0) throw std::runtime_error("Set valenceShell to 0 for reusability");

    // if (Nmax_ != maxShell_) throw std::runtime_error("Set Nmax and maxShell equal");

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Discovery mode (env SAM_EXPAND_DISCOVERY): at each write point, emit the
    // terminal-irrep labels of every tensor tagged KEEP / IGNORE, so a post-ME
    // selection pattern can be searched for (IGNORE = passed kinematic selection
    // but produced empty strengths). Intended for small-Nmax, single-rank runs.
    //   grep '\[DISCOVERY\] IGNORE' <log>   -> the vanishing tensors
    const bool discovery = (std::getenv("SAM_EXPAND_DISCOVERY") != nullptr);

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

    // Dynamic work-queue (SAM_EXPAND_DYNAMIC_BATCH=N, N>0): replace the static
    // partition with a shared MPI atomic counter -- every rank pulls the next N
    // tensors, so a rank that finishes early steals the slow giant-tail from a
    // laggard (no idle ranks). Needs >1 rank and ALL tensors built on every rank
    // (see `owned` below). Trades higher peak memory (more ranks computing giants
    // at once) for wall time -- ideal on big-RAM nodes. 0/unset = static.
    const size_t dyn_batch = [] {
      const char* e = std::getenv("SAM_EXPAND_DYNAMIC_BATCH");
      return e ? std::strtoull(e, nullptr, 10) : (size_t)0;
    }();
#ifdef USE_MPI
    const bool dynamic_mode = (dyn_batch > 0) && (mpi_size > 1);
#else
    const bool dynamic_mode = false;
#endif

    // Resume from a prior (static) run: skip tensors it already finished. Their
    // GLOBAL build indices are residues of that run's rank count -- a killed
    // 4-rank run whose ranks 1,3 completed leaves residues {1,3} mod 4 done.
    // SAM_EXPAND_DONE_MODULUS=M + SAM_EXPAND_DONE_RESIDUES=csv skip them here;
    // SAM_EXPAND_DONE_DIR folds that run's .rank_* files into the merge below.
    // With dynamic mode, only the FAILED residues recompute, balanced over all
    // ranks -- reuse the done half, fix the imbalance on the rest.
    size_t done_modulus = 0;
    std::vector<int> done_residues;
    if (const char* m = std::getenv("SAM_EXPAND_DONE_MODULUS"))
      done_modulus = std::strtoull(m, nullptr, 10);
    if (const char* rs = std::getenv("SAM_EXPAND_DONE_RESIDUES"))
      for (const char* p = rs; *p; )
      {
        char* end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        done_residues.push_back((int)v);
        p = (*end == ',') ? end + 1 : end;
      }
    const bool done_skip = (done_modulus > 0) && !done_residues.empty();
    auto is_done = [done_skip, done_modulus, &done_residues](size_t k) -> bool
    {
      if (!done_skip) return false;
      const int res = (int)(k % done_modulus);
      for (int r : done_residues) if (r == res) return true;
      return false;
    };

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
        if (!passes_selection(tensor)) continue;

        // Ownership at BUILD time: strided stores only this rank's share so the
        // full structure list is never materialized; weighted stores all.
        // dynamic_mode needs every rank able to compute ANY claimed tensor, so
        // every rank builds the full list (structures are small -- the owned-only
        // build never moved memory materially).
        const bool owned = dynamic_mode || weighted_partition ||
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
    if (weighted_partition && !dynamic_mode)
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
    if (!dynamic_mode && std::getenv("SAM_EXPAND_HEAVY_FIRST"))
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
    utils::writeBinary(rf, n_written); // placeholder, patched below

    // Dynamic work-queue: a single global counter (on rank 0) that ranks
    // fetch-and-add to claim the next batch. Passive-target RMA, so a rank that
    // finishes its share keeps pulling until the counter passes n_total -- the
    // giant tail spreads across all ranks instead of stranding a laggard.
#ifdef USE_MPI
    MPI_Win   dyn_win     = MPI_WIN_NULL;
    uint64_t* dyn_counter = nullptr;
    if (dynamic_mode)
    {
      if (mpi_rank == 0)
      {
        MPI_Alloc_mem(sizeof(uint64_t), MPI_INFO_NULL, &dyn_counter);
        *dyn_counter = 0;
        MPI_Win_create(dyn_counter, sizeof(uint64_t), sizeof(uint64_t),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &dyn_win);
      }
      else
        MPI_Win_create(nullptr, 0, sizeof(uint64_t), MPI_INFO_NULL,
                       MPI_COMM_WORLD, &dyn_win);
      MPI_Win_lock_all(0, dyn_win);
    }
#endif

    // Progress for the STATIC path + a chunk counter for the coeff probe; both
    // are advanced only inside the omp single, so no race.
    size_t next_lo  = 0;
    size_t chunk_no = 0;

#pragma omp parallel
{
  sam::initialize_thread();

    while (true)
    {
      size_t base = 0, stop = 0;
      int    have_work = 0;

      // One thread picks the next [base, stop) range and broadcasts it to the
      // team (copyprivate). dynamic: an MPI atomic fetch-and-add on the shared
      // work counter (SERIALIZED-safe -- only this single thread calls MPI, and
      // it does so alone). static: a plain local counter, identical ranges/order
      // to the old for-loop so the output is unchanged.
#pragma omp single copyprivate(base, stop, have_work)
      {
        if (dynamic_mode)
        {
#ifdef USE_MPI
          uint64_t inc = (uint64_t)dyn_batch, s = 0;
          MPI_Fetch_and_op(&inc, &s, MPI_UINT64_T, 0, 0, MPI_SUM, dyn_win);
          MPI_Win_flush(0, dyn_win);
          if ((size_t)s < my_indices.size())
          {
            base = (size_t)s;
            stop = std::min(base + dyn_batch, my_indices.size());
            have_work = 1;
          }
#endif
        }
        else if (next_lo < my_indices.size())
        {
          base = next_lo;
          stop = std::min(base + flush_chunk, my_indices.size());
          next_lo = stop;
          have_work = 1;
        }
      } // implicit barrier + copyprivate broadcast (all threads agree on base/stop)

      if (!have_work) break; // all threads break together -> clean exit

      // compute this batch in parallel
#pragma omp for schedule(dynamic,1)
      for (size_t k = base; k < stop; ++k)
      {
        const size_t it = my_indices[k];
        if (is_done(it)) continue; // finished by a prior run -> reused from done/
        auto& [i, tensor] = tensors_rank[it];

        setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);
        HO::maybe_release_twbme_cache(cache_cap);

        if (k % 1000 == 0)
        {
#pragma omp critical
          std::cerr << "Rank " << mpi_rank << " [thread " << omp_get_thread_num()
                    << "]: " << k << " / " << my_count << std::endl;
        }
      } // implicit barrier: batch fully computed before flush

      // memory monitor: each thread adds its m-scheme cache size before the
      // single thread reads the total (memlog is uniform across threads, so the
      // barrier is reached by all).
      if (memlog)
      {
#pragma omp atomic
        ms_entries_acc += HO::twbme_mscheme_cache().size();
#pragma omp barrier
      }

      // one thread streams this batch to disk and frees it; the rest wait
#pragma omp single
      {
        for (size_t k = base; k < stop; ++k)
        {
          auto tensor = std::move(tensors_rank[my_indices[k]].second);
          const bool kept = !tensor.LLSSTTstrengths().empty();
          if (discovery)
            std::cout << "[DISCOVERY] " << (kept ? "KEEP  " : "IGNORE")
                      << " rank=" << mpi_rank
                      << " u3=" << tensor.getLastIrrep().u3()
                      << " u4=" << tensor.getLastIrrep().u4()
                      << " shift_q=" << tensor.shift_quanta()
                      << " nLSTT=" << tensor.LLSSTTstrengths().size()
                      << " | " << tensor << std::endl;
          if (kept)
          {
            tensor.writeBinary(rf);
            ++n_written;
          }
          // tensor destructs here: batch's strengths freed before the next batch
        }

        // bound the decomp cache: evict LRU files over SAM_DECOMP_CACHE_CAP.
        // Safe here -- all threads are parked at the barrier, so no borrowed
        // ExpansionRef is live.
        decompdb.evict_to_cap();

        // Return this batch's freed pages (the m-scheme maps released during the
        // parallel for, plus anything evict_to_cap dropped) to the OS. Threads
        // are parked here, so the malloc lock is uncontended.
        trim_heap();

        // Coeff-cache probe/fix (SAM_EXPAND_COEFF_CLEAR=N): every N batches wipe
        // the u3lib coupling-coefficient caches (uncapped, shared per rank, never
        // released otherwise) and record the RSS they freed -- a direct footprint
        // measurement and a periodic-clear knob. Safe here (all threads parked).
        // -1 = no clear this batch.
        double coeff_freed_gb = -1.0;
        if (coeff_clear_n && (chunk_no % coeff_clear_n == 0))
        {
          const double r0 = rss_now_gb();
          u3::clear();
          trim_heap();
          coeff_freed_gb = r0 - rss_now_gb();
        }
        ++chunk_no;

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
      } // implicit barrier: flush done before the next batch computes
    }

  sam::finalize_thread();
}

#ifdef USE_MPI
    if (dynamic_mode)
    {
      MPI_Win_unlock_all(dyn_win);
      MPI_Win_free(&dyn_win);        // collective: blocks until the last rank's
      if (dyn_counter) MPI_Free_mem(dyn_counter); // claims are done
    }
#endif

    // patch the real count into the placeholder slot at the head of the file
    if (!rf) throw std::runtime_error("Write failed on rank file: " + rank_filename);
    rf.seekp(count_pos);
    utils::writeBinary(rf, n_written);
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
      utils::writeBinary(out, maxShell_);
      auto count_pos = out.tellp();          // save position of n_tensors field
      size_t placeholder = 0;
      utils::writeBinary(out, placeholder);
      std::cerr << "[expand_new] Rank 0: header written, streaming rank files" << std::endl;

      size_t total_tensors = 0;
      for (int r = 0; r < mpi_size; ++r)
      {
        std::string rf_name = filename + ".rank_" + std::to_string(r);
        std::cerr << "[expand_new] Rank 0: streaming rank file " << r << std::endl;
        std::ifstream rf(rf_name, std::ios::binary);
        size_t n; utils::readBinary(rf, n); // read per-rank count
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

      // Fold in a prior run's completed rank files (SAM_EXPAND_DONE_DIR): stream
      // them into the output too, so the reused (skipped) tensors are present.
      // Kept on disk -- never removed.
      if (const char* dd = std::getenv("SAM_EXPAND_DONE_DIR"))
        for (const auto& e : std::filesystem::directory_iterator(dd))
        {
          if (e.path().filename().string().find(".rank_") == std::string::npos)
            continue;
          std::ifstream df(e.path(), std::ios::binary);
          if (!df) continue;
          size_t n; utils::readBinary(df, n);
          std::cerr << "[expand_new] Rank 0: folding done file " << e.path()
                    << " (" << n << " tensors)" << std::endl;
          total_tensors += n;
          if (n > 0)
          {
            out << df.rdbuf();
            if (!out)
              throw std::runtime_error("Write failed folding done file " +
                                       e.path().string());
          }
        }

      // Seek back and overwrite the placeholder with the real count
      out.seekp(count_pos);
      utils::writeBinary(out, total_tensors);
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
    utils::readBinary(s, maxShell);

    // Read number of tensors
    size_t n_tensors;
    utils::readBinary(s, n_tensors);
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
        if (!passes_selection(tensor)) continue;

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
    sam::require_writable(filename);

    if (valenceShell_ > 0) throw std::runtime_error("Set valenceShell to 0 for reusability");

    // if (Nmax_ != maxShell_) throw std::runtime_error("Set Nmax and maxShell equal");

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Discovery mode (env SAM_EXPAND_DISCOVERY): see expand_new_MPI. Tags each
    // written/ignored tensor with its terminal-irrep labels.
    const bool discovery = (std::getenv("SAM_EXPAND_DISCOVERY") != nullptr);

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

    // Dynamic work-queue (SAM_EXPAND_DYNAMIC_BATCH=N, N>0): replace the static
    // partition with a shared MPI atomic counter -- every rank pulls the next N
    // tensors, so a rank that finishes early steals the slow giant-tail from a
    // laggard (no idle ranks). Needs >1 rank and ALL tensors built on every rank
    // (see `owned` below). Trades higher peak memory (more ranks computing giants
    // at once) for wall time -- ideal on big-RAM nodes. 0/unset = static.
    const size_t dyn_batch = [] {
      const char* e = std::getenv("SAM_EXPAND_DYNAMIC_BATCH");
      return e ? std::strtoull(e, nullptr, 10) : (size_t)0;
    }();
#ifdef USE_MPI
    const bool dynamic_mode = (dyn_batch > 0) && (mpi_size > 1);
#else
    const bool dynamic_mode = false;
#endif

    // Resume from a prior (static) run: skip tensors it already finished. Their
    // GLOBAL build indices are residues of that run's rank count -- a killed
    // 4-rank run whose ranks 1,3 completed leaves residues {1,3} mod 4 done.
    // SAM_EXPAND_DONE_MODULUS=M + SAM_EXPAND_DONE_RESIDUES=csv skip them here;
    // SAM_EXPAND_DONE_DIR folds that run's .rank_* files into the merge below.
    // With dynamic mode, only the FAILED residues recompute, balanced over all
    // ranks -- reuse the done half, fix the imbalance on the rest.
    size_t done_modulus = 0;
    std::vector<int> done_residues;
    if (const char* m = std::getenv("SAM_EXPAND_DONE_MODULUS"))
      done_modulus = std::strtoull(m, nullptr, 10);
    if (const char* rs = std::getenv("SAM_EXPAND_DONE_RESIDUES"))
      for (const char* p = rs; *p; )
      {
        char* end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        done_residues.push_back((int)v);
        p = (*end == ',') ? end + 1 : end;
      }
    const bool done_skip = (done_modulus > 0) && !done_residues.empty();
    auto is_done = [done_skip, done_modulus, &done_residues](size_t k) -> bool
    {
      if (!done_skip) return false;
      const int res = (int)(k % done_modulus);
      for (int r : done_residues) if (r == res) return true;
      return false;
    };

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
        if (!passes_selection(tensor)) continue;

        // Ownership at BUILD time: strided stores only this rank's share so the
        // full structure list is never materialized; weighted stores all.
        // dynamic_mode needs every rank able to compute ANY claimed tensor, so
        // every rank builds the full list (structures are small -- the owned-only
        // build never moved memory materially).
        const bool owned = dynamic_mode || weighted_partition ||
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
    if (weighted_partition && !dynamic_mode)
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
    if (!dynamic_mode && std::getenv("SAM_EXPAND_HEAVY_FIRST"))
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
    utils::writeBinary(rf, n_written); // placeholder, patched below

    // Dynamic work-queue: a single global counter (on rank 0) that ranks
    // fetch-and-add to claim the next batch. Passive-target RMA, so a rank that
    // finishes its share keeps pulling until the counter passes n_total -- the
    // giant tail spreads across all ranks instead of stranding a laggard.
#ifdef USE_MPI
    MPI_Win   dyn_win     = MPI_WIN_NULL;
    uint64_t* dyn_counter = nullptr;
    if (dynamic_mode)
    {
      if (mpi_rank == 0)
      {
        MPI_Alloc_mem(sizeof(uint64_t), MPI_INFO_NULL, &dyn_counter);
        *dyn_counter = 0;
        MPI_Win_create(dyn_counter, sizeof(uint64_t), sizeof(uint64_t),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &dyn_win);
      }
      else
        MPI_Win_create(nullptr, 0, sizeof(uint64_t), MPI_INFO_NULL,
                       MPI_COMM_WORLD, &dyn_win);
      MPI_Win_lock_all(0, dyn_win);
    }
#endif

    // Progress for the STATIC path + a chunk counter for the coeff probe; both
    // are advanced only inside the omp single, so no race.
    size_t next_lo  = 0;
    size_t chunk_no = 0;

#pragma omp parallel
{
  sam::initialize_thread();

    while (true)
    {
      size_t base = 0, stop = 0;
      int    have_work = 0;

      // One thread picks the next [base, stop) range and broadcasts it to the
      // team (copyprivate). dynamic: an MPI atomic fetch-and-add on the shared
      // work counter (SERIALIZED-safe -- only this single thread calls MPI, and
      // it does so alone). static: a plain local counter, identical ranges/order
      // to the old for-loop so the output is unchanged.
#pragma omp single copyprivate(base, stop, have_work)
      {
        if (dynamic_mode)
        {
#ifdef USE_MPI
          uint64_t inc = (uint64_t)dyn_batch, s = 0;
          MPI_Fetch_and_op(&inc, &s, MPI_UINT64_T, 0, 0, MPI_SUM, dyn_win);
          MPI_Win_flush(0, dyn_win);
          if ((size_t)s < my_indices.size())
          {
            base = (size_t)s;
            stop = std::min(base + dyn_batch, my_indices.size());
            have_work = 1;
          }
#endif
        }
        else if (next_lo < my_indices.size())
        {
          base = next_lo;
          stop = std::min(base + flush_chunk, my_indices.size());
          next_lo = stop;
          have_work = 1;
        }
      } // implicit barrier + copyprivate broadcast (all threads agree on base/stop)

      if (!have_work) break; // all threads break together -> clean exit

      // compute this batch in parallel
#pragma omp for schedule(dynamic,1)
      for (size_t k = base; k < stop; ++k)
      {
        const size_t it = my_indices[k];
        if (is_done(it)) continue; // finished by a prior run -> reused from done/
        auto& [i, tensor] = tensors_rank[it];

        setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);
        HO::maybe_release_twbme_cache(cache_cap);

        if (k % 1000 == 0)
        {
#pragma omp critical
          std::cerr << "Rank " << mpi_rank << " [thread " << omp_get_thread_num()
                    << "]: " << k << " / " << my_count << std::endl;
        }
      } // implicit barrier: batch fully computed before flush

      // memory monitor: each thread adds its m-scheme cache size before the
      // single thread reads the total (memlog is uniform across threads, so the
      // barrier is reached by all).
      if (memlog)
      {
#pragma omp atomic
        ms_entries_acc += HO::twbme_mscheme_cache().size();
#pragma omp barrier
      }

      // one thread streams this batch to disk and frees it; the rest wait
#pragma omp single
      {
        for (size_t k = base; k < stop; ++k)
        {
          auto tensor = std::move(tensors_rank[my_indices[k]].second);
          const bool kept = !tensor.LLSSTTstrengths().empty();
          if (discovery)
            std::cout << "[DISCOVERY] " << (kept ? "KEEP  " : "IGNORE")
                      << " rank=" << mpi_rank
                      << " u3=" << tensor.getLastIrrep().u3()
                      << " u4=" << tensor.getLastIrrep().u4()
                      << " shift_q=" << tensor.shift_quanta()
                      << " nLSTT=" << tensor.LLSSTTstrengths().size()
                      << " | " << tensor << std::endl;
          if (kept)
          {
            tensor.writeBinary(rf);
            ++n_written;
          }
          // tensor destructs here: batch's strengths freed before the next batch
        }

        // bound the decomp cache: evict LRU files over SAM_DECOMP_CACHE_CAP.
        // Safe here -- all threads are parked at the barrier, so no borrowed
        // ExpansionRef is live.
        decompdb.evict_to_cap();

        // Return this batch's freed pages (the m-scheme maps released during the
        // parallel for, plus anything evict_to_cap dropped) to the OS. Threads
        // are parked here, so the malloc lock is uncontended.
        trim_heap();

        // Coeff-cache probe/fix (SAM_EXPAND_COEFF_CLEAR=N): every N batches wipe
        // the u3lib coupling-coefficient caches (uncapped, shared per rank, never
        // released otherwise) and record the RSS they freed -- a direct footprint
        // measurement and a periodic-clear knob. Safe here (all threads parked).
        // -1 = no clear this batch.
        double coeff_freed_gb = -1.0;
        if (coeff_clear_n && (chunk_no % coeff_clear_n == 0))
        {
          const double r0 = rss_now_gb();
          u3::clear();
          trim_heap();
          coeff_freed_gb = r0 - rss_now_gb();
        }
        ++chunk_no;

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
      } // implicit barrier: flush done before the next batch computes
    }

  sam::finalize_thread();
}

#ifdef USE_MPI
    if (dynamic_mode)
    {
      MPI_Win_unlock_all(dyn_win);
      MPI_Win_free(&dyn_win);        // collective: blocks until the last rank's
      if (dyn_counter) MPI_Free_mem(dyn_counter); // claims are done
    }
#endif

    // patch the real count into the placeholder slot at the head of the file
    if (!rf) throw std::runtime_error("Write failed on rank file: " + rank_filename);
    rf.seekp(count_pos);
    utils::writeBinary(rf, n_written);
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
      utils::readBinary(f, maxShell);
      auto count_pos = f.tellg();          // right after maxShell: position of n_tensors field
      size_t n_tensors_prev;
      utils::readBinary(f, n_tensors_prev);
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
        size_t n; utils::readBinary(rf, n); // read per-rank count
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

      // Fold in a prior run's completed rank files (SAM_EXPAND_DONE_DIR): append
      // them too, so the reused (skipped) tensors are present. Kept -- not removed.
      if (const char* dd = std::getenv("SAM_EXPAND_DONE_DIR"))
        for (const auto& e : std::filesystem::directory_iterator(dd))
        {
          if (e.path().filename().string().find(".rank_") == std::string::npos)
            continue;
          std::ifstream df(e.path(), std::ios::binary);
          if (!df) continue;
          size_t n; utils::readBinary(df, n);
          std::cerr << "[expand_exist] Rank 0: folding done file " << e.path()
                    << " (" << n << " tensors)" << std::endl;
          n_new += n;
          if (n > 0)
          {
            f << df.rdbuf();
            if (!f)
              throw std::runtime_error("Write failed folding done file " +
                                       e.path().string());
          }
        }

      size_t total = n_tensors_prev + n_new;
      std::cerr << "[expand_exist] Rank 0: updating header (maxShell=" << maxShell_
                << ", n_tensors=" << total << ")" << std::endl;
      f.seekp(maxShell_pos);
      utils::writeBinary(f, maxShell_);  // update maxShell to new (larger) value
      utils::writeBinary(f, total);      // update tensor count (lands at count_pos)
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
    utils::readBinary(s, maxShell);

    // Read number of tensors
    size_t n_tensors;
    utils::readBinary(s, n_tensors);

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

    sam::throw_missing_expansion(filename, name_, maxShell_ + valenceShell_,
                                 "TwoBodyOperatorJT");
  }

} // namespace JT
