#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <omp.h>

#include <qcx_convert.h>
#include <qcx_print.h>
using qcx_utils::print;
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>
#include <so3Coeffs/SO3_Fano.h>

#include <sam.h>

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

    // U3xU4::MultiShellTensors_phys tensors_rank;
    std::vector<std::pair<size_t, U3xU4::MultiShellTensor_phys>> tensors_rank;

    // for (size_t i = mpi_rank; i < all_na_nb_nc_nd.size(); i += mpi_size)
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
          tensors_rank.emplace_back(i, std::move(tensor));
      }
    }

    // (II) Partition the tensor list across ranks by m-scheme-memory weight,
    //      then run the expansion via OpenMP within each rank's block.
    //
    // The per-rank m-scheme cache footprint is set by the DISTINCT (a,b,c,d)
    // shell quadruples the rank touches; tensors sharing a shell-combo `i` reuse
    // the same cache entries. A strided round-robin (it += mpi_size) balances
    // tensor COUNT but scatters each combo across all ranks, bloating every
    // cache toward the full table and leaving a ~2x peak-memory skew. Instead we
    // weight each shell-combo by an m-scheme-size proxy and cut the (already
    // combo-contiguous) list into equal-weight CONTIGUOUS blocks -- one per rank,
    // never splitting a combo. This balances peak per-rank memory, keeps each
    // combo's cache reuse on a single rank, and -- because blocks are contiguous
    // and merged in rank order -- makes the output canonical (global-index
    // order, independent of rank count).
    std::vector<size_t> block_begin(mpi_size + 1, tensors_rank.size());
    block_begin[0] = 0;
    {
      // shell-degeneracy proxy: #m-scheme s.p. states in HO shell n
      auto shell_deg = [](int n) -> double { return double((n + 1) * (n + 2)); };

      // per-combo tensor counts, so a combo's weight is charged once (cache
      // reuse) but spread over its tensors for a smooth prefix sum
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

      // greedy equal-weight cuts, snapped to shell-combo boundaries
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
    }

    const size_t my_begin = block_begin[mpi_rank];
    const size_t my_end   = block_begin[mpi_rank + 1];
    const size_t my_count = my_end - my_begin;
    std::cerr << "Rank " << mpi_rank << ": processing " << my_count
              << " / " << tensors_rank.size() << " tensors "
              << "[" << my_begin << ", " << my_end << ")" << std::endl;

    // Shared cache of single-shell tensor decompositions (thread-safe)
    U3xU4::SSTDecomp_phys_db decompdb(maxShell_);

    const std::size_t cache_cap = mscheme_cache_cap();

#pragma omp parallel
{
  sam::initialize_thread();

#pragma omp for schedule(dynamic,1)
    for (size_t it = my_begin; it < my_end; ++it)
    {
      auto& [i, tensor] = tensors_rank[it];

      setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);
      HO::maybe_release_twbme_cache(cache_cap);

      size_t local_idx = it - my_begin;
      if (local_idx % 1000 == 0)
      {
#pragma omp critical
        std::cerr << "Rank " << mpi_rank << " [thread " << omp_get_thread_num()
                  << "]: " << local_idx << " / " << my_count << std::endl;
      }
    }

/*
#pragma omp critical
  {
    tensors_local.reserve(tensors_local.size() + tensors_tl.size());

    for (auto& t : tensors_tl)
      tensors_local.push_back(std::move(t));
  }
*/

  sam::finalize_thread();
}

    // (III) Free everything no longer needed

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

    // (d) flush newly computed decompositions to disk (each rank writes its own
    // contributions; finalize is internally MPI turn-based)
    decompdb.finalize();

    // (e) compact tensors_rank: keep only entries owned by this rank
    U3xU4::MultiShellTensors_phys tensors_rank_final;
    tensors_rank_final.reserve(my_count);
    for (auto& [idx, t] : tensors_rank)
    {
      if (!t.LLSSTTstrengths().empty()) tensors_rank_final.push_back(std::move(t));
      // else std::cout << "Ignore: " << t << std::endl;
    }

    tensors_rank.clear();
    tensors_rank.shrink_to_fit();

    // (IV) Each rank writes its tensors to its own temp file, then frees immediately.
    //      All ranks write concurrently — no sequential dependency here.
    std::string rank_filename = filename + ".rank_" + std::to_string(mpi_rank);
    std::cerr << "[expand_new] Rank " << mpi_rank << ": opening rank file " << rank_filename << std::endl;
    {
      std::ofstream rf(rank_filename, std::ios::binary);
      if (!rf) throw std::runtime_error("Failed to open rank file: " + rank_filename);
      size_t n = tensors_rank_final.size();
      std::cerr << "[expand_new] Rank " << mpi_rank << ": writing " << n << " tensors" << std::endl;
      qcx_utils::writeBinary(rf, n);
      for (size_t i = 0; i < tensors_rank_final.size(); ++i)
      {
        tensors_rank_final[i].writeBinary(rf);
        if ((i + 1) % 1000 == 0 || i + 1 == tensors_rank_final.size())
          std::cerr << "[expand_new] Rank " << mpi_rank << ": wrote tensor "
                    << (i + 1) << " / " << tensors_rank_final.size() << std::endl;
      }
      if (!rf) throw std::runtime_error("Write failed on rank file: " + rank_filename);
    }
    std::cerr << "[expand_new] Rank " << mpi_rank << ": rank file closed OK ("
              << tensors_rank_final.size() << " tensors)" << std::endl;

    // Free immediately — all ranks drop their tensors before anyone starts the merge,
    // so peak node memory = 1 rank's tensors (not mpi_size ranks simultaneously).
    tensors_rank_final.clear();
    tensors_rank_final.shrink_to_fit();
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

    // (VI) Final read and return on all ranks
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

    // U3xU4::MultiShellTensors_phys tensors_rank;
    std::vector<std::pair<size_t, U3xU4::MultiShellTensor_phys>> tensors_rank;

    // for (size_t i = mpi_rank; i < all_na_nb_nc_nd.size(); i += mpi_size)
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
          tensors_rank.emplace_back(i, std::move(tensor));
      }
    }

    // (II) Partition the tensor list across ranks by m-scheme-memory weight,
    //      then run the expansion via OpenMP within each rank's block.
    //
    // The per-rank m-scheme cache footprint is set by the DISTINCT (a,b,c,d)
    // shell quadruples the rank touches; tensors sharing a shell-combo `i` reuse
    // the same cache entries. A strided round-robin (it += mpi_size) balances
    // tensor COUNT but scatters each combo across all ranks, bloating every
    // cache toward the full table and leaving a ~2x peak-memory skew. Instead we
    // weight each shell-combo by an m-scheme-size proxy and cut the (already
    // combo-contiguous) list into equal-weight CONTIGUOUS blocks -- one per rank,
    // never splitting a combo. This balances peak per-rank memory, keeps each
    // combo's cache reuse on a single rank, and -- because blocks are contiguous
    // and merged in rank order -- makes the output canonical (global-index
    // order, independent of rank count).
    std::vector<size_t> block_begin(mpi_size + 1, tensors_rank.size());
    block_begin[0] = 0;
    {
      // shell-degeneracy proxy: #m-scheme s.p. states in HO shell n
      auto shell_deg = [](int n) -> double { return double((n + 1) * (n + 2)); };

      // per-combo tensor counts, so a combo's weight is charged once (cache
      // reuse) but spread over its tensors for a smooth prefix sum
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

      // greedy equal-weight cuts, snapped to shell-combo boundaries
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
    }

    const size_t my_begin = block_begin[mpi_rank];
    const size_t my_end   = block_begin[mpi_rank + 1];
    const size_t my_count = my_end - my_begin;
    std::cerr << "Rank " << mpi_rank << ": processing " << my_count
              << " / " << tensors_rank.size() << " tensors "
              << "[" << my_begin << ", " << my_end << ")" << std::endl;

    // Shared cache of single-shell tensor decompositions (thread-safe)
    U3xU4::SSTDecomp_phys_db decompdb(maxShell_);

    const std::size_t cache_cap = mscheme_cache_cap();

#pragma omp parallel
{
  sam::initialize_thread();

#pragma omp for schedule(dynamic,1)
    for (size_t it = my_begin; it < my_end; ++it)
    {
      auto& [i, tensor] = tensors_rank[it];

      setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);
      HO::maybe_release_twbme_cache(cache_cap);

      size_t local_idx = it - my_begin;
      if (local_idx % 1000 == 0)
      {
#pragma omp critical
        std::cerr << "Rank " << mpi_rank << " [thread " << omp_get_thread_num()
                  << "]: " << local_idx << " / " << my_count << std::endl;
      }
    }

/*
#pragma omp critical
  {
    tensors_local.reserve(tensors_local.size() + tensors_tl.size());

    for (auto& t : tensors_tl)
      tensors_local.push_back(std::move(t));
  }
*/

  sam::finalize_thread();
}

    // (III) Free everything no longer needed

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

    // (e) compact tensors_rank: keep only entries owned by this rank
    U3xU4::MultiShellTensors_phys tensors_rank_final;
    tensors_rank_final.reserve(my_count);
    for (auto& [idx, t] : tensors_rank)
    {
      if (!t.LLSSTTstrengths().empty()) tensors_rank_final.push_back(std::move(t));
      // else std::cout << "Ignore: " << t << std::endl;
    }
    
    tensors_rank.clear();
    tensors_rank.shrink_to_fit();

    // (IV) Each rank writes its tensors to its own temp file, then frees immediately.
    //      All ranks write concurrently — no sequential dependency here.
    std::string rank_filename = filename + ".rank_" + std::to_string(mpi_rank);
    std::cerr << "[expand_exist] Rank " << mpi_rank << ": opening rank file " << rank_filename << std::endl;
    {
      std::ofstream rf(rank_filename, std::ios::binary);
      if (!rf) throw std::runtime_error("Failed to open rank file: " + rank_filename);
      size_t n = tensors_rank_final.size();
      std::cerr << "[expand_exist] Rank " << mpi_rank << ": writing " << n << " tensors" << std::endl;
      qcx_utils::writeBinary(rf, n);
      for (size_t i = 0; i < tensors_rank_final.size(); ++i)
      {
        tensors_rank_final[i].writeBinary(rf);
        if ((i + 1) % 1000 == 0 || i + 1 == tensors_rank_final.size())
          std::cerr << "[expand_exist] Rank " << mpi_rank << ": wrote tensor "
                    << (i + 1) << " / " << tensors_rank_final.size() << std::endl;
      }
      if (!rf) throw std::runtime_error("Write failed on rank file: " + rank_filename);
    }
    std::cerr << "[expand_exist] Rank " << mpi_rank << ": rank file closed OK ("
              << tensors_rank_final.size() << " tensors)" << std::endl;

    // Free immediately — all ranks drop their tensors before anyone starts the merge,
    // so peak node memory = 1 rank's tensors (not mpi_size ranks simultaneously).
    tensors_rank_final.clear();
    tensors_rank_final.shrink_to_fit();
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
    // Final read and return on all ranks
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
