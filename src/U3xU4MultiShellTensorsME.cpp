#include <algorithm>
#include <array>
#include <map>
#include <ostream>
#include <unordered_map>

#include <boost/container_hash/hash.hpp>

#include <qcx_matrix.h>
#include <qcx_ND.h>
#include <qcx_print.h>
using qcx_utils::print;

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Fano.h>

#include <U3SO3xU3SO3.h>
#include <U4SU2SU2xU4SU2SU2.h>

#include <U3xU4ManyNucleonBasis.h>
#include <U3xU4MultiShellTensors.h>

#include <sam.h>

#include <U3xU4MultiShellTensorsME_helpers.h>

namespace U3xU4
{
  // Custom sparse format defined in qcx_matrix.h with OpenMP
  template <typename TensorType>
  qcx_utils::CSRMatrix MatrixElements_CSR(
                  const std::vector<TensorType>& tensors, 
                  const U3xU4::ManyNucleonBasis& basis,
                  const U3xU4::Table_gelfand& gelfand_labels)
  {
    // for (auto tensor : tensors) std::cout << tensor << std::endl;

    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    // basis info
    const auto irreps = basis.irreps();
    const auto dim = basis.dim();
    const auto& labels = basis.decomposition();
    const auto JJ = basis.JJ();
    const auto MMJ = JJ;
    const auto MMT = basis.MMT();
    // tensors info
    const auto tensors_count = tensors.size();
    // all tensors have the same J, MJ, MT, but T may vary
    const auto JJ_opt = tensors[0].JJ();
    const auto MMJ_opt = tensors[0].MMJ();
    assert(MMJ_opt == 0);
    const auto MMT_opt = tensors[0].MMT();

    const auto cgcJ = cgcs.get({JJ, JJ_opt, JJ, MMJ, MMJ_opt, MMJ});

    if (tensors_count == 0 || std::fabs(cgcJ) == 0.0)
    {
      // Eigen::SparseMatrix<double> mat(dim, dim);
      qcx_utils::CSRMatrix mat(dim);
      return mat;
    }

    // Get MPI info
#ifdef USE_MPI
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

    // PREPARATION PHASE --- BEGIN --- Identical across all ranks

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<size_t>> connected_tensors;
    std::vector<std::tuple<size_t, size_t, size_t>> connected_irreps;

    // (I) flatten the outer two loops into a 1D vector of pairs
    std::vector<std::pair<size_t,size_t>> dist_pairs;
    for (size_t b = 0; b < basis.n_distributions(); b++)
    {
#ifdef EXPLOIT_SYMMETRIC
      for (size_t k = 0; k <= b; k++)  // k <= b replaces the continue
#else
      for (size_t k = 0; k < basis.n_distributions(); k++)
#endif
      {
        dist_pairs.push_back({b, k});
      }
    }

    // (II) define a result struct for one (bra_gamma, ket_gamma) hit
    struct GammaResult {
      std::vector<size_t>                    tensors2;
      std::vector<std::pair<size_t,size_t>>  irrep_pairs; // (bra_id, ket_id)
      size_t                                 cost;
    };
    std::vector<std::vector<GammaResult>> all_results(dist_pairs.size());
    //          ^                         ^
    //          one slot per dist pair    each slot holds all GammaResults
    //                                    found for that pair

    // (III) parallel for over dist_pairs 
#pragma omp parallel for schedule(dynamic)
    // for (size_t bra_dist_id = 0; bra_dist_id < basis.n_distributions(); bra_dist_id++)
    for (size_t p = 0; p < dist_pairs.size(); p++)
    {
      auto [bra_dist_id, ket_dist_id] = dist_pairs[p];

      auto bra_dist = basis.distribution(bra_dist_id);
      const auto& bra_gamma_irreps_ids = basis.group_dists[bra_dist_id];
      auto bra_dist_to_gamma = map_by_positive_order(bra_dist);

      // print("\nbra_dist_id: {}\t", bra_dist_id);
      // print("bra_dist_to_gamma: {}\n", bra_dist_to_gamma)

      // for (size_t ket_dist_id = 0; ket_dist_id < basis.n_distributions(); ket_dist_id++)
      {
// #ifdef EXPLOIT_SYMMETRIC
//         if (bra_dist_id < ket_dist_id) continue;
// #endif

        auto ket_dist = basis.distribution(ket_dist_id);
        const auto& ket_gamma_irreps_ids = basis.group_dists[ket_dist_id];
        auto ket_dist_to_gamma = map_by_positive_order(ket_dist);

        // print("ket_dist_id: {} \t", ket_dist_id);
        // print("ket_dist_to_gamma: {}\n", ket_dist_to_gamma);

        // Predicate to check if particles are conserved in all shells
        auto all_shells_particles = [&](const auto& tensor)
        {
          // in a shell:
          // ket_dist[n] - G.num_ats() + G.num_ads()) == bra_dist[n] &&
          // ket_dist[n] >= G.num_ats()

          std::vector<int> delta(ket_dist.size(), 0);

          // Accumulate net changes from Gammas
          for (const auto& G : tensor.Gammas())
          {
            auto n = G.shell();
            if (n >= ket_dist.size()) return false;

            delta[n] = G.num_ads() - G.num_ats();

            // Early failure: destroying more than available
            if (ket_dist[n] < G.num_ats()) return false;
          }

          // Check all shells (including untouched ones)
          for (size_t n = 0; n < ket_dist.size(); ++n)
            if (ket_dist[n] + delta[n] != bra_dist[n]) return false;

          return true;
        };

        std::vector<size_t> tensors1;
        // tensors1.reserve(tensors.size());
        tensors1.reserve(100);
        for (size_t i = 0; i < tensors.size(); i++)
          if (all_shells_particles(tensors[i])) tensors1.push_back(i);
        if (tensors1.empty()) continue;
        // print("tensors1: {}\n", tensors1.size());

        for (const auto& [bra_gamma_id, bra_irrep_ids] : bra_gamma_irreps_ids)
        {
          auto bra_gamma = basis.gamma(bra_gamma_id);
          // print("bra_gamma_id: {}\n", bra_gamma_id);

          for (const auto& [ket_gamma_id, ket_irrep_ids] : ket_gamma_irreps_ids)
          {
#ifdef EXPLOIT_SYMMETRIC
            if (bra_gamma_id < ket_gamma_id) continue;
#endif
            auto ket_gamma = basis.gamma(ket_gamma_id);
            // print("ket_gamma_id: {}\n", ket_gamma_id);

            // Predicate to check if single-shell coupling is allowed
            auto single_shell_coupling = [&](const auto& G)
            {
              auto n = G.shell();

              su3::irrep i3, f3;
              u4::irrep i4, f4;

              if (bra_dist_to_gamma[n] >= 0)
              {
                f3 = bra_gamma[bra_dist_to_gamma[n]].su3;
                f4 = bra_gamma[bra_dist_to_gamma[n]].u4;
              }

              if (ket_dist_to_gamma[n] >= 0)
              {
                i3 = ket_gamma[ket_dist_to_gamma[n]].su3;
                i4 = ket_gamma[ket_dist_to_gamma[n]].u4;
              }

              return u3::Rho(i3, G.u3(), f3) > 0 && u4::Eta(i4, G.u4(), f4) > 0;
            };
            // Predicate to check if all single-shell coupings are allowed
            auto all_shells_coupling = [&](const auto& tensor)
            {
              return std::all_of(tensor.Gammas().begin(),
                                 tensor.Gammas().end(),
                                 single_shell_coupling);
            };

            std::vector<size_t> tensors2;
            // tensors2.reserve(tensors1.size());
            tensors2.reserve(100);
            for (const auto& i : tensors1)
              if (all_shells_coupling(tensors[i])) tensors2.push_back(i);
            if (tensors2.empty()) continue;
            // print("tensors2: {}\n", tensors2.size());


            // instead of pushing to connected_irreps/connected_tensors,
            // collect the irrep pairs locally
            std::vector<std::pair<size_t,size_t>> irrep_pairs;
            size_t cost = 0;
            for (const auto& bra_id : bra_irrep_ids)
              for (const auto& ket_id : ket_irrep_ids)
              {
#ifdef EXPLOIT_SYMMETRIC
                if (bra_id < ket_id) continue;
#endif
                // connected_irreps.emplace_back(bra_id, ket_id, connected_tensors.size());

                irrep_pairs.push_back({bra_id, ket_id});

                // compute cost in parallel alongside irrep_pairs
                // size_t bra_amax = irreps[bra_id].alphamax();
                const auto& bra_labels = labels[bra_id];
                size_t bra_dim = bra_labels.back()[5] +
                                 bra_labels.back()[0] * bra_labels.back()[2];

                // size_t ket_amax = irreps[ket_id].alphamax();
                const auto& ket_labels = labels[ket_id];
                size_t ket_dim = ket_labels.back()[5] +
                                 ket_labels.back()[0] * ket_labels.back()[2];

                // cost += bra_amax * bra_dim * ket_amax * ket_dim;
                cost += bra_dim * ket_dim;
              }

            // connected_tensors.push_back(std::move(tensors2));

            cost += tensors2.size() * irrep_pairs.size();

            // write only to dist_pair's own slot — no race
            all_results[p].push_back({std::move(tensors2), std::move(irrep_pairs), cost});
          }
        }
      }
    }

    // (IV) serial assembly

    // Build a flat sorted list of work items before distributing
    struct WorkItem { size_t p_idx, r_idx, cost; };
    std::vector<WorkItem> work_items;
    for (size_t p = 0; p < dist_pairs.size(); p++)
      for (size_t r = 0; r < all_results[p].size(); r++)
        work_items.push_back({p, r, all_results[p][r].cost});

    dist_pairs.clear();
    dist_pairs.shrink_to_fit();

    // LPT: largest work items first, so the block round-robin below hands the
    // heaviest blocks out first (rough cost balancing across ranks).
    std::sort(work_items.begin(), work_items.end(),
              [](const auto& a, const auto& b) { return a.cost > b.cost; });

    // Terminal (bra_irr, ket_irr) of each basis irrep. Used to (a) distribute
    // whole terminal-irrep BLOCKS across MPI ranks so each block's coupling
    // coefficients are computed on a single rank (cf. lsu3shell block
    // distribution), and (b) sort the per-rank work list into contiguous
    // blocks below. Reference stays valid; std::array key is cheap to compare.
    std::vector<std::array<int16_t, 5>> term(irreps.size());
    for (size_t id = 0; id < term.size(); id++)
    {
      auto ir = irreps[id];
      const U3xU4::irrep& t = ir.omega.empty() ? ir.gamma[0].u3u4()
                                               : ir.omega.back().u3u4();
      term[id] = { (int16_t)t.lam(), (int16_t)t.mu(),
                   (int16_t)t.P1(), (int16_t)t.P2(), (int16_t)t.P3() };
    }
    auto block_key = [&](size_t bra_id, size_t ket_id)
    {
      std::array<int16_t, 10> k;
      const auto& b = term[bra_id]; const auto& q = term[ket_id];
      for (int z = 0; z < 5; z++) { k[z] = b[z]; k[z + 5] = q[z]; }
      return k;
    };
#ifdef USE_MPI
    // Assign each terminal-irrep block to one rank on first encounter
    // (round-robin). Deterministic across ranks because the work_item order is
    // identical everywhere -> every block lands on exactly one rank.
    std::map<std::array<int16_t, 10>, int> block_rank;
    int next_block_rank = 0;
#endif

    for (auto& item : work_items)
    // for (size_t p = 0; p < dist_pairs.size(); p++)
    {
      auto& all_result = all_results[item.p_idx][item.r_idx];

      // for (auto& all_result : all_results[p])
      {
        // connected_tensors.size() is now correct because we're
        // doing this sequentially in the original loop order
        // size_t tensors2_id = connected_tensors.size();

        bool tensors2_added = false;
        size_t local_tensors2_id = 0;

        for (auto& [bra_id, ket_id] : all_result.irrep_pairs)
        {
#ifdef USE_MPI
          // keep this pair only if its terminal-irrep block is owned by this
          // rank (whole block on one rank -> coefficients computed once)
          auto bk = block_key(bra_id, ket_id);
          auto it = block_rank.find(bk);
          int owner;
          if (it == block_rank.end())
          {
            owner = next_block_rank;
            block_rank.emplace(bk, owner);
            next_block_rank = (next_block_rank + 1) % mpi_size;
          }
          else owner = it->second;
          if (owner == mpi_rank)
#endif
          {
            if (!tensors2_added)
            {
              local_tensors2_id = connected_tensors.size();
              connected_tensors.push_back(std::move(all_result.tensors2));
              tensors2_added = true;
            }
            connected_irreps.emplace_back(bra_id, ket_id, local_tensors2_id);
          }
        }
      }
    }

    work_items.clear();
    work_items.shrink_to_fit();
    all_results.clear();
    all_results.shrink_to_fit();

    // --- Tensor reordering (cf. lsu3shell TENSOR_LABELS ordering) -------------
    // Order the tensors within each connected set by terminal irrep so tensors
    // sharing an SU(3)xU(4) character are processed consecutively -- the
    // block-local coupling-coefficient entries for that tensor_irr then stay
    // hot across the tensor loop. (Only affects locality, not results; the
    // ShareGammas tensor type has no single terminal irrep, so skip it there.)
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

    // --- Reorder the work list by terminal (bra_irr, ket_irr) ----------------
    // Group connected pairs that share terminal irreps so each block's coupling
    // coefficients are computed once and reused with locality (see the
    // cached_U3SO3/cached_U4SU2SU2 block tables above). block_id_of_ci[ci]
    // marks the block of the ci-th (sorted) pair; the compute loop publishes it
    // to g_asm_block so the per-thread coefficient tables drop stale blocks.
    std::vector<uint32_t> block_id_of_ci;
    {
      // `term` was computed above (also used for the MPI block distribution).
      std::sort(connected_irreps.begin(), connected_irreps.end(),
        [&](const auto& x, const auto& y)
        {
          const auto& bx = term[std::get<0>(x)]; const auto& kx = term[std::get<1>(x)];
          const auto& by = term[std::get<0>(y)]; const auto& ky = term[std::get<1>(y)];
          if (bx != by) return bx < by;
          return kx < ky;
        });

      block_id_of_ci.resize(connected_irreps.size());
      uint32_t blk = 0;
      for (size_t ci = 0; ci < connected_irreps.size(); ci++)
      {
        if (ci > 0)
        {
          const auto& a = connected_irreps[ci];
          const auto& p = connected_irreps[ci - 1];
          if (term[std::get<0>(a)] != term[std::get<0>(p)] ||
              term[std::get<1>(a)] != term[std::get<1>(p)])
            blk++;
        }
        block_id_of_ci[ci] = blk;
      }
    }

    // --- Row-owner groups for direct-write assembly ---------------------------
    // Every kept pair has bra_id >= ket_id, and add_symmetric routes each entry
    // to row max(i, j), so all rows a pair writes lie in bra_id's row range.
    // Each bra_id has exactly one terminal irrep, and the work list is sorted
    // by (term_bra, term_ket): pairs with different term_bra therefore write
    // DISJOINT row sets, and the term_bra groups are contiguous ci ranges.
    // One thread per group can write straight into the global matrix --
    // race-free, with no thread-local matrices and no serial merge phase.
    // Groups are visited heaviest-first (by pair count) so dynamic scheduling
    // balances the tail.
    std::vector<size_t> group_start;
    for (size_t ci = 0; ci < connected_irreps.size(); ci++)
      if (ci == 0 ||
          term[std::get<0>(connected_irreps[ci])] !=
          term[std::get<0>(connected_irreps[ci - 1])])
        group_start.push_back(ci);
    group_start.push_back(connected_irreps.size());

    const size_t n_groups = group_start.empty() ? 0 : group_start.size() - 1;
    std::vector<size_t> group_order(n_groups);
    for (size_t g = 0; g < n_groups; g++) group_order[g] = g;
    std::sort(group_order.begin(), group_order.end(), [&](size_t a, size_t b)
    { return (group_start[a + 1] - group_start[a]) >
             (group_start[b + 1] - group_start[b]); });

#ifdef DIAGNOSE_MEMORY
    print_rss("after prep phase", mpi_rank, mpi_size);
#endif

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
#ifdef USE_MPI
    std::cerr << "Matrix elements preparation time: " << elapsed.count()/60 
              << "min on rank " << mpi_rank << std::endl;
    std::cerr << "Number of connected irreps on rank " << mpi_rank
              << ": " << connected_irreps.size() << std::endl;
#else
    std::cout << "Matrix elements preparation time: " << elapsed.count()/60 << "min" << std::endl;
    std::cout << "Number of connected irreps: " << connected_irreps.size() << std::endl;
#endif
    

    // PREPARATION PHASE --- END
#ifdef USE_MPI
    std::cerr << "Start computing matrix on rank " << mpi_rank << std::endl;
#else
    std::cout << "Start computing matrix" << std::endl;
#endif

    qcx_utils::CSRMatrix global(dim);
#ifdef EXPLOIT_SYMMETRIC
    // Only the lower triangle is stored; mark global BEFORE the parallel
    // region (the compute loop uses add() with explicit max/min routing so no
    // thread ever writes this flag concurrently).
    global.symmetric_ = true;
#endif

    // RME database
    U3xU4::RME_db RMEdb(basis.maxShell());
#ifdef DIAGNOSE_MEMORY
    print_rss("after RMEdb init", mpi_rank, mpi_size);
#endif

#ifdef TIME_ASSEMBLY
    auto _asm_compute0 = std::chrono::high_resolution_clock::now();
    for (auto& _p : g_asm_prof) _p = AsmProf{};
#endif

#ifdef USE_OPENMP
#pragma omp parallel
{
    sam::initialize_thread();
#endif

    // Dynamic loop over term_bra owner groups (heaviest first): each group
    // owns a disjoint set of rows, so threads write straight into `global`
    // with no thread-local matrices and no merge. Within a group the pairs
    // keep their (term_bra, term_ket) block order, so the block-local
    // coupling-coefficient tables behave exactly as before (see g_asm_block).
#pragma omp for schedule(dynamic, 1)
    for (size_t gi = 0; gi < n_groups; gi++)
    {
    const size_t grp = group_order[gi];
    for (size_t ci = group_start[grp]; ci < group_start[grp + 1]; ci++)
    {
      auto bra_id = std::get<0>(connected_irreps[ci]);
      auto ket_id = std::get<1>(connected_irreps[ci]);
      auto connected_tensors_id = std::get<2>(connected_irreps[ci]);

      // publish this pair's terminal-irrep block so the per-thread coupling-
      // coefficient tables reuse within the block and drop it on change
      g_asm_block = block_id_of_ci[ci];

      const auto& tensors2 = connected_tensors[connected_tensors_id];
      {
        {
          {
            // for (const auto& bra_id : bra_irrep_ids)
            {
              const auto& bra = irreps[bra_id];

              // get alphamax
              size_t bra_amax = bra.alphamax();

              // get offset
              auto bra_offset = irreps.offset(bra_id);

              // get labels
              const auto& bra_labels = labels[bra_id];
              size_t bra_dim = bra_labels.back()[5] +
                              bra_labels.back()[0] * bra_labels.back()[2];
/*
              print("bra id {} : ", bra_id);
              print(" gamma = {}", bra.gamma);
              print(" ; omega = {}", bra.omega);
              print(" ; kappamax LL zetamax SS TT offset = {}\n", bra_labels);
*/
              // for (const auto& ket_id : ket_irrep_ids)
              {
#ifdef EXPLOIT_SYMMETRIC
                if (bra_id < ket_id) continue;
#endif
                const auto& ket = irreps[ket_id];

                // get alphamax
                size_t ket_amax = ket.alphamax();

                // get offset
                auto ket_offset = irreps.offset(ket_id);

                // get labels
                const auto& ket_labels = labels[ket_id];
                size_t ket_dim = ket_labels.back()[5] +
                              ket_labels.back()[0] * ket_labels.back()[2];
/*
                print("ket id {} : ", ket_id);
                print(" gamma = {}", ket.gamma);
                print(" ; omega = {}", ket.omega);
                print(" ; kappamax LL zetamax SS TT offset = {}\n", ket_labels);
*/
                // unify bra and ket irreps
                U3xU4::IrrepUnifier iu;
                const auto& [ubra, uket] = iu.unify(bra, ket);
                // extract the last irreps
                const auto& ket_irr = (ubra.omega.empty())
                             ? uket.gamma[0].u3u4() : uket.omega.back().u3u4();
                const auto& bra_irr = (ubra.omega.empty())
                             ? ubra.gamma[0].u3u4() : ubra.omega.back().u3u4();

                // MEs block for this irrep pair, stored row-major in a
                // thread-local flat buffer that only grows -- no per-pair
                // allocation and no Eigen dense matrix. Element (i, j) is at
                // MEs[i * C + j]. assign() reuses the buffer when R*C fits the
                // existing capacity and zeroes the used region.
                const size_t R = bra_amax * bra_dim;
                const size_t C = ket_amax * ket_dim;
                thread_local std::vector<double> MEs;
                MEs.assign(R * C, 0.0);

                for (const auto& tensor_id : tensors2)
                {
                  const auto& tensor = tensors[tensor_id];

                  // std::cout << "Before unified: " << tensor << std::endl;
                  ASM_TIC(_t_unify);
                  auto uT = U3xU4::unify_tensor(tensor, ubra, uket);
                  ASM_TOC(unify, _t_unify);
                  if (!uT.has_value()) continue;
                  // std::cout << "After unified: " << tensor << std::endl;

                  const auto& Gammas = uT->Gammas();
                  const auto& Omegas_LLSSTTstrengths = uT->Omegas_LLSSTTstrengths();

                  ASM_TIC(_t_rme);
                  auto RMEs = uT->RME(ubra, uket, gelfand_labels, RMEdb);
                  ASM_TOC(rme, _t_rme);
                  assert(RMEs.size() == Omegas_LLSSTTstrengths.size());

#ifdef TIME_ASSEMBLY
                  // times the rme_id loop (coeff lookups + accumulation);
                  // RAII stops at the end of the enclosing tensor-loop body
                  AsmScope _sw(g_asm_prof[ASM_TID].work);
#endif
                  for (size_t rme_id = 0; rme_id < RMEs.size(); rme_id++)
                  {
                    const auto& rmes = RMEs[rme_id];
                    const auto& Omegas = Omegas_LLSSTTstrengths[rme_id].first;
                    const auto& LLSSTTstrengths = Omegas_LLSSTTstrengths[rme_id].second;

                    const auto& irr = (Omegas.empty())
                                    ? Gammas.back().u3u4() : Omegas.back().u3u4();

                    size_t tensor_amax = U3xU4::alphamax(Gammas, Omegas);
                    auto rhomax = rmes.rhomax(), etamax = rmes.etamax();

                    // going through subgroup labels of the tensor
                    for (const auto& [kmax, LL, zmax, SS, TT, strengths]
                                                        : LLSSTTstrengths)
                    {
                      // going through bra labels
                      for (const auto& bra_label : bra_labels)
                      {
                        const auto& bra_kmax = bra_label[0];
                        const auto& bra_LL = bra_label[1];
                        const auto& bra_zmax = bra_label[2];
                        const auto& bra_SS = bra_label[3];
                        const auto& bra_TT = bra_label[4];
                        const auto& bra_label_offset = bra_label[5];

                        // going through ket labels
                        for (const auto& ket_label : ket_labels)
                        {
                          const auto& ket_kmax = ket_label[0];
                          const auto& ket_LL = ket_label[1];
                          const auto& ket_zmax = ket_label[2];
                          const auto& ket_SS = ket_label[3];
                          const auto& ket_TT = ket_label[4];
                          const auto& ket_label_offset = ket_label[5];

                          // std::cout << "Start to calculate coupling coeffs" << std::endl;
                          auto cgcT = cgcs.get({ket_TT, TT, bra_TT, MMT, MMT_opt, MMT});
                          auto cgcU2 = cgcJ * cgcT * racah9.get({ket_LL, ket_SS, JJ,
                                                                 LL, SS, JJ_opt,
                                                                 bra_LL, bra_SS, JJ});
                          if (std::fabs(cgcU2) < 1e-12) continue;

                          const auto& cgcU3 = cached_U3SO3(wigU3SO3db,
                            ket_irr.lam(), ket_irr.mu(), ket_LL / 2,
                            irr.lam(), irr.mu(), LL / 2,
                            bra_irr.lam(), bra_irr.mu(), bra_LL / 2);
                          if (cgcU3.empty()) continue;

                          const auto& cgcU4 = cached_U4SU2SU2(wigU4SU2SU2db,
                            ket_irr.P1(), ket_irr.P2(), ket_irr.P3(), ket_SS, ket_TT,
                            irr.P1(), irr.P2(), irr.P3(), SS, TT,
                            bra_irr.P1(), bra_irr.P2(), bra_irr.P3(), bra_SS, bra_TT);
                          if (cgcU4.empty()) continue;
                          // std::cout << "Finish calculating coupling coeffs" << std::endl;

                          // std::cout << "Start to calculate FMEs" << std::endl;
                          for (size_t bra_a = 0; bra_a < bra_amax; bra_a++)
                          {
                            for (size_t bra_k = 0; bra_k < bra_kmax; bra_k++)
                            {
                              for (size_t bra_z = 0; bra_z < bra_zmax; bra_z++)
                              {
                                // bra index in the matrix
                                size_t i =  bra_a * bra_dim
                                    + bra_label_offset + (bra_k * bra_zmax + bra_z);

                                for (size_t ket_a = 0; ket_a < ket_amax; ket_a++)
                                {
                                  for (size_t ket_k = 0; ket_k < ket_kmax; ket_k++)
                                  {
                                    for (size_t ket_z = 0; ket_z < ket_zmax; ket_z++)
                                    {
                                      // ket index in the matrix
                                      size_t j = ket_a * ket_dim
                                        + ket_label_offset + (ket_k * ket_zmax + ket_z);
#ifdef EXPLOIT_SYMMETRIC
                                      if (bra_offset == ket_offset && i < j) continue;
#endif
                                      // std::cout << "i:" << i << " j:" << j << std::endl;

                                      // Accumulate this (i, j) leaf in a register
                                      // and write MEs once, instead of a memory
                                      // read-modify-write per innermost iteration.
                                      double acc = 0.0;
                                      for (size_t tensor_a = 0; tensor_a < tensor_amax; tensor_a++)
                                      {
                                        for (size_t k = 0; k < kmax; k++)
                                        {
                                          for (size_t z = 0; z < zmax; z++)
                                          {
                                            const double sc = cgcU2 *
                                                (*strengths)[(tensor_a * kmax + k) * zmax + z];

                                            // multiplicities of the Wigner-Eckart
                                            // theorem; hoist cgcU3 (invariant to
                                            // eta) and skip structurally-zero coeffs
                                            for (size_t rho = 0; rho < rhomax; rho++)
                                            {
                                              const double c3 = cgcU3(ket_k, k, bra_k, rho);
                                              if (c3 == 0.0) continue;
                                              const double sc3 = sc * c3;

                                              for (size_t eta = 0; eta < etamax; eta++)
                                                acc += sc3
                                                     * cgcU4(ket_z, z, bra_z, eta)
                                                     * rmes(bra_a, ket_a, tensor_a, rho, eta);
                                            }
                                          }
                                        }
                                      }
                                      MEs[i * C + j] += acc;
                                    }
                                  }
                                }
                              }
                            }
                          }
                          // std::cout << "Finish calculating FMEs" << std::endl;
                        }
                      }
                    }
                  }
                }

                // std::cout << "Start to fill sparse matrix" << std::endl;
                // Direct write into global: this thread owns every row of the
                // current term_bra group (see the owner-group construction),
                // so no other thread can touch these rows. The lower-triangle
                // routing (row = max, col = min) is done here explicitly with
                // add() -- add_symmetric() would redundantly store the
                // symmetric_ flag from every thread.
                for (size_t i = 0; i < R; i++)
                {
                  Eigen::Index i_ = i + bra_offset;
		              assert(i_ >= 0 && i_ < dim);

                  for (size_t j = 0; j < C; j++)
                  {
                    Eigen::Index j_ = j + ket_offset;
		                assert(j_ >= 0 && j_ < dim);

                    double v = MEs[i * C + j];
#ifdef EXPLOIT_SYMMETRIC
                    if (std::fabs(v) > 1e-8)
                      global.add(std::max(i_, j_), std::min(i_, j_), v);
#else
                    if (std::fabs(v) > 1e-8) global.add(i_, j_, v);
#endif
                  }
                }
                // std::cout << "Finish filling sparse matrix; " << std::endl;
              }
            }
          }
        }
      }
    }
    } // group loop

#ifdef USE_OPENMP
    sam::finalize_thread();
} // pragma omp parallel
#endif

#ifdef TIME_ASSEMBLY
    {
      auto _now = std::chrono::high_resolution_clock::now();
      std::cerr << "[assembly] compute (direct-write, no merge): "
                << std::chrono::duration<double>(_now - _asm_compute0).count()
                << " s" << std::endl;
      double _u = 0, _r = 0, _w = 0;
      for (const auto& _s : g_asm_prof)
        { _u += _s.unify; _r += _s.rme; _w += _s.work; }
      std::cerr << "[assembly]   (thread-summed) unify=" << _u
                << "  rme=" << _r << "  work(coeff+accum)=" << _w
                << " s" << std::endl;
    }
    auto _asm_finalize0 = std::chrono::high_resolution_clock::now();
#endif

#ifdef DIAGNOSE_MEMORY
    print_rss("after OMP block (pre-finalize)", mpi_rank, mpi_size);
#endif
    
    sam::clear();
    RMEdb.finalize();
    connected_irreps.clear();
    connected_irreps.shrink_to_fit();
    connected_tensors.clear();
    connected_tensors.shrink_to_fit();

#ifdef DIAGNOSE_MEMORY
    print_rss("after sam::clear + RMEdb::finalize", mpi_rank, mpi_size);
#endif

#ifdef USE_MPI
    std::cerr << "Finalize global CSR on rank " << mpi_rank << std::endl;
    global.finalize();
    std::cerr << "Finish computing " << global.size() << " matrix elements on rank " 
              << mpi_rank << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
#else
    std::cerr << "Finalize global CSR" << std::endl;
    global.finalize();
    std::cout << "Finish computing " << global.size() << " matrix elements" << std::endl;
#endif

#ifdef TIME_ASSEMBLY
    {
      auto _now = std::chrono::high_resolution_clock::now();
      std::cerr << "[assembly] teardown+finalize: "
                << std::chrono::duration<double>(_now - _asm_finalize0).count()
                << " s" << std::endl;
    }
#endif

#ifdef DIAGNOSE_MEMORY
    print_rss("after CSR finalize", mpi_rank, mpi_size);
#endif

    return global;
  }

  template qcx_utils::CSRMatrix MatrixElements_CSR(
                  const MultiShellTensors_phys& tensors, 
                  const U3xU4::ManyNucleonBasis& basis,
                  const U3xU4::Table_gelfand& gelfand_labels);

  template qcx_utils::CSRMatrix MatrixElements_CSR(
                  const MultiShellTensors_ShareGammas_phys& tensors, 
                  const U3xU4::ManyNucleonBasis& basis,
                  const U3xU4::Table_gelfand& gelfand_labels);
  /////////////////// COMPUTING RECTANGULAR MATRIX (CSR) ///////////////////

  // CSR analogue of the rectangular MatrixElements_Eigen above. Builds the
  // two-basis transition matrix (bras_dim x kets_dim, non-symmetric) into a
  // memory-lean CSRMatrix instead of an Eigen sparse matrix: entries stream
  // straight into per-thread row storage with no triplet doubling, and each
  // terminal-irrep block is assigned to a single MPI rank so no rank ever
  // holds the whole matrix. Unlike the Eigen rectangular builder, it uses the
  // block-local coupling-coefficient caches (cached_U3SO3/cached_U4SU2SU2 via
  // g_asm_block), matching the square CSR path.
  template <typename TensorType>
  qcx_utils::CSRMatrix MatrixElements_CSR(
                  const std::vector<TensorType>& tensors,
                  const U3xU4::ManyNucleonBasis& bras,
                  const U3xU4::ManyNucleonBasis& kets,
                  const U3xU4::Table_gelfand& gelfand_labels)
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    // bra basis info
    auto bras_irreps = bras.irreps();
    auto bras_dim = bras.dim();
    const auto& bras_labels = bras.decomposition();
    auto bras_JJ = bras.JJ();
    auto bras_MMJ = bras_JJ;
    auto bras_MMT = bras.MMT();

    // ket basis info
    auto kets_irreps = kets.irreps();
    auto kets_dim = kets.dim();
    const auto& kets_labels = kets.decomposition();
    auto kets_JJ = kets.JJ();
    auto kets_MMJ = kets_JJ;
    auto kets_MMT = kets.MMT();

    // tensors info
    auto tensors_count = tensors.size();
    // all tensors have the same J, MJ, MT, but T may vary
    auto JJ_opt = tensors[0].JJ();
    auto MMJ_opt = tensors[0].MMJ();
    assert(MMJ_opt == 0);
    auto MMT_opt = tensors[0].MMT();

    auto MMJ = std::min(bras_MMJ, kets_MMJ);
    auto cgcJ = cgcs.get({kets_JJ, JJ_opt, bras_JJ, MMJ, MMJ_opt, MMJ});

    if (tensors_count == 0 || std::fabs(cgcJ) == 0.0)
      return qcx_utils::CSRMatrix(bras_dim, kets_dim);

    // Get MPI info
#ifdef USE_MPI
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

    // PREPARATION PHASE --- BEGIN --- Identical across all ranks

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<size_t>> connected_tensors;
    std::vector<std::tuple<size_t, size_t, size_t>> connected_irreps;

    // (I) flatten the outer two loops into a 1D vector of (bra_dist, ket_dist)
    std::vector<std::pair<size_t,size_t>> dist_pairs;
    for (size_t b = 0; b < bras.n_distributions(); b++)
      for (size_t k = 0; k < kets.n_distributions(); k++)
        dist_pairs.push_back({b, k});

    // (II) result struct for one (bra_gamma, ket_gamma) hit
    struct GammaResult {
      std::vector<size_t>                    tensors2;
      std::vector<std::pair<size_t,size_t>>  irrep_pairs; // (bra_id, ket_id)
    };
    std::vector<std::vector<GammaResult>> all_results(dist_pairs.size());

    // (III) parallel for over dist_pairs
#pragma omp parallel for schedule(dynamic)
    for (size_t p = 0; p < dist_pairs.size(); p++)
    {
      auto [bra_dist_id, ket_dist_id] = dist_pairs[p];

      auto bra_dist = bras.distribution(bra_dist_id);
      const auto& bra_gamma_irreps_ids = bras.group_dists[bra_dist_id];
      auto bra_dist_to_gamma = map_by_positive_order(bra_dist);

      {
        auto ket_dist = kets.distribution(ket_dist_id);
        const auto& ket_gamma_irreps_ids = kets.group_dists[ket_dist_id];
        auto ket_dist_to_gamma = map_by_positive_order(ket_dist);

        // Predicate to check if particles are conserved in all shells
        auto all_shells_particles = [&](const auto& tensor)
        {
          size_t max_shells = std::max(ket_dist.size(), bra_dist.size());
          std::vector<int> delta(max_shells, 0);

          for (const auto& G : tensor.Gammas())
          {
            size_t n = G.shell();
            if (n >= max_shells) return false;

            delta[n] += G.num_ads() - G.num_ats();

            int ket_occ = (n < ket_dist.size()) ? ket_dist[n] : 0;
            if (ket_occ < G.num_ats()) return false;
          }

          for (size_t n = 0; n < max_shells; ++n)
          {
            int ket_occ = (n < ket_dist.size()) ? ket_dist[n] : 0;
            int bra_occ = (n < bra_dist.size()) ? bra_dist[n] : 0;

            if (ket_occ + delta[n] != bra_occ) return false;
          }

          return true;
        };

        std::vector<size_t> tensors1;
        tensors1.reserve(100);
        for (size_t i = 0; i < tensors.size(); i++)
          if (all_shells_particles(tensors[i])) tensors1.push_back(i);
        if (tensors1.empty()) continue;

        for (const auto& [bra_gamma_id, bra_irrep_ids] : bra_gamma_irreps_ids)
        {
          auto bra_gamma = bras.gamma(bra_gamma_id);

          for (const auto& [ket_gamma_id, ket_irrep_ids] : ket_gamma_irreps_ids)
          {
            auto ket_gamma = kets.gamma(ket_gamma_id);

            // Predicate to check if single-shell coupling is allowed
            auto single_shell_coupling = [&](const auto& G)
            {
              auto n = G.shell();

              su3::irrep i3, f3;
              u4::irrep i4, f4;

              if (n < bra_dist_to_gamma.size() && bra_dist_to_gamma[n] >= 0)
              {
                f3 = bra_gamma[bra_dist_to_gamma[n]].su3;
                f4 = bra_gamma[bra_dist_to_gamma[n]].u4;
              }

              if (n < ket_dist_to_gamma.size() && ket_dist_to_gamma[n] >= 0)
              {
                i3 = ket_gamma[ket_dist_to_gamma[n]].su3;
                i4 = ket_gamma[ket_dist_to_gamma[n]].u4;
              }

              return u3::Rho(i3, G.u3(), f3) > 0 && u4::Eta(i4, G.u4(), f4) > 0;
            };
            auto all_shells_coupling = [&](const auto& tensor)
            {
              return std::all_of(tensor.Gammas().begin(),
                                 tensor.Gammas().end(),
                                 single_shell_coupling);
            };

            std::vector<size_t> tensors2;
            tensors2.reserve(100);
            for (const auto& i : tensors1)
              if (all_shells_coupling(tensors[i])) tensors2.push_back(i);
            if (tensors2.empty()) continue;

            std::vector<std::pair<size_t,size_t>> irrep_pairs;
            for (const auto& bra_id : bra_irrep_ids)
              for (const auto& ket_id : ket_irrep_ids)
                irrep_pairs.push_back({bra_id, ket_id});

            all_results[p].push_back({std::move(tensors2), std::move(irrep_pairs)});
          }
        }
      }
    }

    // (IV) serial assembly with MPI terminal-irrep block distribution.
    // Terminal (lam,mu,P1,P2,P3) of each basis irrep, per basis (two bases:
    // bra terminal from bras, ket terminal from kets).
    auto make_term = [](const auto& db)
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
    auto block_key = [&](size_t bra_id, size_t ket_id)
    {
      std::array<int16_t, 10> k;
      const auto& b = term_bra[bra_id]; const auto& q = term_ket[ket_id];
      for (int z = 0; z < 5; z++) { k[z] = b[z]; k[z + 5] = q[z]; }
      return k;
    };

    // Flatten to a cost-weighted work list; LPT (heaviest first) so the block
    // round-robin below hands the heaviest blocks out first.
    struct WorkItem { size_t p_idx, r_idx, cost; };
    std::vector<WorkItem> work_items;
    for (size_t p = 0; p < dist_pairs.size(); p++)
      for (size_t r = 0; r < all_results[p].size(); r++)
      {
        auto& res = all_results[p][r];
        size_t cost = 0;
        for (auto& [bra_id, ket_id] : res.irrep_pairs)
        {
          const auto& bl = bras_labels[bra_id];
          size_t bd = bl.back()[5] + bl.back()[0] * bl.back()[2];
          const auto& kl = kets_labels[ket_id];
          size_t kd = kl.back()[5] + kl.back()[0] * kl.back()[2];
          cost += bd * kd;
        }
        cost += res.tensors2.size() * res.irrep_pairs.size();
        work_items.push_back({p, r, cost});
      }
    dist_pairs.clear();
    dist_pairs.shrink_to_fit();

    std::sort(work_items.begin(), work_items.end(),
              [](const auto& a, const auto& b) { return a.cost > b.cost; });

#ifdef USE_MPI
    // Assign each terminal-irrep block to one rank on first encounter
    // (round-robin). Deterministic across ranks: work_item order is identical
    // everywhere -> every block lands on exactly one rank.
    std::map<std::array<int16_t, 10>, int> block_rank;
    int next_block_rank = 0;
#endif

    for (auto& item : work_items)
    {
      auto& res = all_results[item.p_idx][item.r_idx];

      bool tensors2_added = false;
      size_t local_tensors2_id = 0;

      for (auto& [bra_id, ket_id] : res.irrep_pairs)
      {
#ifdef USE_MPI
        // keep this pair only if its terminal-irrep block is owned by this rank
        auto bk = block_key(bra_id, ket_id);
        auto it = block_rank.find(bk);
        int owner;
        if (it == block_rank.end())
        {
          owner = next_block_rank;
          block_rank.emplace(bk, owner);
          next_block_rank = (next_block_rank + 1) % mpi_size;
        }
        else owner = it->second;
        if (owner == mpi_rank)
#endif
        {
          if (!tensors2_added)
          {
            local_tensors2_id = connected_tensors.size();
            connected_tensors.push_back(std::move(res.tensors2));
            tensors2_added = true;
          }
          connected_irreps.emplace_back(bra_id, ket_id, local_tensors2_id);
        }
      }
    }

    work_items.clear();
    work_items.shrink_to_fit();
    all_results.clear();
    all_results.shrink_to_fit();

    // Reorder the per-rank work list into terminal-irrep blocks so the
    // block-local coupling coefficients are computed once and reused.
    auto block_id_of_ci =
        reorder_into_blocks(connected_irreps, connected_tensors,
                            bras_irreps, kets_irreps, tensors);

    // --- Row-owner groups for direct-write assembly ---------------------------
    // Rectangular case: every entry's row i_ lies in bra_id's row range
    // (add(i_, j_, ...) with i_ from the bra basis), and each bra_id has one
    // terminal irrep. The work list is sorted by (term_bra, term_ket), so
    // term_bra groups are contiguous ci ranges writing DISJOINT row sets:
    // threads write straight into the global matrix -- no thread-local
    // matrices, no serial merge. Heaviest groups first for load balance.
    std::vector<size_t> group_start;
    for (size_t ci = 0; ci < connected_irreps.size(); ci++)
      if (ci == 0 ||
          term_bra[std::get<0>(connected_irreps[ci])] !=
          term_bra[std::get<0>(connected_irreps[ci - 1])])
        group_start.push_back(ci);
    group_start.push_back(connected_irreps.size());

    const size_t n_groups = group_start.empty() ? 0 : group_start.size() - 1;
    std::vector<size_t> group_order(n_groups);
    for (size_t g = 0; g < n_groups; g++) group_order[g] = g;
    std::sort(group_order.begin(), group_order.end(), [&](size_t a, size_t b)
    { return (group_start[a + 1] - group_start[a]) >
             (group_start[b + 1] - group_start[b]); });

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
#ifdef USE_MPI
    std::cerr << "Matrix elements preparation time: " << elapsed.count()/60
              << "min on rank " << mpi_rank << std::endl;
    std::cerr << "Number of connected irreps on rank " << mpi_rank
              << ": " << connected_irreps.size() << std::endl;
#else
    std::cout << "Matrix elements preparation time: " << elapsed.count()/60 << "min" << std::endl;
    std::cout << "Number of connected irreps: " << connected_irreps.size() << std::endl;
#endif

    // PREPARATION PHASE --- END

    qcx_utils::CSRMatrix global(bras_dim, kets_dim);

    // RME database
    U3xU4::RME_db RMEdb(std::max(bras.maxShell(), kets.maxShell()));

#ifdef USE_OPENMP
#pragma omp parallel
{
    sam::initialize_thread();
#endif

    // Dynamic loop over term_bra owner groups (heaviest first): each group
    // owns a disjoint set of rows, so threads write straight into `global`
    // (see the owner-group construction above).
#pragma omp for schedule(dynamic, 1)
    for (size_t gi = 0; gi < n_groups; gi++)
    {
    const size_t grp = group_order[gi];
    for (size_t ci = group_start[grp]; ci < group_start[grp + 1]; ci++)
    {
      auto bra_id = std::get<0>(connected_irreps[ci]);
      auto ket_id = std::get<1>(connected_irreps[ci]);
      auto connected_tensors_id = std::get<2>(connected_irreps[ci]);

      // publish this pair's terminal-irrep block for the block-local caches
      g_asm_block = block_id_of_ci[ci];

      const auto& tensors2 = connected_tensors[connected_tensors_id];
      {
        const auto& bra = bras_irreps[bra_id];
        size_t bra_amax = bra.alphamax();
        auto bra_offset = bras_irreps.offset(bra_id);
        const auto& bra_labels = bras_labels[bra_id];
        size_t bra_dim = bra_labels.back()[5] +
                         bra_labels.back()[0] * bra_labels.back()[2];

        const auto& ket = kets_irreps[ket_id];
        size_t ket_amax = ket.alphamax();
        auto ket_offset = kets_irreps.offset(ket_id);
        const auto& ket_labels = kets_labels[ket_id];
        size_t ket_dim = ket_labels.back()[5] +
                         ket_labels.back()[0] * ket_labels.back()[2];

        // unify bra and ket irreps
        U3xU4::IrrepUnifier iu;
        const auto& [ubra, uket] = iu.unify(bra, ket);
        const auto& ket_irr = (ubra.omega.empty())
                     ? uket.gamma[0].u3u4() : uket.omega.back().u3u4();
        const auto& bra_irr = (ubra.omega.empty())
                     ? ubra.gamma[0].u3u4() : ubra.omega.back().u3u4();

        // MEs block for this irrep pair, row-major in a thread-local flat
        // buffer that only grows. Element (i, j) is at MEs[i * C + j].
        const size_t R = bra_amax * bra_dim;
        const size_t C = ket_amax * ket_dim;
        thread_local std::vector<double> MEs;
        MEs.assign(R * C, 0.0);

        for (const auto& tensor_id : tensors2)
        {
          const auto& tensor = tensors[tensor_id];

          ASM_TIC(_t_unify);
          auto uT = U3xU4::unify_tensor(tensor, ubra, uket);
          ASM_TOC(unify, _t_unify);
          if (!uT.has_value()) continue;

          const auto& Gammas = uT->Gammas();
          const auto& Omegas_LLSSTTstrengths = uT->Omegas_LLSSTTstrengths();

          ASM_TIC(_t_rme);
          auto RMEs = uT->RME(ubra, uket, gelfand_labels, RMEdb);
          ASM_TOC(rme, _t_rme);
          assert(RMEs.size() == Omegas_LLSSTTstrengths.size());

#ifdef TIME_ASSEMBLY
          AsmScope _sw(g_asm_prof[ASM_TID].work);
#endif
          for (size_t rme_id = 0; rme_id < RMEs.size(); rme_id++)
          {
            const auto& rmes = RMEs[rme_id];
            const auto& Omegas = Omegas_LLSSTTstrengths[rme_id].first;
            const auto& LLSSTTstrengths = Omegas_LLSSTTstrengths[rme_id].second;

            const auto& irr = (Omegas.empty())
                            ? Gammas.back().u3u4() : Omegas.back().u3u4();

            size_t tensor_amax = U3xU4::alphamax(Gammas, Omegas);
            auto rhomax = rmes.rhomax(), etamax = rmes.etamax();

            for (const auto& [kmax, LL, zmax, SS, TT, strengths]
                                                : LLSSTTstrengths)
            {
              for (const auto& bra_label : bra_labels)
              {
                const auto& bra_kmax = bra_label[0];
                const auto& bra_LL = bra_label[1];
                const auto& bra_zmax = bra_label[2];
                const auto& bra_SS = bra_label[3];
                const auto& bra_TT = bra_label[4];
                const auto& bra_label_offset = bra_label[5];

                for (const auto& ket_label : ket_labels)
                {
                  const auto& ket_kmax = ket_label[0];
                  const auto& ket_LL = ket_label[1];
                  const auto& ket_zmax = ket_label[2];
                  const auto& ket_SS = ket_label[3];
                  const auto& ket_TT = ket_label[4];
                  const auto& ket_label_offset = ket_label[5];

                  auto cgcT = cgcs.get({ket_TT, TT, bra_TT, kets_MMT, MMT_opt, bras_MMT});
                  auto cgcU2 = cgcJ * cgcT * racah9.get({ket_LL, ket_SS, kets_JJ,
                                                         LL, SS, JJ_opt,
                                                         bra_LL, bra_SS, bras_JJ});
                  if (std::fabs(cgcU2) < 1e-12) continue;

                  const auto& cgcU3 = cached_U3SO3(wigU3SO3db,
                    ket_irr.lam(), ket_irr.mu(), ket_LL / 2,
                    irr.lam(), irr.mu(), LL / 2,
                    bra_irr.lam(), bra_irr.mu(), bra_LL / 2);
                  if (cgcU3.empty()) continue;

                  const auto& cgcU4 = cached_U4SU2SU2(wigU4SU2SU2db,
                    ket_irr.P1(), ket_irr.P2(), ket_irr.P3(), ket_SS, ket_TT,
                    irr.P1(), irr.P2(), irr.P3(), SS, TT,
                    bra_irr.P1(), bra_irr.P2(), bra_irr.P3(), bra_SS, bra_TT);
                  if (cgcU4.empty()) continue;

                  for (size_t bra_a = 0; bra_a < bra_amax; bra_a++)
                  {
                    for (size_t bra_k = 0; bra_k < bra_kmax; bra_k++)
                    {
                      for (size_t bra_z = 0; bra_z < bra_zmax; bra_z++)
                      {
                        size_t i =  bra_a * bra_dim
                            + bra_label_offset + (bra_k * bra_zmax + bra_z);

                        for (size_t ket_a = 0; ket_a < ket_amax; ket_a++)
                        {
                          for (size_t ket_k = 0; ket_k < ket_kmax; ket_k++)
                          {
                            for (size_t ket_z = 0; ket_z < ket_zmax; ket_z++)
                            {
                              size_t j = ket_a * ket_dim
                                + ket_label_offset + (ket_k * ket_zmax + ket_z);

                              double acc = 0.0;
                              for (size_t tensor_a = 0; tensor_a < tensor_amax; tensor_a++)
                              {
                                for (size_t k = 0; k < kmax; k++)
                                {
                                  for (size_t z = 0; z < zmax; z++)
                                  {
                                    const double sc = cgcU2 *
                                        (*strengths)[(tensor_a * kmax + k) * zmax + z];

                                    for (size_t rho = 0; rho < rhomax; rho++)
                                    {
                                      const double c3 = cgcU3(ket_k, k, bra_k, rho);
                                      if (c3 == 0.0) continue;
                                      const double sc3 = sc * c3;

                                      for (size_t eta = 0; eta < etamax; eta++)
                                        acc += sc3
                                             * cgcU4(ket_z, z, bra_z, eta)
                                             * rmes(bra_a, ket_a, tensor_a, rho, eta);
                                    }
                                  }
                                }
                              }
                              MEs[i * C + j] += acc;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }

        // Direct write into global: this thread owns every row of the current
        // term_bra group (disjoint from all other groups).
        for (size_t i = 0; i < R; i++)
        {
          size_t i_ = i + bra_offset;
          for (size_t j = 0; j < C; j++)
          {
            size_t j_ = j + ket_offset;
            double v = MEs[i * C + j];
            if (std::fabs(v) > 1e-10) global.add(i_, j_, v);
          }
        }
      }
    }
    } // group loop

#ifdef USE_OPENMP
    sam::finalize_thread();
} // pragma omp parallel
#endif

    sam::clear();
    RMEdb.finalize();
    connected_irreps.clear();
    connected_irreps.shrink_to_fit();
    connected_tensors.clear();
    connected_tensors.shrink_to_fit();

#ifdef USE_MPI
    std::cerr << "Finalize rectangular CSR on rank " << mpi_rank << std::endl;
    global.finalize();
    std::cerr << "Finish computing " << global.size() << " matrix elements on rank "
              << mpi_rank << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
#else
    global.finalize();
    std::cout << "Finish computing " << global.size() << " matrix elements" << std::endl;
#endif

    return global;
  }

  template qcx_utils::CSRMatrix
    MatrixElements_CSR(const MultiShellTensors_phys& tensors,
                  const U3xU4::ManyNucleonBasis& bras,
                  const U3xU4::ManyNucleonBasis& kets,
                  const U3xU4::Table_gelfand& gelfand_labels);

  template qcx_utils::CSRMatrix
    MatrixElements_CSR(const MultiShellTensors_ShareGammas_phys& tensors,
                  const U3xU4::ManyNucleonBasis& bras,
                  const U3xU4::ManyNucleonBasis& kets,
                  const U3xU4::Table_gelfand& gelfand_labels);

} // namespace U3xU4
