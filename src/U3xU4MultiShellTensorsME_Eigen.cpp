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
  // Sparse format provided by Eigen with OpenMP
  template <typename TensorType>
  Eigen::SparseMatrix<double> MatrixElements_Eigen(
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
      Eigen::SparseMatrix<double> mat(dim, dim);
      return mat;
    }

    // PREPARATION PHASE --- BEGIN

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
            for (const auto& bra_id : bra_irrep_ids)
              for (const auto& ket_id : ket_irrep_ids)
              {
#ifdef EXPLOIT_SYMMETRIC
                if (bra_id < ket_id) continue;
#endif
                // connected_irreps.emplace_back(bra_id, ket_id, connected_tensors.size());

                irrep_pairs.push_back({bra_id, ket_id});
              }

            // connected_tensors.push_back(std::move(tensors2));

            // write only to dist_pair's own slot — no race
            all_results[p].push_back({std::move(tensors2), std::move(irrep_pairs)});
          }
        }
      }
    }

    // (IV) serial assembly
    for (size_t p = 0; p < dist_pairs.size(); p++)
    {
      for (auto& all_result : all_results[p])
      {
        // connected_tensors.size() is now correct because we're
        // doing this sequentially in the original loop order
        size_t tensors2_id = connected_tensors.size();

        for (auto& [bra_id, ket_id] : all_result.irrep_pairs)
          connected_irreps.emplace_back(bra_id, ket_id, tensors2_id);

        connected_tensors.push_back(std::move(all_result.tensors2));
      }
    }

    all_results.clear();
    all_results.shrink_to_fit();
    dist_pairs.clear();
    dist_pairs.shrink_to_fit();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Matrix elements preparation time: " << elapsed.count()/60 << "min" << std::endl;
    std::cout << "Number of connected irreps: " << connected_irreps.size() << std::endl;

    // PREPARATION PHASE --- END

    std::cout << "Start computing matrix" << std::endl;

    // RME database
    U3xU4::RME_db RMEdb(basis.maxShell());

    // Reorder the work list into terminal-irrep blocks (basis + tensor
    // reordering) so the block-local coupling-coefficient tables reuse.
    auto block_id_of_ci =
        reorder_into_blocks(connected_irreps, connected_tensors,
                            irreps, irreps, tensors);

    Eigen::SparseMatrix<double> mat(dim, dim);
    mat.reserve(connected_irreps.size() * 100);

    std::vector<Eigen::Triplet<double>> ija;

    const size_t SIZE_LIMIT = 20000000;
    size_t cnt = 0;

#ifdef USE_OPENMP
#pragma omp parallel
{
    sam::initialize_thread();

    std::vector<Eigen::Triplet<double>> ija_local;
    ija_local.reserve(SIZE_LIMIT);

    auto flush = [&]()
    {
      if (ija_local.empty()) return;

      Eigen::SparseMatrix<double> temp(dim, dim);
      temp.setFromTriplets(ija_local.begin(), ija_local.end());

#pragma omp critical(flushing)
{
      std::cerr << cnt++ << ": " << ija_local.size() << std::endl;
      mat += temp;

      // for (auto& t : ija_local) mat.insert(t.row(), t.col()) = t.value();
}

      ija_local.clear();
    };
#endif

#pragma omp for schedule(static)
    for (size_t ci = 0; ci < connected_irreps.size(); ci++)
    // for (auto [bra_id, ket_id, connected_tensors_id] : connected_irreps)
    {
      auto bra_id = std::get<0>(connected_irreps[ci]);
      auto ket_id = std::get<1>(connected_irreps[ci]);
      auto connected_tensors_id = std::get<2>(connected_irreps[ci]);

      // publish this pair's terminal-irrep block for block-local coeff reuse
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

#ifdef USE_OPENMP
                auto& ija_ = ija_local;
#else
                auto& ija_ = ija;
#endif

                // std::cout << "Start to fill sparse matrix" << std::endl;
                for (size_t i = 0; i < R; i++)
                {
                  size_t i_ = i + bra_offset;
                  for (size_t j = 0; j < C; j++)
                  {
                    size_t j_ = j + ket_offset;

                    if (std::fabs(MEs[i * C + j]) > 1e-8)
                    {
                      ija_.emplace_back(i_, j_, MEs[i * C + j]); // for i_ >= j_
#ifdef EXPLOIT_SYMMETRIC
                      if (i_ > j_) ija_.emplace_back(j_, i_, MEs[i * C + j]);
#endif
                    }
                  }
                }
                // std::cout << "Finish filling sparse matrix; " << std::endl;

#ifdef USE_OPENMP
                if (ija_local.size() >= SIZE_LIMIT) flush();
#endif
              }
            }
          }
        }
      }
    }

#ifdef USE_OPENMP
    // Final flush from local thread
    flush();

    sam::finalize_thread();

} // pragma omp parallel
#else
    std::cout << "Finish computing " << ija.size() << " elements" << std::endl;

    mat.reserve(ija.size());
    mat.setFromTriplets(ija.begin(), ija.end());
#endif

    RMEdb.finalize();
    connected_irreps.clear();
    connected_irreps.shrink_to_fit();
    connected_tensors.clear();
    connected_tensors.shrink_to_fit();

    mat.makeCompressed();

    std::cout << "Finish computing matrix" << std::endl;

    return mat;
  }

  template Eigen::SparseMatrix<double> MatrixElements_Eigen(
                  const MultiShellTensors_phys& tensors, 
                  const U3xU4::ManyNucleonBasis& basis,
                  const U3xU4::Table_gelfand& gelfand_labels);

  template Eigen::SparseMatrix<double> MatrixElements_Eigen(
                  const MultiShellTensors_ShareGammas_phys& tensors, 
                  const U3xU4::ManyNucleonBasis& basis,
                  const U3xU4::Table_gelfand& gelfand_labels);


  /////////////////// COMPUTING RECTANGULAR MATRIX ///////////////////

  // Sparse format provided by Eigen with OpenMP
  template <typename TensorType>
  Eigen::SparseMatrix<double> MatrixElements_Eigen(
                  const std::vector<TensorType>& tensors, 
                  const U3xU4::ManyNucleonBasis& bras,
                  const U3xU4::ManyNucleonBasis& kets,
                  const U3xU4::Table_gelfand& gelfand_labels)
  {
    // for (auto tensor : tensors) std::cout << tensor << std::endl;

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
    auto cgcJ = cgcs.get({kets_JJ, JJ_opt, bras_JJ,
                          MMJ, MMJ_opt, MMJ});

    if (tensors_count == 0 || std::fabs(cgcJ) == 0.0)
    {
      Eigen::SparseMatrix<double> mat(bras_dim, kets_dim);
      return mat;
    }

    // PREPARATION PHASE --- BEGIN

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<size_t>> connected_tensors;
    std::vector<std::tuple<size_t, size_t, size_t>> connected_irreps;

    // (I) flatten the outer two loops into a 1D vector of pairs
    std::vector<std::pair<size_t,size_t>> dist_pairs;
    for (size_t b = 0; b < bras.n_distributions(); b++)
      for (size_t k = 0; k < kets.n_distributions(); k++)
        dist_pairs.push_back({b, k});

    // (II) define a result struct for one (bra_gamma, ket_gamma) hit
    struct GammaResult {
      std::vector<size_t>                    tensors2;
      std::vector<std::pair<size_t,size_t>>  irrep_pairs; // (bra_id, ket_id)
    };
    std::vector<std::vector<GammaResult>> all_results(dist_pairs.size());
    //          ^                         ^
    //          one slot per dist pair    each slot holds all GammaResults
    //                                    found for that pair

    // (III) parallel for over dist_pairs 
#pragma omp parallel for schedule(dynamic)
    // for (size_t bra_dist_id = 0; bra_dist_id < bras.n_distributions(); bra_dist_id++)
    for (size_t p = 0; p < dist_pairs.size(); p++)
    {
      auto [bra_dist_id, ket_dist_id] = dist_pairs[p];
       
      auto bra_dist = bras.distribution(bra_dist_id);
      const auto& bra_gamma_irreps_ids = bras.group_dists[bra_dist_id];
      auto bra_dist_to_gamma = map_by_positive_order(bra_dist);

      // print("\nbra_dist_id: {}\t", bra_dist_id);
      // print("bra_dist_to_gamma: {}\n", bra_dist_to_gamma)

      // for (size_t ket_dist_id = 0; ket_dist_id < kets.n_distributions(); ket_dist_id++)
      {
        auto ket_dist = kets.distribution(ket_dist_id);
        const auto& ket_gamma_irreps_ids = kets.group_dists[ket_dist_id];
        auto ket_dist_to_gamma = map_by_positive_order(ket_dist);

        // print("ket_dist_id: {} \t", ket_dist_id);
        // print("ket_dist_to_gamma: {}\n", ket_dist_to_gamma);

        // Predicate to check if particles are conserved in all shells
        auto all_shells_particles = [&](const auto& tensor)
        {
          // in a shell:
          // ket_dist[n] - G.num_ats() + G.num_ads()) == bra_dist[n] &&
          // ket_dist[n] >= G.num_ats()

          size_t max_shells = std::max(ket_dist.size(), bra_dist.size());
          std::vector<int> delta(max_shells, 0);

          // Accumulate net changes from Gammas
          for (const auto& G : tensor.Gammas())
          {
            size_t n = G.shell();
            if (n >= max_shells) return false;

            delta[n] += G.num_ads() - G.num_ats();

            // Early failure: destroying more than available
            int ket_occ = (n < ket_dist.size()) ? ket_dist[n] : 0;
            if (ket_occ < G.num_ats()) return false;
          }

          // Check conservation in every shell
          for (size_t n = 0; n < max_shells; ++n)
          {
            int ket_occ = (n < ket_dist.size()) ? ket_dist[n] : 0;
            int bra_occ = (n < bra_dist.size()) ? bra_dist[n] : 0;

            if (ket_occ + delta[n] != bra_occ) return false;
          }

          return true;
        };

        std::vector<size_t> tensors1;
        // tensors1.reserve(tensors.size());
        tensors1.reserve(100);
        for (size_t i = 0; i < tensors.size(); i++)
          if (all_shells_particles(tensors[i])) tensors1.push_back(i);
        if (tensors1.empty()) continue;
        // print("Finish checking distributions\n");
        // print("tensors1: {}\n", tensors1.size());

        for (const auto& [bra_gamma_id, bra_irrep_ids] : bra_gamma_irreps_ids)
        {
          auto bra_gamma = bras.gamma(bra_gamma_id);
          // print("bra_gamma_id: {}\n", bra_gamma_id);

          for (const auto& [ket_gamma_id, ket_irrep_ids] : ket_gamma_irreps_ids)
          {
            auto ket_gamma = kets.gamma(ket_gamma_id);
            // print("ket_gamma_id: {}\n", ket_gamma_id);

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
            // print("Finish checking single shell couplings\n");
            // print("tensors2: {}\n", tensors2.size());

            // instead of pushing to connected_irreps/connected_tensors,
            // collect the irrep pairs locally
            std::vector<std::pair<size_t,size_t>> irrep_pairs;
            for (const auto& bra_id : bra_irrep_ids)
              for (const auto& ket_id : ket_irrep_ids)
              {
                // connected_irreps.emplace_back(bra_id, ket_id, connected_tensors.size());
                irrep_pairs.push_back({bra_id, ket_id});
              }
            
            // connected_tensors.push_back(std::move(tensors2));
            all_results[p].push_back({std::move(tensors2), std::move(irrep_pairs)});
          }
        }
      }
    }

    // (IV) serial assembly
    for (size_t p = 0; p < dist_pairs.size(); p++)
    {
      for (auto& all_result : all_results[p])
      {
        // connected_tensors.size() is now correct because we're
        // doing this sequentially in the original loop order
        size_t tensors2_id = connected_tensors.size();

        for (auto& [bra_id, ket_id] : all_result.irrep_pairs)
          connected_irreps.emplace_back(bra_id, ket_id, tensors2_id);

        connected_tensors.push_back(std::move(all_result.tensors2));
      }
    }

    all_results.clear();
    all_results.shrink_to_fit();
    dist_pairs.clear();
    dist_pairs.shrink_to_fit();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Matrix elements preparation time: " << elapsed.count()/60 << "min" << std::endl;
    std::cout << "Number of connected irreps: " << connected_irreps.size() << std::endl;

    // PREPARATION PHASE --- END

    std::cout << "Start computing matrix" << std::endl;

    // RME database
    U3xU4::RME_db RMEdb(std::max(bras.maxShell(), kets.maxShell()));

    // Reorder the work list into terminal-irrep blocks (two-basis transition
    // operator: bra terminal from bras, ket terminal from kets).
    auto block_id_of_ci =
        reorder_into_blocks(connected_irreps, connected_tensors,
                            bras_irreps, kets_irreps, tensors);

    Eigen::SparseMatrix<double> mat(bras_dim, kets_dim);

    std::vector<Eigen::Triplet<double>> ija;

#ifdef USE_OPENMP
    mat.reserve(connected_irreps.size() * 100);
    const size_t SIZE_LIMIT = 20000000;
    size_t cnt = 0;

#pragma omp parallel
{
    sam::initialize_thread();

    std::vector<Eigen::Triplet<double>> ija_local;
    ija_local.reserve(SIZE_LIMIT);

    auto flush = [&]()
    {
      if (ija_local.empty()) return;

      Eigen::SparseMatrix<double> temp(bras_dim, kets_dim);
      temp.setFromTriplets(ija_local.begin(), ija_local.end());

#pragma omp critical(flushing)
{
      std::cerr << cnt++ << ": " << ija_local.size() << std::endl;
      mat += temp;

      // for (auto& t : ija_local) mat.insert(t.row(), t.col()) = t.value();
}

      ija_local.clear();
    };
#endif

#pragma omp for schedule(runtime)
    for (size_t ci = 0; ci < connected_irreps.size(); ci++)
    {
      auto bra_id = std::get<0>(connected_irreps[ci]);
      auto ket_id = std::get<1>(connected_irreps[ci]);
      auto connected_tensors_id = std::get<2>(connected_irreps[ci]);

      const auto& tensors2 = connected_tensors[connected_tensors_id];
      {
        {
          {
            // for (const auto& bra_id : bra_irrep_ids)
            {
              // std::cout << "bra_id: " << bra_id << std::endl;

              const auto& bra = bras_irreps[bra_id];

              // get alphamax
              size_t bra_amax = bra.alphamax();

              // get offset
              auto bra_offset = bras_irreps.offset(bra_id);

              // get labels
              const auto& bra_labels = bras_labels[bra_id];
              size_t bra_dim = bra_labels.back()[5] +
                              bra_labels.back()[0] * bra_labels.back()[2];

              // for (const auto& ket_id : ket_irrep_ids)
              {
                // std::cout << "ket_id: " << ket_id << std::endl;

                const auto& ket = kets_irreps[ket_id];

                // get alphamax 
                size_t ket_amax = ket.alphamax();

                // get offset
                auto ket_offset = kets_irreps.offset(ket_id);

                // get labels
                const auto& ket_labels = kets_labels[ket_id];
                size_t ket_dim = ket_labels.back()[5] +
                              ket_labels.back()[0] * ket_labels.back()[2];

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
                          auto cgcT = cgcs.get({ket_TT, TT, bra_TT, kets_MMT, MMT_opt, bras_MMT});
                          auto cgcU2 = cgcJ * cgcT * racah9.get({ket_LL, ket_SS, kets_JJ,
                                                                 LL, SS, JJ_opt,
                                                                 bra_LL, bra_SS, bras_JJ});
                          if (std::fabs(cgcU2) < 1e-12) continue;
                          
                          const auto cgcU3 = wigU3SO3db.get_U3SO3(
                            ket_irr.lam(), ket_irr.mu(), ket_LL / 2,
                            irr.lam(), irr.mu(), LL / 2,
                            bra_irr.lam(), bra_irr.mu(), bra_LL / 2);
                          if (cgcU3.empty()) continue;

                          const auto cgcU4 = wigU4SU2SU2db.get_U4SU2SU2(
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
                
#ifdef USE_OPENMP
                auto& ija_ = ija_local;
#else
                auto& ija_ = ija;
#endif

                // std::cout << "Start to fill sparse matrix" << std::endl;
                for (size_t i = 0; i < R; i++)
                {
                  size_t i_ = i + bra_offset;
                  for (size_t j = 0; j < C; j++)
                  {
                    size_t j_ = j + ket_offset;

                    if (std::fabs(MEs[i * C + j]) > 1e-10)
                      ija_.emplace_back(i_, j_, MEs[i * C + j]);
                  }
                }
                // std::cout << "Finish filling sparse matrix; " << std::endl;

#ifdef USE_OPENMP
                if (ija_local.size() >= SIZE_LIMIT) flush();
#endif

              }
            }
          }
        }
      }
    }

#ifdef USE_OPENMP
    // Final flush from local thread
    flush();

    sam::finalize_thread();
} // pragma omp parallel
#else
    std::cout << "Finish computing " << ija.size() << " elements" << std::endl;

    mat.reserve(ija.size());
    mat.setFromTriplets(ija.begin(), ija.end());
#endif

    RMEdb.finalize();
    connected_irreps.clear();
    connected_irreps.shrink_to_fit();
    connected_tensors.clear();
    connected_tensors.shrink_to_fit();

    mat.makeCompressed();

    std::cout << "Finish finalizing matrix" << std::endl;

    return mat;
  }

  template Eigen::SparseMatrix<double> 
    MatrixElements_Eigen(const MultiShellTensors_phys& tensors, 
                  const U3xU4::ManyNucleonBasis& bras,
                  const U3xU4::ManyNucleonBasis& kets,
                  const U3xU4::Table_gelfand& gelfand_labels);

  template Eigen::SparseMatrix<double>
    MatrixElements_Eigen(const MultiShellTensors_ShareGammas_phys& tensors,
                  const U3xU4::ManyNucleonBasis& bras,
                  const U3xU4::ManyNucleonBasis& kets,
                  const U3xU4::Table_gelfand& gelfand_labels);

} // namespace U3xU4
