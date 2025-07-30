#include <algorithm>
#include <cassert>
#include <cmath>
#include <tuple>

#include <TwoBodyOperatorJT.h>

namespace JT
{
  void TwoBodyOperator::is() const
  {
    switch (type)
    {
    case 1:
      std::cout << "Two-body part of L.L" << std::endl;
      break;
    case 2:
      std::cout << "Two-body part of L.S" << std::endl;
      break;
    case 3:
      std::cout << "Two-body part of CoM number operator without 1/A" << std::endl;
      break;
    case 4:
      std::cout << "Two-body part of the relative kinetic energy without hw/A" << std::endl;
      break;
    case 5:
      std::cout << "Two-body part of the intrinsic mass radius squared without -1/A" << std::endl;
      break;
    default:
      throw std::invalid_argument("This operator has not been implemented");
    }
  }

  // ME: / na lla jja mmja mmta | O | nc llc jjc mmjc mmtc \
  // 	   \ nb llb jjb mmjb mmtb |	  | nd lld jjb mmjd mmtd /
  double TwoBodyOperator::sphericalME(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb, int nc, int llc,
                                      int jjc, int mmjc, int mmtc, int nd,
                                      int lld, int jjd, int mmjd, int mmtd,
                                      qcx_utils::SO3_CGC& cgcs,
                                      qcx_utils::SO3_Racah& racah,
                                      qcx_utils::SO3_Racah9& racah9) const
  {
    double x = 0.0;
    switch (type)
    {
    case 1: // Two-body part of L.L
      if (na == nc && nb == nd && lla == llc && llb == lld && mmta == mmtc &&
          mmtb == mmtd)
      {
        double phase = ((jja + jjb + jjc + jjd) / 2 % 2 == 0) ? 1.0 : -1.0,
               factor = std::sqrt(llc * (llc + 2.0) * lld * (lld + 2.0)) / 4.0,
               U1 = racah.compute(llc, 1, 2, jja, jjc, llc),
               U2 = racah.compute(lld, 1, 2, jjb, jjd, lld), cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
                 cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
        x = 2 * phase * factor * U1 * U2 * cgc;
      }
      break;
    case 2: // Two-body part of L.S
      if (na == nc && nb == nd && lla == llc && llb == lld && mmta == mmtc &&
          mmtb == mmtd)
      {
        double U1 = racah.compute(llc, 1, 2, jja, jjc, llc),
               U2 = racah.compute(1,lld,2,jjb,jjd,1),
               U3 = racah.compute(lld, 1, 2, jjb, jjd, lld),
               U4 = racah.compute(1,llc,2,jja,jjc,1), 
               phase12 = ((jja - jjc)/2 % 2 == 0) ? 1.0 : -1.0,
               phase34 = ((jjb - jjd)/2 % 2 == 0) ? 1.0 : -1.0,
               factor = (phase12 * std::sqrt(llc * (llc + 2.0) / 4.0) * U1 * U2 
                      + phase34 * std::sqrt(lld * (lld + 2.0) / 4.0) * U3 * U4),
               cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
                 cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
        x = std::sqrt(0.75) * cgc * factor;
      }
      break;
    case 3: // Two-body part of CoM number operator without 1/A
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
                 cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
        x = cgc * (HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9) * 
                   HO::bosonAnnihilatorRME(nb,llb,jjb,nd,lld,jjd,racah9) +
                   HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9) *
                   HO::bosonCreatorRME(nb,llb,jjb,nd,lld,jjd,racah9));
      }
      break;
    case 4: // Two-body part of the relative kinetic energy without hw/A
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
                 cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
        x = cgc * (HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9) * 
                  (HO::bosonCreatorRME(nb,llb,jjb,nd,lld,jjd,racah9) -
                   HO::bosonAnnihilatorRME(nb,llb,jjb,nd,lld,jjd,racah9)) +
                   HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9) *
                  (HO::bosonAnnihilatorRME(nb,llb,jjb,nd,lld,jjd,racah9) -
                   HO::bosonCreatorRME(nb,llb,jjb,nd,lld,jjd,racah9))) / 2.0;
      }
      break;
    case 5: // Two-body part of the intrinsic mass radius squared without -1/A
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
                 cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
        x = 2.0 * cgc * (HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9) * 
                  (HO::bosonCreatorRME(nb,llb,jjb,nd,lld,jjd,racah9) +
                   HO::bosonAnnihilatorRME(nb,llb,jjb,nd,lld,jjd,racah9)) +
                   HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9) *
                  (HO::bosonAnnihilatorRME(nb,llb,jjb,nd,lld,jjd,racah9) +
                   HO::bosonCreatorRME(nb,llb,jjb,nd,lld,jjd,racah9)));
      }
      break;
    default:
      throw std::invalid_argument("This operator has not been implemented");
    }

    return x;
  }

  // Antisymmetrized ME
  double TwoBodyOperator::sphericalME_AS(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb, int nc, int llc,
                                      int jjc, int mmjc, int mmtc, int nd,
                                      int lld, int jjd, int mmjd, int mmtd,
                                      qcx_utils::SO3_CGC& cgcs,
                                      qcx_utils::SO3_Racah& racah,
                                      qcx_utils::SO3_Racah9& racah9) const
  {
    return (sphericalME(na,lla,jja,mmja,mmta, nb,llb,jjb,mmjb,mmtb,
                        nc,llc,jjc,mmjc,mmtc, nd,lld,jjd,mmjd,mmtd, 
                        cgcs,racah,racah9) -
            sphericalME(na,lla,jja,mmja,mmta, nb,llb,jjb,mmjb,mmtb,
                        nd,lld,jjd,mmjd,mmtd, nc,llc,jjc,mmjc,mmtc,
                        cgcs, racah,racah9))/4.0;
  }

  // Generate a list of al {na+1, nb+1, -nc-1, -nd-1} combinations such that
  // 0 <= na, nb, nc, nd <= maxShell
  std::tuple<std::vector<std::array<int, 4>>,
             std::vector<std::vector<std::vector<int>>>,
             std::vector<std::vector<int>>>
  TwoBodyOperator::GenerateAllCombinationsHOShells(int maxShell) const
  {
    std::vector<std::array<int, 4>> na_nb_nc_nds;
    std::vector<std::vector<std::vector<int>>> inverses;
    std::vector<std::vector<int>> phases;

    for (int na = 0; na <= maxShell; na++)
    {
      for (int nb = 0; nb <= maxShell; nb++)
      {
        for (int nc = 0; nc <= maxShell; nc++)
        {
          for (int nd = 0; nd <= maxShell; nd++)
          {
            std::array<int, 4> ns = {na + 1, nb + 1, -nc - 1, -nd - 1};
            auto [sorted_ns, inverse, phase] = sort_shells(ns);

            auto it = std::find(na_nb_nc_nds.begin(), na_nb_nc_nds.end(), sorted_ns);
            if (it != na_nb_nc_nds.end())
            {
              int index = std::distance(na_nb_nc_nds.begin(), it);
              inverses[index].push_back(inverse);
              phases[index].push_back(phase);
            }
            else
            {
              na_nb_nc_nds.push_back(sorted_ns);
              inverses.push_back({inverse});
              phases.push_back({phase});
            }
          }
        }
      }
    }
    return std::make_tuple(na_nb_nc_nds, inverses, phases);
  }

  // Restrict:
  // 0 <= (na - valenceShell + (nb - valenceShell) <= Nmax
  // 0 <= (nc - valenceShell + (nd - valenceShell) <= Nmax
  std::tuple<std::vector<std::array<int, 4>>,
             std::vector<std::vector<std::vector<int>>>,
             std::vector<std::vector<int>>>
  TwoBodyOperator::GenerateAllCombinationsHOShells(int Nmax, int valenceShell,
                                                   int maxShell) const
  {
    std::vector<std::array<int, 4>> na_nb_nc_nds;
    std::vector<std::vector<std::vector<int>>> inverses;
    std::vector<std::vector<int>> phases;

    for (int na = 0; na <= maxShell; na++)
    {
      for (int nb = 0; nb <= maxShell; nb++)
      {
        int adxad_Nhw = 0;
        if (na > valenceShell)
          adxad_Nhw += (na - valenceShell);
        if (nb > valenceShell)
          adxad_Nhw += (nb - valenceShell);
        if (adxad_Nhw > Nmax)
          continue;

        for (int nc = 0; nc <= maxShell; nc++)
        {
          for (int nd = 0; nd <= maxShell; nd++)
          {
            int atxat_Nhw = 0;
            if (nc > valenceShell)
              atxat_Nhw += (nc - valenceShell);
            if (nd > valenceShell)
              atxat_Nhw += (nd - valenceShell);
            if (atxat_Nhw > Nmax)
              continue;

            std::array<int, 4> ns = {na + 1, nb + 1, -nc - 1, -nd - 1};
            auto [sorted_ns, inverse, phase] = sort_shells(ns);

            auto it = std::find(na_nb_nc_nds.begin(), na_nb_nc_nds.end(), sorted_ns);
            if (it != na_nb_nc_nds.end())
            {
              int index = std::distance(na_nb_nc_nds.begin(), it);
              inverses[index].push_back(inverse);
              phases[index].push_back(phase);
            }
            else
            {
              na_nb_nc_nds.push_back(sorted_ns);
              inverses.push_back({inverse});
              phases.push_back({phase});
            }
          }
        }
      }
    }
    return std::make_tuple(na_nb_nc_nds, inverses, phases);
  }

  // Perform permutation of tensor in increasing shell.
  // For tensors of the same shell, ad's are in the front of at's.
  // Exchange of 2 tensors always incurs a phase of -1.0.
  // No exchange of ad and a of the same shell occurs.
  std::tuple<std::array<int, 4>, std::vector<int>, int>
  TwoBodyOperator::sort_shells(const std::array<int, 4>& original) const
  {
    // index vector
    std::vector<int> indices = {0, 1, 2, 3};

    // sort indices based on absolute value of original
    std::sort(indices.begin(), indices.end(), [&](int i, int j)
              { return std::abs(original[i]) < std::abs(original[j]); });

    // build sorted array
    std::array<int, 4> sorted;
    for (int i = 0; i < 4; i++)
      sorted[i] = original[indices[i]];

    // build inverse permutation
    std::vector<int> inverse_indices(4);
    for (int i = 0; i < 4; i++)
      inverse_indices[indices[i]] = i;

    // compute permutation sign using cycle decomposition
    int sign = 1;
    std::vector<bool> visited(4, false);
    for (int i = 0; i < 4; i++)
    {
      if (visited[i])
        continue;
      int cycle_len = 0, j = i;
      while (!visited[j])
      {
        visited[j] = true;
        j = indices[j];
        cycle_len++;
      }
      if (cycle_len > 1 && (cycle_len % 2 == 0))
        sign *= -1;
    }

    return {sorted, inverse_indices, sign};
  }

  // Restores the original ordering using inverse permutation
  std::array<int, 4>
  TwoBodyOperator::inverse_permutation(const std::array<int, 4>& sorted,
                                       const std::vector<int>& inverse) const
  {
    std::array<int, 4> restored;
    for (int i = 0; i < 4; ++i)
      restored[i] = sorted[inverse[i]];
    return restored;
  }

  // Group tensors of the same shell
  // Results are contained in a list of 3-element arrays:
  // first: shell number
  // second: number of ad's
  // thrid: number of at's
  std::vector<std::array<int, 3>>
  TwoBodyOperator::group_same_shell(const std::array<int, 4>& sorted) const
  {
    std::vector<std::array<int, 3>> groups;
    for (int i = 0; i < 4; i++)
    {
      int shell = std::abs(sorted[i]), ads = 0, ats = 0, j;
      for (j = i; j < 4; j++)
        if (sorted[j] == shell)
          ads++;
        else if (sorted[j] == -shell)
          ats++;
        else
          break;
      std::array<int, 3> group = {shell - 1, ads, ats};
      groups.push_back(group);
      i = j - 1;
    }
    return groups;
  }

  // Set coefficient for two-body multi-shell tensor.
  // It returns a flag, which indicate whether this tensor can be ignored.
  bool TwoBodyOperator::setTensorStrength(
      U3xU4::TwoBodyMultiShellTensor& tensor, 
      const std::vector<std::vector<int>>& inverse, 
      const std::vector<int>& phase,
      qcx_utils::SO3_CGC& cgcs,
      qcx_utils::SO3_Racah& racah, qcx_utils::SO3_Racah9& racah9,
      u3::WigU3SO3_db& wigU3SO3db, u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
  {
    /* Coefficients in the expansion of two-body JT-coupled operator.
       The size of this array is equal to rhomax * etamax * kappamax * zetamax.
       To access the elements:
       for (int r = 0; r < rhomax; r++)
         for (int e = 0; e < etamax; e++)
           for (int k = 0; k < kappa; k++)
             for (int z = 0; z < zeta; z++)
               index = ((r * etamax + e) * kappamax + k) * zetamax + z;
    */

    std::vector<double> coeffs_(
        tensor.rhomax * tensor.etamax * tensor.kappamax * tensor.zetamax, 0.0);

    auto [coeffsArr, opsArr] =
        tensor.expand(cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

    for (int i = 0; i < coeffsArr.size(); i++)
    {
      std::array<int, 4> ns, lls, jjs, mmjs, mmts;

      int j = 0;
      for (const auto& op : opsArr[i])
      {
        std::visit(
            [&](auto&& o)
            {
              std::tie(ns[j], lls[j], jjs[j], mmjs[j], mmts[j]) = o.getLabels();
            },
            op);
        j++;
      }

      double me = 0.0;

      for (int k = 0; k < inverse.size(); k++)
      {
        std::array<int, 4> ns_orig = inverse_permutation(ns, inverse[k]),
                           lls_orig = inverse_permutation(lls, inverse[k]),
                           jjs_orig = inverse_permutation(jjs, inverse[k]),
                           mmjs_orig = inverse_permutation(mmjs, inverse[k]),
                           mmts_orig = inverse_permutation(mmts, inverse[k]);

        // me needs computing with normal ordering
        me += phase[k] * sphericalME_AS(
          ns_orig[0], lls_orig[0], jjs_orig[0], mmjs_orig[0], mmts_orig[0],
          ns_orig[1], lls_orig[1], jjs_orig[1], mmjs_orig[1], mmts_orig[1],
          ns_orig[3], lls_orig[3], jjs_orig[3], mmjs_orig[3], mmts_orig[3],
          ns_orig[2], lls_orig[2], jjs_orig[2], mmjs_orig[2], mmts_orig[2],
          cgcs, racah, racah9);
      }

      if (std::fabs(me) < 1e-12)
        continue;

      for (int c = 0; c < coeffs_.size(); c++)
        coeffs_[c] += me * coeffsArr[i][c];
    }

    double norm = 0.0;
    for (int c = 0; c < coeffs_.size(); c++)
    {
      //coeffs_[c] *= phase;
      norm += std::fabs(coeffs_[c]);
    }
    bool ignore = (norm < 1e-10);

    if (ignore == false)
      tensor.coeffs = coeffs_;

    return ignore;
  }

#define RESTRICT_WITH_NMAX
  std::vector<U3xU4::TwoBodyMultiShellTensor> TwoBodyOperator::expand(
      int Nmax, int valenceShell, int maxShell, HO::Table_nljmjmt& labels,
      qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racah, 
      qcx_utils::SO3_Racah9& racah9, u3::WigU3SO3_db& wigU3SO3db, 
      u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
  {
    std::vector<U3xU4::TwoBodyMultiShellTensor> tensors;

#ifdef RESTRICT_WITH_NMAX
    auto [all_na_nb_nc_nd, all_inverse, all_phase] =
        GenerateAllCombinationsHOShells(Nmax, valenceShell, maxShell);
#else
    auto all_na_nb_nc_nd = GenerateAllCombinationsHOShells(maxShell);
#endif
    
    for (int i = 0; i < all_na_nb_nc_nd.size(); i++)
    {
/*
      for (auto sorted_n : all_na_nb_nc_nd[i]) 
        std::cout << sorted_n << " ";
      std::cout << "\n phases = ";
      for (auto phase : all_phase[i])
        std::cout << phase << " ";
      std::cout << "\n";
*/

      //auto groups_ns = group_same_shell(sorted_ns);
      auto groups_ns = group_same_shell(all_na_nb_nc_nd[i]);
      std::vector<std::vector<U3xU4::Tadat>> Tadat_lists;
      for (auto group : groups_ns)
      {
        Tadat_lists.push_back(U3xU4::GenerateSingleShellTensors(
            std::get<0>(group), std::get<1>(group), std::get<2>(group)));
      }

      std::vector<U3xU4::CouplingPath> all_results;
      U3xU4::CoupleSingShellTensors(Tadat_lists, 0, {}, {}, {}, {},
                                    all_results);

      for (const auto& [Tadat_choices, coupling_info_list] : all_results)
      {
        std::vector<U3xU4::irrep> coupled_irreps;
        int Rhomax, Etamax;
        for (const auto& [irrep, rhomax, etamax] : coupling_info_list)
        {
          assert(rhomax == 1);
          assert(etamax == 1);
          coupled_irreps.push_back(irrep);
          Rhomax = rhomax;
          Etamax = etamax;
        }

        // decompose into L
        std::vector<u3::physicalvector> pv3s;
        // decompose into 2S and 2T
        std::vector<u4::physicalvector> pv4s;

        if (coupled_irreps.empty()) // Tadat_choices has one element only
        {
          pv3s = u3::physicalBasis(Tadat_choices[0].T.u3);
          pv4s = u4::physicalBasis(Tadat_choices[0].T.u4);
        }
        else
        {
          pv3s = u3::physicalBasis(coupled_irreps.back().u3);
          pv4s = u4::physicalBasis(coupled_irreps.back().u4);
        }

        for (int pv3 = 0; pv3 < pv3s.size(); pv3++)
        {
          int kappa0 = pv3s[pv3].kappa, LL0 = 2 * pv3s[pv3].L;

          for (int pv4 = 0; pv4 < pv4s.size(); pv4++)
          {
            int zeta0 = pv4s[pv4].zeta, SS0 = pv4s[pv4].SS, TT0 = pv4s[pv4].TT;

            // couple LL and SS to JJ
            if (JJ < std::fabs(LL0 - SS0) || JJ > (LL0 + SS0) || TT0 != TT)
              continue;

            // no coupled irrep, it collapses into a single-shell tensor
            if (coupled_irreps.empty())
            {
              U3xU4::TwoBodyMultiShellTensor tensor(
                  Tadat_choices, coupled_irreps, Tadat_choices[0].rho0max,
                  Tadat_choices[0].eta0max, kappa0, LL0, zeta0, SS0, TT, MMT,
                  JJ, MMJ, {}, labels);
              bool ignore =
                  setTensorStrength(tensor, all_inverse[i], all_phase[i], cgcs, racah,
                                 racah9, wigU3SO3db, wigU4SU2SU2db);
              if (ignore == false)
              {
                //std::cout << tensor << std::endl;
                tensors.push_back(tensor);
              }
            }
            // otherwise, it remains a multi-shell tensor
            else
            {
              U3xU4::TwoBodyMultiShellTensor tensor(
                  Tadat_choices, coupled_irreps, Rhomax, Etamax, kappa0, LL0,
                  zeta0, SS0, TT, MMT, JJ, MMJ, {}, labels);
              bool ignore =
                  setTensorStrength(tensor, all_inverse[i], all_phase[i], cgcs, racah,
                                 racah9, wigU3SO3db, wigU4SU2SU2db);
              if (ignore == false)
              {
                //std::cout << tensor << std::endl;
                tensors.push_back(tensor);
              }
            }
          }
        }
      } 
    }

    return tensors;
  }
} // namespace JT
