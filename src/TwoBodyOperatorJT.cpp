#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>

#include <qcx_convert.h>

#include <Databases.h>
#include <TwoBodyOperatorJT.h>

namespace JT
{
  TwoBodyOperator::TwoBodyOperator(const std::string& name_,
                                   const HO::Table_nljmjmt& labels_, int hw_,
                                   int Nmax_, int valenceShell_,
                                   int maxShell_)
      : labels(labels_)
  {
    hw = (hw_ >= 0) ? qcx_utils::ito8(hw_) : qcx_utils::ito8(20);
    Nmax = (Nmax_ >= 0) ? qcx_utils::ito8(Nmax_) : qcx_utils::ito8(14);
    valenceShell = (valenceShell_ > 0) ? qcx_utils::ito8(valenceShell_) : 0;
    maxShell = (maxShell_ >= 0) ? qcx_utils::ito8(maxShell_) : qcx_utils::ito8(14);

    const std::unordered_set<std::string> known_scalar_ops = {
        "L.L", "L.S", "CoMNumberOp", "Kinetic", "R.R"};

    const std::unordered_set<std::string> known_potentials = {
        "N2LO_EMN500_bare", "N3LO_EMN500_bare", "N4LO_EMN500_bare",
        "N3LO_EM500_bare",  "N2LO_opt_bare",    "N2LO_sat_bare"};

    if (known_scalar_ops.count(name_))
    {
      name = name_;
      JJ = 0;
      MMJ = 0;
      TT = 0;
      MMT = 0;
      parity = true;
    }
    else if (known_potentials.count(name_))
    {
      name = name_;
      JJ = 0;
      MMJ = 0;
      TT = 0;
      MMT = 0;
      parity = true;
      TBME.emplace(name, hw, Nmax, valenceShell, maxShell);
      on_the_fly = false;
    }
    else
      throw std::runtime_error(
          "Name is incorrect or operator is not implemented");
  }

  void TwoBodyOperator::is() const
  {
    if (name == "L.L")
      std::cout << "Two-body part of L.L" << std::endl;
    else if (name == "L.S")
      std::cout << "Two-body part of L.S" << std::endl;
    else if (name == "CoMNumberOp")
      std::cout << "Two-body part of CoM number operator without 1/A"
                << std::endl;
    else if (name == "Kinetic")
      std::cout << "Two-body part of the relative kinetic energy without 1/A"
                << std::endl;
    else if (name == "R.R")
      std::cout
          << "Two-body part of the intrinsic mass radius squared without -1/A"
          << std::endl;
    else if (on_the_fly == false)
      TBME->is();
    else
      throw std::invalid_argument("This operator has not been implemented");
  }

  // ME: / na lla jja mmja mmta | O | nc llc jjc mmjc mmtc \
  // 	   \ nb llb jjb mmjb mmtb |	  | nd lld jjb mmjd mmtd /
  double TwoBodyOperator::sphericalME(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb, int nc, int llc,
                                      int jjc, int mmjc, int mmtc, int nd,
                                      int lld, int jjd, int mmjd, int mmtd,
                                      sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& racah = dbs.racah;
    auto& racah9 = dbs.racah9;

    double x = 0.0;

    if (name == "L.L") // Two-body part of L.L
    {
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
    }
    else if (name == "L.S") // Two-body part of L.S
    {
      if (na == nc && nb == nd && lla == llc && llb == lld && mmta == mmtc &&
          mmtb == mmtd)
      {
        double U1 = racah.compute(llc, 1, 2, jja, jjc, llc),
               U2 = racah.compute(1, lld, 2, jjb, jjd, 1),
               U3 = racah.compute(lld, 1, 2, jjb, jjd, lld),
               U4 = racah.compute(1, llc, 2, jja, jjc, 1),
               phase12 = ((jja - jjc) / 2 % 2 == 0) ? 1.0 : -1.0,
               phase34 = ((jjb - jjd) / 2 % 2 == 0) ? 1.0 : -1.0,
               factor =
                   (phase12 * std::sqrt(llc * (llc + 2.0) / 4.0) * U1 * U2 +
                    phase34 * std::sqrt(lld * (lld + 2.0) / 4.0) * U3 * U4),
               cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
                 cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
        x = std::sqrt(0.75) * cgc * factor;
      }
    }
    else if (name ==
             "CoMNumberOp") // Two-body part of CoM number operator without 1/A
    {
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
                 cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
        x = cgc *
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, racah9) *
                 HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, racah9) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, racah9) *
                 HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, racah9));
      }
    }
    else if (name == "Kinetic") // Two-body part of the relative kinetic energy
                                // without 1/A
    {
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
                 cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
        x = static_cast<double>(hw) * cgc *
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, racah9) *
                 (HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, racah9) -
                  HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, racah9)) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, racah9) *
                 (HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, racah9) -
                  HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, racah9))) /
            2.0;
      }
    }
    else if (name == "R.R") // Two-body part of the intrinsic mass radius
                            // squared without -1/A
    {
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
                 cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
        x = 2.0 * cgc *
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, racah9) *
                 (HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, racah9) +
                  HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, racah9)) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, racah9) *
                 (HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, racah9) +
                  HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, racah9)));
      }
    }
    else
      throw std::invalid_argument("This operator has not been implemented");

    return x;
  }

  // Antisymmetrized ME
  double TwoBodyOperator::sphericalME_AS(int na, int lla, int jja, int mmja,
                                         int mmta, int nb, int llb, int jjb,
                                         int mmjb, int mmtb, int nc, int llc,
                                         int jjc, int mmjc, int mmtc, int nd,
                                         int lld, int jjd, int mmjd, int mmtd,
                                         sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;

    if (on_the_fly == true)
      return (sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                          nc, llc, jjc, mmjc, mmtc, nd, lld, jjd, mmjd, mmtd,
                          dbs) -
              sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                          nd, lld, jjd, mmjd, mmtd, nc, llc, jjc, mmjc, mmtc,
                          dbs)) /
             4.0;
    else
      return TBME->getME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb, nc,
                         llc, jjc, mmjc, mmtc, nd, lld, jjd, mmjd, mmtd, cgcs);
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

            auto it =
                std::find(na_nb_nc_nds.begin(), na_nb_nc_nds.end(), sorted_ns);
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

            auto it =
                std::find(na_nb_nc_nds.begin(), na_nb_nc_nds.end(), sorted_ns);
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
      const std::vector<int>& phase, sam::Databases& dbs) const
  {
    /*
    auto& cgcs = dbs.cgcs;
    auto& racah = dbs.racah;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;
    */

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

    auto [coeffsArr, opsArr] = tensor.expand(dbs);

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
        me += phase[k] * sphericalME_AS(ns_orig[0], lls_orig[0], jjs_orig[0],
                                        mmjs_orig[0], mmts_orig[0], ns_orig[1],
                                        lls_orig[1], jjs_orig[1], mmjs_orig[1],
                                        mmts_orig[1], ns_orig[3], lls_orig[3],
                                        jjs_orig[3], mmjs_orig[3], mmts_orig[3],
                                        ns_orig[2], lls_orig[2], jjs_orig[2],
                                        mmjs_orig[2], mmts_orig[2], dbs);
      }

      if (std::fabs(me) < 1e-12)
        continue;

      for (int c = 0; c < coeffs_.size(); c++)
        coeffs_[c] += me * coeffsArr[i][c];
    }

    double norm = 0.0;
    for (int c = 0; c < coeffs_.size(); c++)
    {
      if (std::fabs(coeffs_[c]) < 1e-10)
      {
        coeffs_[c] = 0.0;
        continue;
      }
      norm += std::fabs(coeffs_[c]);
    }

    bool ignore = (norm < 1e-10);
    if (ignore == false)
      tensor.coeffs = coeffs_;

    return ignore;
  }

#define RESTRICT_WITH_NMAX
  std::vector<U3xU4::TwoBodyMultiShellTensor>
  TwoBodyOperator::expand(sam::Databases& dbs)
  {
    /*
    auto& cgcs = dbs.cgcs;
    auto& racah = dbs.racah;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;
    */

    if (on_the_fly == false)
      TBME->read();

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

      // auto groups_ns = group_same_shell(sorted_ns);
      auto groups_ns = group_same_shell(all_na_nb_nc_nd[i]);
      std::vector<std::vector<U3xU4::SingleShellTensor>> Tadat_lists;
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
        
        if (Tadat_choices.size() > 1)
        {
          for (const auto& tadat : Tadat_choices)
          {
            // std::cout << tadat << "\t";

            // Assert that if there are more than 1 single-shell tensor,
            // they must be all free of multiplicity.
            assert(tadat.rho0max() == 1 && tadat.eta0max() == 1);
          }
          //std::cout << std::endl;
        }

        std::vector<U3xU4::irrep> coupled_irreps;
        int Rhomax, Etamax;
        for (auto coupling_info = coupling_info_list.begin(); 
             coupling_info != coupling_info_list.end(); coupling_info++)
        {
          U3xU4::irrep irr(coupling_info->u3(), coupling_info->u4());
          auto rhomax = coupling_info->rhomax(), etamax = coupling_info->etamax();
          //const auto& [irr, rhomax, etamax] = *coupling_info;

          // std::cout << irr << "," << rhomax << "," << etamax << "\t";

          // Assert that the intershell coupling is free of multiplicity,
          // except for the last one.
          if (std::next(coupling_info) != coupling_info_list.end()) 
            assert(rhomax == 1 && etamax == 1);
          
          coupled_irreps.push_back(irr);
          Rhomax = rhomax;
          Etamax = etamax;
        }
        // std::cout << std::endl;

        // decompose into L
        std::vector<u3::physicalvector> pv3s;
        // decompose into 2S and 2T
        std::vector<u4::physicalvector> pv4s;

        if (coupled_irreps.empty()) // Tadat_choices has one element only
        {
          pv3s = u3::physicalBasis(Tadat_choices[0].T().u3());
          pv4s = u4::physicalBasis(Tadat_choices[0].T().u4());
        }
        else
        {
          pv3s = u3::physicalBasis(coupled_irreps.back().u3());
          pv4s = u4::physicalBasis(coupled_irreps.back().u4());
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
            // otherwise, it remains a multi-shell tensor
            U3xU4::TwoBodyMultiShellTensor tensor(
              Tadat_choices, coupled_irreps, 
              coupled_irreps.empty() ? Tadat_choices[0].rho0max() : Rhomax,
              coupled_irreps.empty() ? Tadat_choices[0].eta0max() : Etamax, 
              kappa0, LL0, zeta0, SS0, TT, MMT, JJ, MMJ, {}, labels);

            bool ignore =
                setTensorStrength(tensor, all_inverse[i], all_phase[i], dbs);

            //std::cout << (ignore ? "Ignore" : "Accept") << tensor << std::endl;

            if (ignore == false)
            {
              // std::cout << tensor << std::endl;
              tensors.push_back(tensor);
            }
/*            
            if (coupled_irreps.empty())
            {
              U3xU4::TwoBodyMultiShellTensor tensor(
                  Tadat_choices, coupled_irreps, Tadat_choices[0].rho0max,
                  Tadat_choices[0].eta0max, kappa0, LL0, zeta0, SS0, TT, MMT,
                  JJ, MMJ, {}, labels);
              bool ignore =
                  setTensorStrength(tensor, all_inverse[i], all_phase[i], dbs);
              if (ignore == false)
              {
                // std::cout << tensor << std::endl;
                tensors.push_back(tensor);
              }
            }
            else
            {
              U3xU4::TwoBodyMultiShellTensor tensor(
                  Tadat_choices, coupled_irreps, Rhomax, Etamax, kappa0, LL0,
                  zeta0, SS0, TT, MMT, JJ, MMJ, {}, labels);
              bool ignore =
                  setTensorStrength(tensor, all_inverse[i], all_phase[i], dbs);
              if (ignore == false)
              {
                // std::cout << tensor << std::endl;
                tensors.push_back(tensor);
              }
            }
*/
          }
        }
      }
    }

    return tensors;
  }

  // Write to binary file
  void TwoBodyOperator::writeBinary(sam::Databases& dbs)
  {
    if (valenceShell > 0)
      throw std::runtime_error("Set valenceShell to 0 for reusability");

    auto tensors = expand(dbs);

    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");

    std::string base_dir = std::string(sam_dir) + "/data/TwoBodyTensorExpansion";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name + "_hw" 
                         + std::to_string(static_cast<int>(hw));
    std::ofstream s(filename,std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for writing: " + filename);

    // Write metadata
    // Angular momentum and isospin
    s.write(reinterpret_cast<const char*>(&JJ), sizeof(int8_t));
    s.write(reinterpret_cast<const char*>(&MMJ), sizeof(int8_t));
    s.write(reinterpret_cast<const char*>(&TT), sizeof(int8_t));
    s.write(reinterpret_cast<const char*>(&MMT), sizeof(int8_t));

    // Parity
    s.write(reinterpret_cast<const char*>(&parity), sizeof(bool));

    // Configuration parameters
    s.write(reinterpret_cast<const char*>(&hw), sizeof(int8_t));
    s.write(reinterpret_cast<const char*>(&Nmax), sizeof(int8_t));
    s.write(reinterpret_cast<const char*>(&valenceShell), sizeof(int8_t));
    s.write(reinterpret_cast<const char*>(&maxShell), sizeof(int8_t));

    // Number of tensors
    size_t num_tensors = tensors.size();
    s.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

    for (auto tensor : tensors)
      tensor.writeBinary(s);
  }

  // Read from binary file
  std::vector<U3xU4::TwoBodyMultiShellTensor> TwoBodyOperator::readBinary()
  {
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");

    std::string base_dir = std::string(sam_dir) + "/data/TwoBodyTensorExpansion";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name + "_hw" 
                         + std::to_string(static_cast<int>(hw));

    std::ifstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for reading: " + filename);

    // Angular momentum and isospin
    int8_t JJ_, MMJ_, TT_, MMT_;
    s.read(reinterpret_cast<char*>(&JJ_), sizeof(int8_t));
    s.read(reinterpret_cast<char*>(&MMJ_), sizeof(int8_t));
    if (JJ_ != JJ || MMJ_ != MMJ)
      throw std::runtime_error("Angular momentum incorrect");
    s.read(reinterpret_cast<char*>(&TT_), sizeof(int8_t));
    s.read(reinterpret_cast<char*>(&MMT_), sizeof(int8_t));
    if (TT_ != TT || MMT_ != MMT)
      throw std::runtime_error("Isospin incorrect");

    // Parity
    bool parity_;
    s.read(reinterpret_cast<char*>(&parity_), sizeof(bool));
    if (parity_ != parity)
      throw std::runtime_error("Parity incorrect");

    // Configuration parameters
    int8_t hw_, Nmax_, valenceShell_, maxShell_;
    s.read(reinterpret_cast<char*>(&hw_), sizeof(int8_t));
    if (hw_ != hw)
      throw std::runtime_error("hw incorrect");
    s.read(reinterpret_cast<char*>(&Nmax_), sizeof(int8_t));
    if (Nmax_ < Nmax)
      throw std::runtime_error("File doesn't reach your Nmax");
    s.read(reinterpret_cast<char*>(&valenceShell_), sizeof(int8_t));
    if (valenceShell_ > 0)
      throw std::runtime_error("File doesn't start from the s-shell");
    s.read(reinterpret_cast<char*>(&maxShell_), sizeof(int8_t));
    if (maxShell_ < maxShell)
      throw std::runtime_error("File doesn't reach your model space");

    // Check if there are all possible tensors
    //if (2 * valenceShell )

    // Read number of tensors
    size_t num_tensors;
    s.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

    std::vector<U3xU4::TwoBodyMultiShellTensor> tensors;
    tensors.clear();
    tensors.reserve(num_tensors);

    for (size_t i = 0; i < num_tensors; ++i)
    { 
      U3xU4::TwoBodyMultiShellTensor tensor(labels);
      tensor.readBinary(s);

      if (tensor.single_shell_tensors.back().shell() > maxShell)
        continue;

      int ab_Nhw = 0, cd_Nhw = 0, ads = 0, ats = 0;
      for (const auto& Tadat : tensor.single_shell_tensors)
      {
        ab_Nhw += ((Tadat.shell() - valenceShell) * Tadat.ads);
        cd_Nhw += ((Tadat.shell() - valenceShell) * Tadat.ats);

        ads += Tadat.ads;
        ats += Tadat.ats;
      }
      if (ab_Nhw > Nmax || cd_Nhw > Nmax)
        continue;

      assert(ads == 2);
      assert(ats == 2);

      tensors.push_back(tensor);
    }
    return tensors;
  }

  Eigen::SparseMatrix<double> TwoBodyOperator::matrix(
    U3xU4::ManyNucleonBasis& basis, sam::Databases& dbs)
  {
    // expand now
    // auto tensors = expand(dbs);

    // write to binary, valenceShell must be 0 for future reuse
    // writeBinary(dbs);

    // read from binary
    auto tensors = readBinary();
    //std::cout << "Expand successfully" << std::endl;

    std::vector<U3xU4::MultiShellTensor_phys> tensors_new;
    // std::cout << "Start converting to MultiShellTensor_phys" << std::endl;
    for (auto tensor : tensors)
    {
      std::vector<U3xU4::SingleShellTensor> 
              Gamma = tensor.single_shell_tensors;
      std::vector<U3xU4::OmegaElement> Omega;
      if (!tensor.coupled_irreps.empty())
      {
        for (size_t irr_id = 0; 
             irr_id < tensor.coupled_irreps.size() - 1; irr_id++)
          Omega.push_back({1,tensor.coupled_irreps[irr_id].u3(),
                           1,tensor.coupled_irreps[irr_id].u4()});
        Omega.push_back({tensor.rhomax,tensor.coupled_irreps.back().u3(),
                         tensor.etamax,tensor.coupled_irreps.back().u4()});
      }

      U3xU4::MultiShellTensor T(Gamma,Omega);
      U3xU4::MultiShellTensor_phys T_phys(T,
        {{static_cast<unsigned short>(tensor.kappamax),
          static_cast<unsigned short>(tensor.LL),
          static_cast<unsigned short>(tensor.zetamax),
          static_cast<unsigned short>(tensor.SS)}},
          {tensor.coeffs},tensor.TT,tensor.MMT,tensor.JJ,tensor.MMJ);

      // std::cout << T_phys << std::endl;

      tensors_new.push_back(T_phys);  
    }
    // std::cout << "Finish converting to MultiShellTensor_phys" << std::endl;

    return U3xU4::MatrixElements(tensors_new,basis,dbs);
  }

} // namespace JT
