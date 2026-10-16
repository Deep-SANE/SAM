#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <omp.h>

#include <convert.h>
#include <print.h>
using utils::print;
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>
#include <so3Coeffs/SO3_Fano.h>

#include <sam.h>

#include <TwoBodyOperatorJT.h>

namespace JT
{
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
            if (!HOShellselection(na, nb, nc, nd)) continue;

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
        if (adxad_Nhw > Nmax) continue;

        for (int nc = 0; nc <= maxShell; nc++)
        {
          for (int nd = 0; nd <= maxShell; nd++)
          {
            int atxat_Nhw = 0;
            if (nc > valenceShell)
              atxat_Nhw += (nc - valenceShell);
            if (nd > valenceShell)
              atxat_Nhw += (nd - valenceShell);
            if (atxat_Nhw > Nmax) continue;

            if (!HOShellselection(na, nb, nc, nd)) continue;

            std::array<int, 4> ns = {na + 1, nb + 1, -nc - 1, -nd - 1};
            auto [sorted_ns, inverse, phase] = sort_shells(ns);
/*
            print("ns: {}\n", ns);
            print("sorted ns: {}\n", sorted_ns);
            print("inverse permutation: {}\n", inverse);
            print("phase: {}\n", phase);
*/
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
  std::tuple<std::vector<std::array<int, 4>>,
             std::vector<std::vector<std::vector<int>>>,
             std::vector<std::vector<int>>>
  TwoBodyOperator::GenerateAllCombinationsHOShells(int Nmax, int valenceShell,
                            int maxShell, int Nmax_old, int maxShell_old) const
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
        if (adxad_Nhw > Nmax) continue;

        for (int nc = 0; nc <= maxShell; nc++)
        {
          for (int nd = 0; nd <= maxShell; nd++)
          {
            int atxat_Nhw = 0;
            if (nc > valenceShell)
              atxat_Nhw += (nc - valenceShell);
            if (nd > valenceShell)
              atxat_Nhw += (nd - valenceShell);
            if (atxat_Nhw > Nmax) continue;

            if (!HOShellselection(na, nb, nc, nd, Nmax_old, maxShell_old)) continue;

            std::array<int, 4> ns = {na + 1, nb + 1, -nc - 1, -nd - 1};
            auto [sorted_ns, inverse, phase] = sort_shells(ns);
/*
            print("ns: {}\n", ns);
            print("sorted ns: {}\n", sorted_ns);
            print("inverse permutation: {}\n", inverse);
            print("phase: {}\n", phase);
*/
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

  // Set strengths for U3xU4::TwoBodyMultiShellTensor
  // It returns a flag, which indicate whether this tensor can be ignored.
  bool TwoBodyOperator::setTensorStrength(
      U3xU4::TwoBodyMultiShellTensor& tensor,
      const std::vector<std::vector<int>>& inverse,
      const std::vector<int>& phase,
      U3xU4::SSTDecomp_phys_db& db) const
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

    std::vector<double> coeffs(tensor.rhomax() * tensor.etamax() * 
                                tensor.kappamax() * tensor.zetamax(), 0.0);

    auto exp = tensor.expand(db);

    for (size_t i = 0; i < exp.rows(); i++)
    {
      std::array<int, 4> ns, lls, jjs, mmjs, mmts;
      exp.fill_labels(i, 0, ns, lls, jjs, mmjs, mmts);

      double me = 0.0;

      for (int k = 0; k < inverse.size(); k++)
      {
        std::array<int, 4> ns_orig = inverse_permutation(ns, inverse[k]),
                           lls_orig = inverse_permutation(lls, inverse[k]),
                           jjs_orig = inverse_permutation(jjs, inverse[k]),
                           mmjs_orig = inverse_permutation(mmjs, inverse[k]),
                           mmts_orig = inverse_permutation(mmts, inverse[k]);

        // me needs computing with normal ordering
        double me_;
        if (on_the_fly)
          me_ = sphericalME_AS(ns_orig[0], lls_orig[0], jjs_orig[0], mmjs_orig[0], mmts_orig[0],
                               ns_orig[1], lls_orig[1], jjs_orig[1], mmjs_orig[1], mmts_orig[1],
                               ns_orig[3], lls_orig[3], jjs_orig[3], mmjs_orig[3], mmts_orig[3],
                               ns_orig[2], lls_orig[2], jjs_orig[2], mmjs_orig[2], mmts_orig[2]);
        else
        {
          if (TBME_scalar)
            me_ = TBME_scalar->getME(ns_orig[0], lls_orig[0], jjs_orig[0], mmjs_orig[0], mmts_orig[0],
                               ns_orig[1], lls_orig[1], jjs_orig[1], mmjs_orig[1], mmts_orig[1],
                               ns_orig[3], lls_orig[3], jjs_orig[3], mmjs_orig[3], mmts_orig[3],
                               ns_orig[2], lls_orig[2], jjs_orig[2], mmjs_orig[2], mmts_orig[2]) / 4.0;
          else
            me_ = TBME_general->getME(ns_orig[0], lls_orig[0], jjs_orig[0], mmjs_orig[0], mmts_orig[0],
                               ns_orig[1], lls_orig[1], jjs_orig[1], mmjs_orig[1], mmts_orig[1],
                               ns_orig[3], lls_orig[3], jjs_orig[3], mmjs_orig[3], mmts_orig[3],
                               ns_orig[2], lls_orig[2], jjs_orig[2], mmjs_orig[2], mmts_orig[2]) / 4.0;
        }

        me += phase[k] * me_;
      }

      if (std::fabs(me) < 1e-10) continue;

      const double* row = exp.coeff_row(i);
      for (int c = 0; c < coeffs.size(); c++)
        coeffs[c] += me * row[c];
    }

    double norm = 0.0;
    for (int c = 0; c < coeffs.size(); c++)
    {
      if (std::fabs(coeffs[c]) < 1e-10)
      {
        coeffs[c] = 0.0;
        continue;
      }
      norm += std::fabs(coeffs[c]);
    }

    bool ignore = (norm < 1e-10);
    if (ignore == false) tensor.setCoeffs(coeffs);

    return ignore;
  }

  // Set strengths for U3xU4::MultiShellTensor_phys
  void TwoBodyOperator::setTensorStrength(
      U3xU4::MultiShellTensor_phys& tensor,
      const std::vector<std::vector<int>>& inverse,
      const std::vector<int>& phase,
      U3xU4::SSTDecomp_phys_db& db) const
  {
    // std::cout << "Gammas count: " << tensor.Gammas().size() << std::endl;
    /* if (tensor.Gammas().size() == 1)
    {
      print("Inverse: {}\n", inverse);
      print("Phase: {}\n", phase);
    } */

    std::vector<std::tuple<uint8_t,              // kappamax
                           HO::qn_t,             // LL
                           uint8_t,              // zetamax
                           HO::qn_t,             // SS
                           HO::qn_t,             // TT
                           std::vector<double>>> LLSSTTstrengths;

    // decompose into L
    u3::physicalvectors pv3s = u3::physicalBasis(tensor.getLastIrrep().u3());

    // decompose into 2S and 2T
    u4::physicalvectors pv4s = u4::physicalBasis(tensor.getLastIrrep().u4());

    for (const auto& pv3 : pv3s)
    {
      int kappa0 = pv3.kappa(), LL0 = 2 * pv3.L();

      if (LLselection(LL0) == false) continue;

      for (const auto& pv4 : pv4s)
      {
        int zeta0 = pv4.zeta(), SS0 = pv4.SS(), TT0 = pv4.TT();

        if (SSselection(SS0) == false) continue;

        // if the operator has definite isospin assignment
        if (TT() >= 0 && TT() != TT0) continue;

        // couple LL and SS to JJ
        if (u2::mult(LL0, SS0, JJ()) == 0) continue;

        std::vector<double> strengths(tensor.alphamax() * kappa0 * zeta0, 0.0);

        if (on_the_fly)
        {
          auto exp = tensor.expand(LL0, SS0, TT0, db);

          for (size_t i = 0; i < exp.rows(); i++)
          {
            assert(exp.nops == 4);

            std::array<int, 4> ns, lls, jjs, mmjs, mmts;
            exp.fill_labels(i, 0, ns, lls, jjs, mmjs, mmts);

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
                                          mmjs_orig[2], mmts_orig[2]);
            }

            if (std::fabs(me) < 1e-10) continue;

            for (int c = 0; c < strengths.size(); c++)
              strengths[c] += me * exp.coeff_row(i)[c];
          }
        }
        else
        {
          if (TBME_scalar)
            strengths = tensor.getStrengths(LL0, SS0, TT0, *TBME_scalar, inverse, phase, db);
          else
            strengths = tensor.getStrengths(LL0, SS0, TT0, *TBME_general, inverse, phase, db);
        }

        double norm = 0.0;
        for (auto& s : strengths)
        {
          if (std::fabs(s) < 1e-10)
          {
            s = 0.0;
            continue;
          }
          norm += std::fabs(s);
        }

        if (std::fabs(norm) > 1e-8)
          LLSSTTstrengths.emplace_back(kappa0, LL0, zeta0, SS0, TT0, strengths);
      }
    }

    tensor.setLLSSTTstrengths(LLSSTTstrengths);
  }

  // Selection rules
  bool TwoBodyOperator::HOShellselection(int na, int nb, int nc, int nd) const
  {
    if (parity_ == 0)
    {
      int shift = na + nb - nc - nd;
      if (shift % 2 != 0) return false;

      if (name_ == "CoMNumberOp")
        if (shift != 0) return false;
      else if (name_ == "Rrel" || name_ == "Kinetic")
        if (shift != 0 && abs(shift) != 2) return false;
    }
    
    return true;
  }

  bool TwoBodyOperator::HOShellselection(int na, int nb, int nc, int nd,
                                         int Nmax_old, int maxShell_old) const
  {
    // Check if already exists
    if (na <= maxShell_old && nb <= maxShell_old &&
        nc <= maxShell_old && nd <= maxShell_old &&
        (na + nb) <= Nmax_old && (nc + nd) <= Nmax_old)
      return false;

    if (parity_ == 0)
    {
      // Check shift
      int shift = na + nb - nc - nd;
      if (shift % 2 != 0) return false;

      if (name_ == "CoMNumberOp")
        if (shift != 0) return false;
      else if (name_ == "Rrel" || name_ == "Kinetic")
        if (shift != 0 && abs(shift) != 2) return false;
    }
    
    return true;
  }

  bool TwoBodyOperator::LLselection(const int& LL) const
  {
    if (name_ == "L.L" || name_ == "L.S" || name_ == "Rrel" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (LL != 0) return false;
    }
    
    return true;
  }
  
  bool TwoBodyOperator::SSselection(const int& SS) const
  {
    if (name_ == "L.L" || name_ == "L.S" || name_ == "Rrel" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (SS != 0) return false;
    }
    
    return true;
  }

  bool TwoBodyOperator::U3selection(const u3::irrep& irr) const
  {
    if (name_ == "L.L" || name_ == "L.S" || name_ == "CoMNumberOp")
    {
      if (irr != u3::irrep(0, 0, 0)) return false;
    }
    else if (name_ == "Rrel" || name_ == "Kinetic")
    {
      if (irr != u3::irrep(0, 0, 0) &&
          irr != u3::irrep(2, 0, 0) &&
          irr != u3::irrep(2, 2, 0)) 
        return false;
    }
    else if (known_potentials.count(name_))
    {
      if (irr.n13() % 2 != 0) return false;
    }
    
    return true;
  }

  bool TwoBodyOperator::U4selection(const u4::irrep& irr) const
  {
    if (name_ == "L.L" || name_ == "L.S" || name_ == "Rrel" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (irr != u4::irrep(0, 0, 0, 0)) return false;
    }
    else if (known_potentials.count(name_))
    {
      if (irr != u4::irrep(0, 0, 0, 0) &&
          irr != u4::irrep(2, 1, 1, 0) &&
          irr != u4::irrep(2, 2, 0, 0) &&
          irr != u4::irrep(4, 2, 2, 0)) 
        return false;
    }
    else if (name_ == "GamowTellerMinus_2B_N3LO_EM500_bare")
    {
      if (irr != u4::irrep(2, 1, 1, 0) &&
          irr != u4::irrep(2, 2, 0, 0) &&
          irr != u4::irrep(4, 2, 2, 0)) 
        return false;
    }
    
    return true;
  }

} // namespace JT
