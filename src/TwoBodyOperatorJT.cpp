#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>

#include <qcx_convert.h>
#include <qcx_print.h>
using qcx_utils::print;
#include <so3Coeffs/SO3_CGC.h>

#include <Databases.h>
#include <TwoBodyOperatorJT.h>

namespace JT
{
  TwoBodyOperator::TwoBodyOperator(const std::string& name, int hw, int Nmax, 
      int valenceShell, int maxShell, const HO::Table_nljmjmt& labels)
      : labels_(labels)
  {
    if (maxShell > 14)
      throw std::out_of_range(
          "Sorry, we only have maxShell up to 14 at the moment");

    hw_ = (hw >= 0) ? qcx_utils::ito8(hw) : 20;
    Nmax_ = (Nmax >= 0) ? qcx_utils::ito8(Nmax) : 14;
    valenceShell_ = (valenceShell > 0) ? qcx_utils::ito8(valenceShell) : 0;
    maxShell_ = (maxShell >= 0) ? qcx_utils::ito8(maxShell) : 14;

    name_ = name;
    JJ_ = 0;
    MMJ_ = 0;
    TT_ = 0;
    MMT_ = 0;
    parity_ = 1;

    if (known_scalar_ops.count(name)) 
    {
      on_the_fly = true;
    }
    else if (known_potentials.count(name))
    {
      TT_ = -1;
      TBME.emplace(name, hw, Nmax, valenceShell, maxShell);
      on_the_fly = false;
    }
    else
      throw std::runtime_error(
          "Name is incorrect or operator is not implemented");
  }

  void TwoBodyOperator::is() const
  {
    if (name_ == "L.L")
      std::cout << "Two-body part of L.L" << std::endl;
    else if (name_ == "L.S")
      std::cout << "Two-body part of L.S" << std::endl;
    else if (name_ == "CoMNumberOp")
      std::cout << "Two-body part of CoM number operator without 1/A"
                << std::endl;
    else if (name_ == "Kinetic")
      std::cout << "Two-body part of the relative kinetic energy without 1/A"
                << std::endl;
    else if (name_ == "Rrel")
      std::cout
          << "Two-body part of the intrinsic mass radius squared without -1/A"
          << std::endl;
    else if (on_the_fly == false)
      std::cout << "Two-body potential with " << name_ << std::endl;
      //TBME->is();
    else
      throw std::invalid_argument("This operator has not been implemented");
  }

  std::ostream& operator<<(std::ostream& s, const TwoBodyOperator& x)
  {
    if (x.name_ == "L.L")
      s << "Two-body part of L.L";
    else if (x.name_ == "L.S")
      s << "Two-body part of L.S";
    else if (x.name_ == "CoMNumberOp")
      s << "Two-body part of CoM number operator without 1/A";
    else if (x.name_ == "Kinetic")
      s << "Two-body part of the relative kinetic energy without 1/A";
    else if (x.name_ == "R.R")
      s << "Two-body part of the intrinsic mass radius squared without -1/A";
    else if (x.on_the_fly == false)
      s << "Two-body potential with " << x.name_;
      //TBME->is();
    else
      throw std::invalid_argument("This operator has not been implemented");

    return s;
  }

  std::string TwoBodyOperator::dirname() const
  {
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");
      
    std::string dir = std::string(sam_dir) + "/data/TwoBodyOperatorJT/" + name_;
    std::filesystem::create_directories(dir);

    return dir;
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

    if (name_ == "L.L") // Two-body part of L.L
    {
      if (na == nc && nb == nd && lla == llc && llb == lld && mmta == mmtc &&
          mmtb == mmtd)
      {
        double phase = ((jja + jjb + jjc + jjd) / 2 % 2 == 0) ? 1.0 : -1.0,
               factor = std::sqrt(llc * (llc + 2.0) * lld * (lld + 2.0)) / 4.0,
               U1 = racah.get({llc, 1, 2, jja, jjc, llc}),
               U2 = racah.get({lld, 1, 2, jjb, jjd, lld}), cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = 2 * phase * factor * U1 * U2 * cgc;
      }
    }
    else if (name_ == "L.S") // Two-body part of L.S
    {
      if (na == nc && nb == nd && lla == llc && llb == lld && mmta == mmtc &&
          mmtb == mmtd)
      {
        double U1 = racah.get({llc, 1, 2, jja, jjc, llc}),
               U2 = racah.get({1, lld, 2, jjb, jjd, 1}),
               U3 = racah.get({lld, 1, 2, jjb, jjd, lld}),
               U4 = racah.get({1, llc, 2, jja, jjc, 1}),
               phase12 = ((jja - jjc) / 2 % 2 == 0) ? 1.0 : -1.0,
               phase34 = ((jjb - jjd) / 2 % 2 == 0) ? 1.0 : -1.0,
               factor =
                   (phase12 * std::sqrt(llc * (llc + 2.0) / 4.0) * U1 * U2 +
                    phase34 * std::sqrt(lld * (lld + 2.0) / 4.0) * U3 * U4),
               cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = std::sqrt(0.75) * cgc * factor;
      }
    }
    else if (name_ ==
             "CoMNumberOp") // Two-body part of CoM number operator without 1/A
    {
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = cgc *
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                 HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, dbs) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                 HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, dbs));
      }
    }
    else if (name_ == "Kinetic") // Two-body part of the relative kinetic energy
                                // without 1/A
    {
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = static_cast<double>(hw_) * cgc *
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                 (HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, dbs) -
                  HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, dbs)) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                 (HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, dbs) -
                  HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, dbs))) /
            2.0;
      }
    }
    else if (name_ == "Rrel") // Two-body part of the intrinsic mass radius
                            // squared without -1/A
    {
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = HO::oscilatorLength2(hw_, -1) * cgc *
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                 (HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, dbs) +
                  HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, dbs)) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                 (HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, dbs) +
                  HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, dbs)));
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
    if (on_the_fly == true)
    {
      return (sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                          nc, llc, jjc, mmjc, mmtc, nd, lld, jjd, mmjd, mmtd,
                          dbs) -
              sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                          nd, lld, jjd, mmjd, mmtd, nc, llc, jjc, mmjc, mmtc,
                          dbs)) / 4.0;
    }
    else
    {
      return TBME->getME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb, 
                         nc, llc, jjc, mmjc, mmtc, nd, lld, jjd, mmjd, mmtd, 
                         dbs) / 4.0;
    }
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
      const std::vector<int>& phase, sam::Databases& dbs) const
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

    auto [coeffsArr, opsArr] = tensor.expand(dbs);

    for (size_t i = 0; i < coeffsArr.size(); i++)
    {
      std::array<int, 4> ns, lls, jjs, mmjs, mmts;

      int j = 0;
      for (const auto& op : opsArr[i])
      {
        std::visit(
            [&](auto&& o)
            {
              //std::cout << o << "\t";
              std::tie(ns[j], lls[j], jjs[j], mmjs[j], mmts[j]) = o.getLabels();
            },
            op);
        j++;
      }
      //std::cout << "\n";

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

      if (std::fabs(me) < 1e-10) continue;

      for (int c = 0; c < coeffs.size(); c++)
        coeffs[c] += me * coeffsArr[i][c];
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

  // Selection rules
  bool TwoBodyOperator::LLselection(const int& LL) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" || name_ == "Rrel" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (LL != 0) flag = false;
    }
    
    return flag;
  }
  
  bool TwoBodyOperator::SSselection(const int& SS) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" || name_ == "Rrel" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (SS != 0) flag = false;
    }
    
    return flag;
  }

  bool TwoBodyOperator::U3selection(const u3::irrep& irr) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" || name_ == "CoMNumberOp")
    {
      if (irr != u3::irrep(0, 0, 0)) flag = false;
    }
    else if (name_ == "Rrel" || name_ == "Kinetic")
    {
      if (irr != u3::irrep(0, 0, 0) &&
          irr != u3::irrep(2, 0, 0) &&
          irr != u3::irrep(2, 2, 0)) 
        flag = false;
    }
    else if (known_potentials.count(name_))
    {
      if (irr.n13() % 2 != 0) flag = false;
    }
    
    return flag;
  }

  bool TwoBodyOperator::U4selection(const u4::irrep& irr) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" || name_ == "Rrel" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (irr != u4::irrep(0, 0, 0, 0)) flag = false;
    }
    
    return flag;
  }

  // Set strengths for U3xU4::MultiShellTensor_phys
  void TwoBodyOperator::setTensorStrength(
      U3xU4::MultiShellTensor_phys& tensor,
      const std::vector<std::vector<int>>& inverse,
      const std::vector<int>& phase, sam::Databases& dbs) const
  {
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
        // if (JJ() < std::fabs(LL0 - SS0) || JJ() > (LL0 + SS0)) continue;
        if (u2::mult(LL0, SS0, JJ()) == 0) continue;

        std::vector<double> strengths(tensor.alphamax() * kappa0 * zeta0, 0.0);

        auto [coeffsArr, opsArr] = tensor.expand(LL0, SS0, TT0, dbs);

        for (size_t i = 0; i < coeffsArr.size(); i++)
        {
          assert(opsArr[i].size() == 4);

          std::array<int, 4> ns, lls, jjs, mmjs, mmts;

          int j = 0;
          for (const auto& op : opsArr[i])
          {
            std::visit([&](auto&& o)
            {
              std::tie(ns[j], lls[j], jjs[j], mmjs[j], mmts[j]) = o.getLabels();
            }, op);
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

          if (std::fabs(me) < 1e-10) continue;

          for (int c = 0; c < strengths.size(); c++)
            strengths[c] += me * coeffsArr[i][c];
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

#define RESTRICT_WITH_NMAX
#define RESTRICT_JJ
  U3xU4::TwoBodyMultiShellTensors 
    TwoBodyOperator::expand_old(sam::Databases& dbs)
  {
    if (on_the_fly == false)
      TBME->read();

    U3xU4::TwoBodyMultiShellTensors tensors;

#ifdef RESTRICT_WITH_NMAX
    auto [all_na_nb_nc_nd, all_inverse, all_phase] =
        GenerateAllCombinationsHOShells(Nmax_, valenceShell_, maxShell_);
#else
    auto all_na_nb_nc_nd = GenerateAllCombinationsHOShells(maxShell_);
#endif

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
          pv3s = std::move(u3::physicalBasis(Gammas[0].u3()));
          pv4s = std::move(u4::physicalBasis(Gammas[0].u4()));
        }
        else
        {
          pv3s = std::move(u3::physicalBasis(coupled_irreps.back().u3()));
          pv4s = std::move(u4::physicalBasis(coupled_irreps.back().u4()));
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
              kappa0, LL0, zeta0, SS0, TT0, MMT_, JJ_, MMJ_, {}, labels_);

            bool ignore =
                setTensorStrength(tensor, all_inverse[i], all_phase[i], dbs);

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
                kappa0, LL0, zeta0, SS0, TT0, MMT_, JJ0, MMJ_, {}, labels_);

              bool ignore =
                setTensorStrength(tensor, all_inverse[i], all_phase[i], dbs);

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

    return tensors;
  }

  U3xU4::MultiShellTensors_phys 
    TwoBodyOperator::expand_new(sam::Databases& dbs)
  {
    if (on_the_fly == false)
      TBME->read();

    U3xU4::MultiShellTensors_phys tensors;

#ifdef RESTRICT_WITH_NMAX
    auto [all_na_nb_nc_nd, all_inverse, all_phase] =
        GenerateAllCombinationsHOShells(Nmax_, valenceShell_, maxShell_);
#else
    auto all_na_nb_nc_nd = GenerateAllCombinationsHOShells(maxShell_);
#endif

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

      for (const auto& [Gammas, Omegas] : all_results)
      {
        U3xU4::MultiShellTensor_phys tensor(Gammas, Omegas, 
                                            MMT_, JJ_, MMJ_, labels_);

        if (U3selection(tensor.getLastIrrep().u3()) == false) continue;
        if (U4selection(tensor.getLastIrrep().u4()) == false) continue;

        setTensorStrength(tensor, all_inverse[i], all_phase[i], dbs);

        if (!tensor.LLSSTTstrengths().empty())
          tensors.push_back(std::move(tensor));
      }
    }

    return tensors;
  }

  U3xU4::MultiShellTensors_phys 
    TwoBodyOperator::expand(sam::Databases& dbs)
  {
    std::string base_dir = dirname() + "/Expansion";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_ + "_hw" 
                         + std::to_string(static_cast<int>(hw_));

    // if this file exists, returns immediately
    if (std::filesystem::exists(filename))
    {
      return readBinary(filename, dbs);
    }
    else
    {
      auto tensors = expand_new(dbs);
      writeBinary(filename, tensors);
      return tensors;
    }
  }

  // Write to binary file
  void TwoBodyOperator::writeBinary(const std::string& filename, 
                     const U3xU4::MultiShellTensors_phys& tensors) const
  {
    std::cout << "Start writing " << name_ << " to binary file" << std::endl;

    if (valenceShell_ > 0)
      throw std::runtime_error("Set valenceShell to 0 for reusability");

    if (Nmax_ != maxShell_)
      throw std::runtime_error("Set Nmax and maxShell equal");

    std::ofstream s(filename,std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for writing: " + filename);

    qcx_utils::writeBinary(s, maxShell_);

    // Number of tensors
    size_t n_tensors = tensors.size();
    qcx_utils::writeBinary(s, n_tensors);

    for (const auto& tensor : tensors) tensor.writeBinary(s);

    std::cout << "Finish writing to binary file" << std::endl;
  }

  // Read from binary file
  U3xU4::MultiShellTensors_phys TwoBodyOperator::readBinary(
    const std::string& filename, sam::Databases& dbs)
  {
    std::ifstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for reading: " + filename);

    HO::qn_t maxShell;
    qcx_utils::readBinary(s, maxShell);

    std::cout << "Start reading " << name_ 
              << " from binary file with capacity maxShell = " 
              << static_cast<int>(maxShell) << std::endl;
    
    if (maxShell_ - 2 * valenceShell_ > maxShell /* Nmax */)
    {
      std::cout << "File doesn't reach your Nmax, so it will be expanded now"
                << std::endl;
      valenceShell_ = 0;
      Nmax_ = maxShell_;

      auto tensors = expand_new(dbs);
      writeBinary(filename, tensors);
      return tensors;
    }

    // Read number of tensors
    size_t n_tensors;
    qcx_utils::readBinary(s, n_tensors);

    U3xU4::MultiShellTensors_phys tensors;
    tensors.reserve(n_tensors);

    for (size_t i = 0; i < n_tensors; ++i)
    { 
      U3xU4::MultiShellTensor_phys tensor(labels_);
      tensor.readBinary(s);

      int ab_Nhw = 0, cd_Nhw = 0, ads = 0, ats = 0;
      for (const auto& G : tensor.Gammas())
      {
        ab_Nhw += ((G.shell() - valenceShell_) * G.num_ads());
        cd_Nhw += ((G.shell() - valenceShell_) * G.num_ats());

        ads += G.num_ads();
        ats += G.num_ats();
      }
      if (ab_Nhw > Nmax_ || cd_Nhw > Nmax_) continue;

      assert(ads == 2);
      assert(ats == 2);

      tensors.push_back(std::move(tensor));
    }

    std::cout << "Finish reading from binary file" << std::endl;

    return tensors;
  }

  // Matrix representation
  Eigen::SparseMatrix<double> TwoBodyOperator::matrix(
    const U3xU4::ManyNucleonBasis& basis, sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return U3xU4::MatrixElements(tensors, basis, dbs);
  }
  Eigen::SparseMatrix<double> TwoBodyOperator::matrix(
    const U3xU4::ManyNucleonBasis& bras, 
    const U3xU4::ManyNucleonBasis& kets, sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return U3xU4::MatrixElements(tensors, bras, kets, dbs);
  }

  // Tensor Analyzers
  std::vector<std::pair<u3::irrep, sam::strength_stats>>
    TwoBodyOperator::U3_tensor_analyzer_write(sam::Databases& dbs)
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_U3.dat";

    auto tensors = expand(dbs);
    return sam::U3_tensor_analyzer(tensors, filename);
  }
  
  std::vector<std::pair<u3::irrep, sam::strength_stats>>
    TwoBodyOperator::U3_tensor_analyzer(sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return sam::U3_tensor_analyzer(tensors);
  }

  std::vector<std::pair<u4::irrep, sam::strength_stats>>
    TwoBodyOperator::U4_tensor_analyzer_write(sam::Databases& dbs)
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_U4.dat";

    auto tensors = expand(dbs);
    return sam::U4_tensor_analyzer(tensors, filename);
  }
  
  std::vector<std::pair<u4::irrep, sam::strength_stats>>
    TwoBodyOperator::U4_tensor_analyzer(sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return sam::U4_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::LL_tensor_analyzer_write(sam::Databases& dbs)
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_LL.dat";

    auto tensors = expand(dbs);
    return sam::LL_tensor_analyzer(tensors, filename);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::LL_tensor_analyzer(sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return sam::LL_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::SS_tensor_analyzer_write(sam::Databases& dbs)
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_SS.dat";

    auto tensors = expand(dbs);
    return sam::SS_tensor_analyzer(tensors, filename);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::SS_tensor_analyzer(sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return sam::SS_tensor_analyzer(tensors);
  }
} // namespace JT
