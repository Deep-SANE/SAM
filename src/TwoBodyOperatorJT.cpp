#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>
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
  TwoBodyOperator::TwoBodyOperator(const std::string& name, int hw, int Nmax, 
                                   int valenceShell, int maxShell,
                                   const HO::Table_nljmjmt& HO_labels,
                                   const U3xU4::Table_gelfand& gelfand_labels)
      : HO_labels_(HO_labels), gelfand_labels_(gelfand_labels)
  {
    if (maxShell > 14)
      throw std::out_of_range("Sorry, we only have maxShell up to 14");

    hw_ = (hw >= 0) ? qcx_utils::ito8(hw) : 20;
    Nmax_ = (Nmax >= 0) ? qcx_utils::ito8(Nmax) : 14;
    valenceShell_ = (valenceShell > 0) ? qcx_utils::ito8(valenceShell) : 0;
    maxShell_ = (maxShell >= 0) ? qcx_utils::ito8(maxShell) : 14;

    name_ = name;
    JJ_ = 0;
    MMJ_ = 0;
    TT_ = 0;
    MMT_ = 0;
    parity_ = 0;
    symmetric_ = true;

    if (known_scalar_ops.count(name)) 
    {
      on_the_fly = true;
    }
    else if (known_potentials.count(name))
    {
      TT_ = -1;
      TBME_scalar.emplace(name, hw, Nmax, valenceShell, maxShell);
      on_the_fly = false;
    }
    else if (known_general_ops.count(name))
    {
      TBME_general.emplace(name, hw, Nmax, valenceShell, maxShell);
      on_the_fly = false;
      symmetric_ = false;

      if (name.find("GamowTellerMinus") != std::string::npos)
      {
        JJ_ = 2;
        TT_ = 2;
        MMT_ = -2;
      }
    }
    else // Unknown nuclear potentials, e.g., from NO2B truncation
    {
#ifdef USE_MPI
      int mpi_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

      if (mpi_rank == 0)
#endif
      std::cout << "WARNING: This operator isn't known" << std::endl;

      TT_ = -1;
      on_the_fly = false;

      TBME_scalar.emplace(name, hw, Nmax, valenceShell, maxShell);
      // TBME_general.emplace(name, hw, Nmax, valenceShell, maxShell);   
    }
  }

  void TwoBodyOperator::is() const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
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
        std::cout << "Two-body operator: " << name_ << std::endl;
    }

    if (name_ != "L.L" && name_ != "L.S" && name_ != "CoMNumberOp" &&
        name_ != "Kinetic" && name_ != "Rrel" && on_the_fly == true)
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
    else if (x.name_ == "Rrel")
      s << "Two-body part of the intrinsic mass radius squared without -1/A";
    else if (x.on_the_fly == false)
      s << "Two-body operator: " << x.name_;
    else
      throw std::invalid_argument("This operator has not been implemented");

    return s;
  }

  std::string TwoBodyOperator::dirname() const
  {
#ifdef SAM_DATA_DIR
    const char* sam_data = std::getenv("SAM_DATA_DIR");    
    if (!sam_data)
      throw std::runtime_error("Environment variable SAM_DATA_DIR isn't set");

    std::string dir = std::string(sam_data) + "/TwoBodyOperatorJT/" + name_;
#else
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR isn't set");
      
    std::string dir = std::string(sam_dir) + "/data/TwoBodyOperatorJT/" + name_;
#endif
    
    std::filesystem::create_directories(dir);
    return dir;
  }

  // ME: / na lla jja mmja mmta | O | nc llc jjc mmjc mmtc \
  // 	   \ nb llb jjb mmjb mmtb |	  | nd lld jjb mmjd mmtd /
  double TwoBodyOperator::sphericalME(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb, int nc, int llc,
                                      int jjc, int mmjc, int mmtc, int nd,
                                      int lld, int jjd, int mmjd, int mmtd) const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah = qcx_utils::racahs();
    auto& racah9 = qcx_utils::fanos();

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
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                 HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                 HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd));
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
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                 (HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd) -
                  HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd)) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                 (HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd) -
                  HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd))) /
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
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                 (HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd) +
                  HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd)) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                 (HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd) +
                  HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd)));
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
                                         int lld, int jjd, int mmjd, int mmtd) const
  {
    assert(on_the_fly == true);
    return (sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                        nc, llc, jjc, mmjc, mmtc, nd, lld, jjd, mmjd, mmtd) -
            sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                        nd, lld, jjd, mmjd, mmtd, nc, llc, jjc, mmjc, mmtc)) / 4.0;
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

#pragma omp parallel
{
  sam::initialize_thread();

  // U3xU4::MultiShellTensors_phys tensors_tl;

#pragma omp for schedule(dynamic,1)
    for (size_t k = 0; k < tensors.size(); ++k)
    {
      auto& [i, tensor] = tensors[k];
      setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);

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

    // (II) Run tensor expansion via OpenMP + MPI

    size_t my_count = (mpi_rank < (int)tensors_rank.size())
                      ? (tensors_rank.size() - mpi_rank + mpi_size - 1) / mpi_size
                      : 0;
    std::cerr << "Rank " << mpi_rank << ": processing " << my_count
              << " / " << tensors_rank.size() << " tensors" << std::endl;

    // Shared cache of single-shell tensor decompositions (thread-safe)
    U3xU4::SSTDecomp_phys_db decompdb(maxShell_);

#pragma omp parallel
{
  sam::initialize_thread();

#pragma omp for schedule(dynamic,1)
    for (size_t it = mpi_rank; it < tensors_rank.size(); it += mpi_size)
    {
      auto& [i, tensor] = tensors_rank[it];

      setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);

      size_t local_idx = (it - (size_t)mpi_rank) / (size_t)mpi_size;
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

    // (II) Run tensor expansion via OpenMP + MPI

    size_t my_count = (mpi_rank < (int)tensors_rank.size())
                      ? (tensors_rank.size() - mpi_rank + mpi_size - 1) / mpi_size
                      : 0;
    std::cerr << "Rank " << mpi_rank << ": processing " << my_count
              << " / " << tensors_rank.size() << " tensors" << std::endl;

    // Shared cache of single-shell tensor decompositions (thread-safe)
    U3xU4::SSTDecomp_phys_db decompdb(maxShell_);

#pragma omp parallel
{
  sam::initialize_thread();

#pragma omp for schedule(dynamic,1)
    for (size_t it = mpi_rank; it < tensors_rank.size(); it += mpi_size)
    {
      auto& [i, tensor] = tensors_rank[it];

      setTensorStrength(tensor, all_inverse[i], all_phase[i], decompdb);

      size_t local_idx = (it - (size_t)mpi_rank) / (size_t)mpi_size;
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

  // Write to binary file
  void TwoBodyOperator::writeBinary(const std::string& filename, 
                     const U3xU4::MultiShellTensors_phys& tensors) const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::cerr << "Start writing " << name_ 
              << " to binary file on rank " << mpi_rank << std::endl;
#else
    std::cout << "Start writing " << name_ << " to binary file" << std::endl;
#endif

    if (valenceShell_ > 0) throw std::runtime_error("Set valenceShell to 0 for reusability");

    // if (Nmax_ != maxShell_) hrow std::runtime_error("Set Nmax and maxShell equal");

    std::ofstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for writing: " + filename);

    qcx_utils::writeBinary(s, maxShell_);

    // Number of tensors
    size_t n_tensors = tensors.size();
    qcx_utils::writeBinary(s, n_tensors);

    for (const auto& tensor : tensors) tensor.writeBinary(s);

#ifdef USE_MPI
    std::cerr << "Finish writing to binary file on rank " << mpi_rank << std::endl;
#else
    std::cout << "Finish writing to binary file" << std::endl;
#endif
  }

  // Read from binary file
  U3xU4::MultiShellTensors_phys TwoBodyOperator::readBinary(
    const std::string& filename)
  {
    std::ifstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for reading: " + filename);

    HO::qn_t maxShell;
    qcx_utils::readBinary(s, maxShell);

#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::cerr << "Start reading " << name_ 
              << " from binary file with capacity maxShell = " 
              << static_cast<int>(maxShell) << " on rank " << mpi_rank << std::endl;
#else
    std::cout << "Start reading " << name_ 
              << " from binary file with capacity maxShell = " 
              << static_cast<int>(maxShell) << std::endl;
#endif

    // Read number of tensors
    size_t n_tensors;
    qcx_utils::readBinary(s, n_tensors);

    U3xU4::MultiShellTensors_phys tensors;
    tensors.reserve(n_tensors);

    for (size_t i = 0; i < n_tensors; ++i)
    { 
      U3xU4::MultiShellTensor_phys tensor(HO_labels_);
      tensor.readBinary(s);

      int ab_Nhw = 0, cd_Nhw = 0, ads = 0, ats = 0;
      for (const auto& G : tensor.Gammas())
      {
        ab_Nhw += ((G.shell() - valenceShell_) * G.num_ads());
        cd_Nhw += ((G.shell() - valenceShell_) * G.num_ats());

        ads += G.num_ads();
        ats += G.num_ats();
      }
      
      // number of quanta shifted by a+'s and a-'s cannot exceed Nmax
      if (ab_Nhw > Nmax_ || cd_Nhw > Nmax_ || abs(ab_Nhw - cd_Nhw) > Nmax_) 
        continue;

      assert(ads == 2);
      assert(ats == 2);

      tensors.push_back(std::move(tensor));
    }

#ifdef USE_MPI
    std::cerr << "Finish reading to binary file on rank " << mpi_rank << std::endl;
#else
    std::cout << "Finish reading to binary file" << std::endl;
#endif

    if ((maxShell_ > maxShell) or (maxShell_ - 2 * valenceShell_ > maxShell /* = Nmax */))
    {
#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
      std::cout << "File doesn't reach your maxShell/Nmax, it will be expanded now" << std::endl;

      valenceShell_ = 0;
      Nmax_ = maxShell_;

      auto requested_hw = hw_;
      // only change hw to 20 if the matrix elements are computed on the fly
      if (on_the_fly) hw_ = 20;

      // auto tensors = expand_new();
      expand_exist(maxShell, maxShell, tensors); // maxShell is from file
      writeBinary(filename, tensors);

      hw_ = requested_hw;
      scaleStrengths(tensors);
      return tensors;
    }

    scaleStrengths(tensors);
    return tensors;
  }

  U3xU4::MultiShellTensors_phys TwoBodyOperator::readBinary_MPI(
    const std::string& filename)
  {
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::ifstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for reading: " + filename);

    HO::qn_t maxShell; /* = Nmax */
    qcx_utils::readBinary(s, maxShell);

    if ((maxShell_ > maxShell) or (maxShell_ - 2 * valenceShell_ > maxShell))
    {
      if (mpi_rank == 0)
        std::cout << "File doesn't reach your maxShell/Nmax, it will be expanded now"
                  << std::endl;

      valenceShell_ = 0;
      Nmax_ = maxShell_;

      auto requested_hw = hw_;
      // only change hw to 20 if the matrix elements are computed on the fly
      if (on_the_fly) hw_ = 20;

      s.close(); // close before expansion rewrites the file
      expand_exist_MPI(filename, maxShell, maxShell); // maxShell is from file

      hw_ = requested_hw;

      s.open(filename, std::ios::binary); // reopen updated file
      qcx_utils::readBinary(s, maxShell); // re-read header
    }

    std::cerr << "Start reading " << name_ 
              << " from binary file with capacity maxShell = " 
              << static_cast<int>(maxShell) << " on rank " << mpi_rank << std::endl;

    // Read number of tensors
    size_t n_tensors;
    qcx_utils::readBinary(s, n_tensors);

    U3xU4::MultiShellTensors_phys tensors;
    tensors.reserve(n_tensors);

    for (size_t i = 0; i < n_tensors; ++i)
    { 
      U3xU4::MultiShellTensor_phys tensor(HO_labels_);
      tensor.readBinary(s);

      int ab_Nhw = 0, cd_Nhw = 0, ads = 0, ats = 0;
      for (const auto& G : tensor.Gammas())
      {
        ab_Nhw += ((G.shell() - valenceShell_) * G.num_ads());
        cd_Nhw += ((G.shell() - valenceShell_) * G.num_ats());

        ads += G.num_ads();
        ats += G.num_ats();
      }
      
      // number of quanta shifted by a+'s and a-'s cannot exceed Nmax
      if (ab_Nhw > Nmax_ || cd_Nhw > Nmax_ || abs(ab_Nhw - cd_Nhw) > Nmax_) 
        continue;

      assert(ads == 2);
      assert(ats == 2);

      tensors.push_back(std::move(tensor));
    }

    std::cerr << "Finish reading from binary file on rank " << mpi_rank << std::endl;

    scaleStrengths(tensors);
    return tensors;
  }

  // Matrix representation
  qcx_utils::CSRMatrix 
  TwoBodyOperator::matrix_CSR(const U3xU4::ManyNucleonBasis& basis)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements_CSR(tensors, basis, gelfand_labels_);
  }

  Eigen::SparseMatrix<double> 
  TwoBodyOperator::matrix_Eigen(const U3xU4::ManyNucleonBasis& basis)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements_Eigen(tensors, basis, gelfand_labels_);
  }
  
  Eigen::SparseMatrix<double> TwoBodyOperator::matrix_Eigen(
    const U3xU4::ManyNucleonBasis& bras, 
    const U3xU4::ManyNucleonBasis& kets)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements_Eigen(tensors, bras, kets, gelfand_labels_);
  }

  // Tensor Analyzers
  std::vector<std::pair<u3::irrep, sam::strength_stats>>
    TwoBodyOperator::U3_tensor_analyzer_write()
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_U3.dat";

    auto tensors = expand();
    return sam::U3_tensor_analyzer(tensors, filename);
  }
  
  std::vector<std::pair<u3::irrep, sam::strength_stats>>
    TwoBodyOperator::U3_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::U3_tensor_analyzer(tensors);
  }

  std::vector<std::pair<u4::irrep, sam::strength_stats>>
    TwoBodyOperator::U4_tensor_analyzer_write()
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_U4.dat";

    auto tensors = expand();
    return sam::U4_tensor_analyzer(tensors, filename);
  }
  
  std::vector<std::pair<u4::irrep, sam::strength_stats>>
    TwoBodyOperator::U4_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::U4_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::LL_tensor_analyzer_write()
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_LL.dat";

    auto tensors = expand();
    return sam::LL_tensor_analyzer(tensors, filename);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::LL_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::LL_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::SS_tensor_analyzer_write()
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_SS.dat";

    auto tensors = expand();
    return sam::SS_tensor_analyzer(tensors, filename);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::SS_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::SS_tensor_analyzer(tensors);
  }
} // namespace JT
