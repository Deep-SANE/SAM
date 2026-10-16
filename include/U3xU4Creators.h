#pragma once

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include <HOutilities.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4gelfand.h>
#include <U3xU4irrep.h>
#include <RME.h>

#include <convert.h>
#include <Int.h>

#ifdef HAVE_FMT
#include <fmt/format.h>
#endif

#include <U3xU4Expansion.h>

namespace U3xU4
{
  /////////////////// CREATION OPERATOR/TENSOR /////////////////

  /* Class for defining creation operator a^dagger
     with tensor components given by Gelfand patterns
                     [n,0,0,n12b,0,n11b]
                    ad
                     [1,0,0,0,n13p,0,0,n12p,0,n11p]
     Since all the single particle states (or tensor characters
     of ad) are fixed within a shell, we only want to characterize
     ad with two indices, one refers to the index of U(3) Gelfand
     pattern (column of Slater determinant), the other refers to
     the index of U(4) Gelfand pattern (row of Gelfand pattern).

     row index: 0 to 3
     column index: if we go to n=20 shell, the shell degeneracy is
        (20 + 1)(20 + 2)/2 = 231
     Thus both of these indices can be represented by uint8 (0 to 255)
  */
  class ad_gelfand
  {
  private:
    uint8_t r_, // U(4) Gelfand for [1,0,0,0]
            c_; // U(3) gelfand for [n,0,0]

    const u3::gelfands& sps3_; // Gelfand array for [n,0,0]
    const u4::gelfands& sps4_; // Gelfand array for [1,0,0,0]

  public:
    // Constructor
    ad_gelfand(utils::UInt8 c, utils::UInt8 r,
               const u3::gelfands& sps3,
               const u4::gelfands& sps4)
        : c_(c), r_(r), sps3_(sps3), sps4_(sps4)
    {
      assert(r < 4);
    }

    // Get bit position in Slater Determinant
    std::pair<size_t, size_t> getSDLabels() const
    {
      return std::make_pair(static_cast<size_t>(r_), 
                            static_cast<size_t>(c_));
    }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const ad_gelfand& x);
  };
  using ads_gelfand = std::vector<ad_gelfand>;

  /* Class for defining creation operator a^dagger
     with tensor components given by
                     [n,0,0][1,0,0,0]
                    ad
                      ll jj mmj mmt
     Here ss = tt = 1 is automatically implied.

     Assume n = 20 => max(ll) = 40, max(jj) = 41, 
     min(mmj) = -41, max(mmj) = 41, mmt = +-1 
     Thus quantum numbers can be int8: -128 to 127
  */
  class ad_phys
  {
  private:
    // A lookup table for the labels and location in the table
    const HO::Table_nljmjmt& labels_;
    uint16_t index;

  public:
    // Constructor
    
    ad_phys(utils::Int8 n,   // shell number
            utils::Int8 ll,  // twice orbital angular momentum
            utils::Int8 jj,  // twice total angular momentum
            utils::Int8 mmj, // twice total angular momentum projection
            utils::Int8 mmt, // twice isospin projection
            const HO::Table_nljmjmt& labels)
        : labels_(labels)
    {
      index = labels_.getIndex(n, ll, jj, mmj, mmt);
    }

    // Reconstruct directly from a stored table index. The index enumeration in
    // Table_nljmjmt is maxShell-invariant (shells are laid out in order), so a
    // cached index identifies the same state across runs/model spaces. Used by
    // the decomposition database.
    ad_phys(uint16_t idx, const HO::Table_nljmjmt& labels)
        : labels_(labels), index(idx)
    { }

    // Get labels
    const auto& getLabels() const { return labels_.getLabels(index); }

    // Get the table index (full observable state of this operator)
    uint16_t getIndex() const { return index; }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const ad_phys& x);
  };
  using ads_phys = std::vector<ad_phys>;
  /////////////// PRODUCT OF CREATION OPERATORS/TENSORS /////////////////

  /* Class for holding coupled products of ad's, it can have up to 3 ad's
     ad, [ad x ad], [[ad x ad] x ad]
     Tensor components are represented by Gelfand patterns
  */
  class Tad_gelfand
  {
  private:
    // tensor character of ad [n,0,0] x [1,0,0,0]
    HO::qn_t n_;

    const u3::gelfands& sps3_; // Gelfand array for [n,0,0]
    const u4::gelfands& sps4_; // Gelfand array for [1,0,0,0]

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U30 x U40) x ad](U3 x U4)
    U3xU4::irreps Omegas_;

    // tensor component represented by two Gelfand patterns
    // which is associated with the last element of prod_characters
    u3::gelfand g3_;
    u4::gelfand g4_;

  public:
    // Constructor
    Tad_gelfand(utils::Int8 n, 
                const u3::gelfands& sps3, const u4::gelfands& sps4,
                const U3xU4::irreps& Omegas, 
                const u3::gelfand& g3, const u4::gelfand& g4)
        : n_(n), sps3_(sps3), sps4_(sps4), Omegas_(Omegas), g3_(g3), g4_(g4)
    { }

    // Getters
    int shell() const { return static_cast<int>(n_); }
    const auto& Omegas() const { return Omegas_; }
    const auto& g3() const { return g3_; }
    const auto& g4() const { return g4_; }

    // Get the number of ad's
    int num_ads() const { return Omegas_.size() + 1; }

    // Expand in terms of uncoupled products of ad's
    std::pair<std::vector<double>, std::vector<U3xU4::ads_gelfand>>
    expand() const;
  };
  using Tads_gelfand = std::vector<Tad_gelfand>;

  /* Class for holding coupled products of ad's, it can have up to 3 ad's
     ad, [ad x ad], [[ad x ad] x ad]
     Tensor components are represented by kappa LL zeta SS TT MMT JJ MMJ
  */
  class Tad_phys
  {
  private:
    // A lookup table for the labels
    const HO::Table_nljmjmt& labels_;

    // tensor character of ad [n,0,0] x [1,0,0,0]
    HO::qn_t n_;

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U30 x U40) x ad](U3 x U4)
    U3xU4::irreps Omegas_;

    // tensor components
    uint8_t kappamax_;
    HO::qn_t LL_, SS_, TT_, MMT_, JJ_, MMJ_;

  public:
    // Constructor
    Tad_phys(utils::Int8 n, const U3xU4::irreps& Omegas, 
             utils::Int8 LL, utils::Int8 SS, 
             utils::Int8 TT, utils::Int8 MMT, 
             utils::Int8 JJ, utils::Int8 MMJ,
             const HO::Table_nljmjmt& labels)
        : n_(n), Omegas_(Omegas), LL_(LL), SS_(SS), TT_(TT),
          MMT_(MMT), JJ_(JJ), MMJ_(MMJ), labels_(labels)
    {
      if (Omegas_.empty())
        kappamax_ = 1;
      else
        kappamax_ = utils::itou8(
                   u3::Kappa(Omegas_.back().u3(), static_cast<int>(LL_) / 2));
    }

    // Getters
    int shell() const { return static_cast<int>(n_); }
    const auto& Omegas() const { return Omegas_; }
    int kappamax() const { return static_cast<int>(kappamax_); }
    int LL() const { return static_cast<int>(LL_); }
    int SS() const { return static_cast<int>(SS_); }
    int TT() const { return static_cast<int>(TT_); }
    int MMT() const { return static_cast<int>(MMT_); }
    int JJ() const { return static_cast<int>(JJ_); }
    int MMJ() const { return static_cast<int>(MMJ_); }

    // Get the number of ad's
    int num_ads() const { return Omegas_.size() + 1; }

    // Expand in terms of uncoupled products of ad's (flat Expansion; coeff
    // rows have kappamax columns, every operator is a creation ad).
    Expansion expand() const;
  };
  using Tads_phys = std::vector<Tad_phys>;
} // namespace U3xU4
