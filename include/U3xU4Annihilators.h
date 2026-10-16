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
  /////////////////// ANNIHILATION OPERATOR/TENSOR /////////////////

  /* Class for defining annihilation tensor a^tilde
     with tensor components given by Gelfand patterns
                     [n,n,0,n,n22b,n11b]
                    at
                     [1,1,1,0,1,1,n33p,1,n22p,n11p]
     Turn it to annihilation operator
              [n,0,0,n-n22b,0,n-n11b]
     (-)^phi a
              [1,0,0,0,1-n33p,0,0,1-n22p,0,1-n11p]
     where the phase phi = n22b + n11b + n33p + n22p + n11p
  */
  class at_gelfand
  {
  private:
    uint8_t r_, // U(4) Gelfand for [1,1,1,0]
            c_; // U(3) gelfand for [n,n,0]

    const u3::gelfands& sps3_;      // Gelfand array for [n,0,0]
    const u3::gelfands& sps3_conj_; // Gelfand array for [n,n,0]

    const u4::gelfands& sps4_;      // Gelfand array for [1,0,0,0]
    const u4::gelfands& sps4_conj_; // Gelfand array for [1,1,1,0]

    bool tensor = true; // flag indicating if this object is tensor or operator

  public:
    // Constructor
    at_gelfand(utils::UInt8 c, utils::UInt8 r,
               const u3::gelfands& sps3, const u3::gelfands& sps3_conj,
               const u4::gelfands& sps4, const u4::gelfands& sps4_conj)
        : c_(c), r_(r),
          sps3_(sps3), sps3_conj_(sps3_conj),
          sps4_(sps4), sps4_conj_(sps4_conj)
    { 
      assert(r < 4);
    }

    // Get bit position in Slater Determinant
    std::pair<size_t, size_t> getSDLabels() const
    {
      assert(!tensor);
      return std::make_pair(static_cast<size_t>(r_), 
                            static_cast<size_t>(c_));
    }

    // Turn it to a by conjugating, this method returns a phase
    // and change r & c to Gelfand patterns for conjugate irreps,
    // [1,0,0,0] & [n,0,0], respectively.
    // After this method is executed, tensor is turned to false
    double turnToAnnihilation();

    // Check if it is annihilation tensor or annihilation operator
    bool isTensor() const { return tensor; }

    // Print, note that if this is called after turnToAnnihilation()
    // sps3 and sps4 must the the conjugate Gelfand arrays
    friend std::ostream& operator<<(std::ostream& s, const at_gelfand& x);
  };
  using ats_gelfand = std::vector<at_gelfand>;

  /* Class for defining annihilation tensor a^tilde
     with tensor components given by Gelfand patterns
                     [n,n,0][1,1,1,0]
                    at
                     ll jj mmj mmt
     Turn it to annihilation operator
              [n,0,0][1,0,0,0]
     (-)^phi a
              ll jj -mmj -mmt
     where the phase phi = (n + l + 1)/2 - j - mj - mt
  */
  class at_phys
  {
  private:
    // flag indicating if this object is tensor or operator
    bool tensor = true;

    // A lookup table for the labels and location in the table
    const HO::Table_nljmjmt& labels_;
    uint16_t index;

  public:
    // Constructor
    at_phys(utils::Int8 n,   // shell number
            utils::Int8 ll,  // twice orbital angular momentum
            utils::Int8 jj,  // twice total angular momentum
            utils::Int8 mmj, // twice total angular momentum projection
            utils::Int8 mmt, // twice isospin projection
            const HO::Table_nljmjmt& labels)
        : labels_(labels)
    {
      index = labels_.getIndex(n, ll, jj, mmj, mmt);
    }

    // Reconstruct directly from a stored table index, as an annihilation
    // operator (tensor = false). Tad/Tat expansions always return a^tilde's
    // already turned to annihilation, so the decomposition database caches them
    // in that state. The index enumeration in Table_nljmjmt is maxShell-
    // invariant, so a cached index identifies the same state across runs.
    at_phys(uint16_t idx, const HO::Table_nljmjmt& labels)
        : tensor(false), labels_(labels), index(idx)
    { }

    // Get labels
    const auto& getLabels() const { return labels_.getLabels(index); }

    // Get the table index (with isTensor(), the full state of this operator)
    uint16_t getIndex() const { return index; }

    // Turn it to a by conjugating, this method returns a phase
    // and change mmj to -mmj, mmt to -mmt
    // After this method is executed, tensor is turned to false
    double turnToAnnihilation();

    // Check if it is annihilation tensor or annihilation operator
    bool isTensor() const { return tensor; }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const at_phys& x);
  };
  using ats_phys = std::vector<at_phys>;
  /////////////// PRODUCT OF ANNIHILATION OPERATORS/TENSORS ///////////////

  /* Class for holding coupled products of at's, it can have up to 3 at's
     at, [at x at], [[at x at] x at]
     Tensor components are represented by Gelfand patterns
  */
  class Tat_gelfand
  {
  private:
    // tensor character of at [n,n,0] x [1,1,1,0]
    HO::qn_t n_;

    const u3::gelfands& sps3_;      // Gelfand array for [n,0,0]
    const u3::gelfands& sps3_conj_; // Gelfand array for [n,n,0]

    const u4::gelfands& sps4_;      // Gelfand array for [1,0,0,0]
    const u4::gelfands& sps4_conj_; // Gelfand array for [1,1,1,0]

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U30 x U40) x at](U3 x U4)
    U3xU4::irreps Omegas_;

    // tensor component represented by two Gelfand patterns
    u3::gelfand g3_;
    u4::gelfand g4_;

  public:
    // Constructor
    Tat_gelfand(utils::Int8 n,
                const u3::gelfands& sps3, const u3::gelfands& sps3_conj,
                const u4::gelfands& sps4, const u4::gelfands& sps4_conj,
                const U3xU4::irreps& Omegas, 
                const u3::gelfand& g3, const u4::gelfand& g4)
        : n_(n), 
          sps3_(sps3), sps3_conj_(sps3_conj),
          sps4_(sps4), sps4_conj_(sps4_conj),
          Omegas_(Omegas), g3_(g3), g4_(g4)
    { }

    // Getters
    int shell() const { return static_cast<int>(n_); }
    const auto& Omegas() const { return Omegas_; }
    const auto& g3() const { return g3_; }
    const auto& g4() const { return g4_; }

    // Get the number of at's
    int num_ats() const { return Omegas_.size() + 1; }

    // Expand in terms of uncoupled products of a's
    // (not at's because a phase is added)
    std::pair<std::vector<double>, std::vector<U3xU4::ats_gelfand>>
    expand() const;
  };
  using Tats_gelfand = std::vector<Tat_gelfand>;

  /* Class for holding coupled products of at's, it can have up to 3 at's
     at, [at x at], [[at x at] x at]
     Tensor components are represented by kappa LL zeta SS TT MMT JJ MMJ
  */
  class Tat_phys
  {
  private:
    // A lookup table for the labels
    const HO::Table_nljmjmt& labels_;

    // tensor character of at [n,n,0] x [1,1,1,0]
    HO::qn_t n_;

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U30 x U40) x at](U3 x U4)
    U3xU4::irreps Omegas_;

    // tensor components
    uint8_t kappamax_;
    HO::qn_t LL_, SS_, TT_, MMT_, JJ_, MMJ_;

  public:
    // Constructor
    Tat_phys(utils::Int8 n, const U3xU4::irreps& Omegas, 
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

    // Get the number of at's
    int num_ats() const { return Omegas_.size() + 1; }

    // Expand in terms of uncoupled products of a's (flat Expansion; coeff rows
    // have kappamax columns, every operator is an annihilation a~).
    Expansion expand() const;
  };
  using Tats_phys = std::vector<Tat_phys>;
} // namespace U3xU4
