#pragma once

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

#include <qcx_convert.h>
#include <qcx_Int.h>

#ifdef HAVE_FMT
#include <fmt/format.h>
#endif

namespace U3xU4
{
  // Decomposition cache (defined in U3xU4SingleShellTensorsDecomposition_Database.h,
  // which includes this header). Forward-declared here so SingleShellTensor_phys
  // methods can take it by reference without a circular include.
  class SSTDecomp_phys_db;

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
    ad_gelfand(qcx_utils::UInt8 c, qcx_utils::UInt8 r,
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
    
    ad_phys(qcx_utils::Int8 n,   // shell number
            qcx_utils::Int8 ll,  // twice orbital angular momentum
            qcx_utils::Int8 jj,  // twice total angular momentum
            qcx_utils::Int8 mmj, // twice total angular momentum projection
            qcx_utils::Int8 mmt, // twice isospin projection
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
    at_gelfand(qcx_utils::UInt8 c, qcx_utils::UInt8 r,
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
    at_phys(qcx_utils::Int8 n,   // shell number
            qcx_utils::Int8 ll,  // twice orbital angular momentum
            qcx_utils::Int8 jj,  // twice total angular momentum
            qcx_utils::Int8 mmj, // twice total angular momentum projection
            qcx_utils::Int8 mmt, // twice isospin projection
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
    Tad_gelfand(qcx_utils::Int8 n, 
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
    Tad_phys(qcx_utils::Int8 n, const U3xU4::irreps& Omegas, 
             qcx_utils::Int8 LL, qcx_utils::Int8 SS, 
             qcx_utils::Int8 TT, qcx_utils::Int8 MMT, 
             qcx_utils::Int8 JJ, qcx_utils::Int8 MMJ,
             const HO::Table_nljmjmt& labels)
        : n_(n), Omegas_(Omegas), LL_(LL), SS_(SS), TT_(TT),
          MMT_(MMT), JJ_(JJ), MMJ_(MMJ), labels_(labels)
    {
      if (Omegas_.empty())
        kappamax_ = 1;
      else
        kappamax_ = qcx_utils::itou8(
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

    // Expand in terms of uncoupled products of ad's
    std::pair<std::vector<Eigen::RowVectorXd>, // each element has size ==
                                               // kappamax
              std::vector<U3xU4::ads_phys>>
    expand() const;
  };
  using Tads_phys = std::vector<Tad_phys>;

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
    Tat_gelfand(qcx_utils::Int8 n,
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
    Tat_phys(qcx_utils::Int8 n, const U3xU4::irreps& Omegas, 
             qcx_utils::Int8 LL, qcx_utils::Int8 SS, 
             qcx_utils::Int8 TT, qcx_utils::Int8 MMT, 
             qcx_utils::Int8 JJ, qcx_utils::Int8 MMJ,
             const HO::Table_nljmjmt& labels)
        : n_(n), Omegas_(Omegas), LL_(LL), SS_(SS), TT_(TT),
          MMT_(MMT), JJ_(JJ), MMJ_(MMJ), labels_(labels)
    {
      if (Omegas_.empty())
        kappamax_ = 1;
      else
        kappamax_ = qcx_utils::itou8(
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

    // Expand in terms of uncoupled products of at's
    std::pair<std::vector<Eigen::RowVectorXd>, // each element has size ==
                                               // kappamax
              std::vector<U3xU4::ats_phys>>
    expand() const;
  };
  using Tats_phys = std::vector<Tat_phys>;

  ////// PRODUCT OF CREATION AND ANNIHILATION OPERATORS/TENSORS ///////

  /* Class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
     Tensor components are not specified
  */
  class SingleShellTensor : public U3xU4::GammaElement
  {
  private:
    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U3 x U4) x ad](U3_1 x U4_1)
    U3xU4::irreps Omega_ads_;
    // number of ad's
    uint8_t ads_;

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U3 x U4) x at](U3_2 x U4_2)
    U3xU4::irreps Omega_ats_;
    // number of at's
    uint8_t ats_;

    // final tensor coupling stored in GammaElement base class

    // name of binary file that stores RMEs
    std::optional<std::string> filename_;

  public:
    // Constructor
    SingleShellTensor() : U3xU4::GammaElement(0, 0, {}, 0, {})
    { }

    SingleShellTensor(qcx_utils::UInt8 n,
                      qcx_utils::UInt8 ads, qcx_utils::UInt8 ats,
                      const U3xU4::irreps& Omega_ads,
                      const U3xU4::irreps& Omega_ats,
                      const U3xU4::irrep& T, 
                      qcx_utils::UInt8 rho0max, qcx_utils::UInt8 eta0max)
        : U3xU4::GammaElement(n, rho0max, T.u3(), eta0max, T.u4()),
          ads_(ads),
          ats_(ats),
          Omega_ads_(Omega_ads),
          Omega_ats_(Omega_ats)
    { }

    // comparison:
    // auto operator<=>(const SingleShellTensor&) const = default;
    bool operator==(const SingleShellTensor& other) const
    {
      return ads_ == other.ads_ &&
             ats_ == other.ats_ &&
             Omega_ads_ == other.Omega_ads_ &&
             Omega_ats_ == other.Omega_ats_ &&
             static_cast<const U3xU4::GammaElement&>(*this) ==
             static_cast<const U3xU4::GammaElement&>(other);
      // filename_ intentionally ignored
    }

    // Getters
    const auto& Omega_ads() const { return Omega_ads_; }
    const auto& Omega_ats() const { return Omega_ats_; }

    // Get the number of ad's
    int num_ads() const { return static_cast<int>(ads_); }

    // Get the number of at's
    int num_ats() const { return static_cast<int>(ats_); }

    // Get change of oscillator quanta
    int shift_quanta() const { return shell() * (num_ads() - num_ats()); }

    // Check if it's identity
    bool identity() const { return (ads_ == 0) && (ats_ == 0); }

    const U3xU4::irrep& T() const { return *this; }

    int rho0max() const { return rhomax(); }
    int eta0max() const { return etamax(); }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const SingleShellTensor& x);

    // Write to binary file
    void writeBinary(std::ostream& s) const;

    // Read from binary file
    void readBinary(std::istream& s);

    // Get name of the binary RME file
    const std::string& filename() const 
    { 
      assert(filename_.has_value());
      return *filename_; 
    }

    /* Compute RMEs of single-shell U(3)xU(4) tensors
                      rho0 U(3)_0
    < Gamma_f ||| Tadat           ||| Gamma_i >
                      eta0 U(4)_0
      where rho0 and eta0 are from the structure of the tensor itself.

      Inputs:
      i: initial irrep
      f: final irrep

      Outputs:
      RMEs are stored in an array via a sam::RME object
    */
    bool RME_exist(const U3xU4::GammaElement& f, const U3xU4::GammaElement& i) const;

    sam::RME RME(const U3xU4::GammaElement& f, const U3xU4::GammaElement& i) const;

    sam::RME RME(const U3xU4::GammaElement& f, const U3xU4::GammaElement& i,
                 const U3xU4::Table_gelfand& labels) const;
  };
  using SingleShellTensors = std::vector<SingleShellTensor>;

  // Find all single-shell tensors composed of
  // num_ads creation tensors and num_ats annihilation tensors
  SingleShellTensors
    GenerateSingleShellTensors(int n, int num_ads, int num_ats);

  /* Class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
     Tensor components are represented by Gelfand patterns
  */
  class SingleShellTensor_gelfand : public U3xU4::SingleShellTensor
  {
  private:
    const u3::gelfands& sps3_;      // Gelfand array for [n,0,0]
    const u3::gelfands& sps3_conj_; // Gelfand array for [n,n,0]

    const u4::gelfands& sps4_;      // Gelfand array for [1,0,0,0]
    const u4::gelfands& sps4_conj_; // Gelfand array for [1,1,1,0]

    // tensor component represented by two Gelfand patterns
    u3::gelfand g3_; // associated with T.u3
    u4::gelfand g4_; // associated with T.u4

  public:
    // Constructor
    SingleShellTensor_gelfand(
      qcx_utils::UInt8 n, 
      const u3::gelfands& sps3, const u3::gelfands& sps3_conj,
      const u4::gelfands& sps4, const u4::gelfands& sps4_conj, 
      qcx_utils::UInt8 ads, qcx_utils::UInt8 ats,
      const U3xU4::irreps& Omega_ads,
      const U3xU4::irreps& Omega_ats,
      const U3xU4::irrep& T, 
      qcx_utils::UInt8 rho0max, qcx_utils::UInt8 eta0max,
      const u3::gelfand& g3, const u4::gelfand& g4)
        : SingleShellTensor(n, ads, ats, Omega_ads, Omega_ats, 
                                            T, rho0max, eta0max),
          sps3_(sps3), sps3_conj_(sps3_conj),
          sps4_(sps4), sps4_conj_(sps4_conj),
          g3_(g3),
          g4_(g4)
    { }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const SingleShellTensor_gelfand& x);

    // Expand in terms of uncoupled products of ad's and a's
    // (not at's because a phase is added)
    // coeffs have rhomax x etamax columns, index of each column is
    // rho * etamax + eta
    std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
               std::vector<U3xU4::ads_gelfand>, // ad's
               std::vector<U3xU4::ats_gelfand>> // a's
    expand() const;
  };
  using SingleShellTensors_gelfand = std::vector<SingleShellTensor_gelfand>;

  /* Class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
     Tensor components are represented by kappa LL zeta SS TT MMT JJ MMJ
  */
  class SingleShellTensor_phys : public U3xU4::SingleShellTensor
  {
  private:
    // A lookup table for the labels
    const HO::Table_nljmjmt& labels_;

    // tensor components
    uint8_t kappamax_, zetamax_;
    HO::qn_t LL_, SS_, TT_, MMT_, JJ_, MMJ_;

  public:
    // Constructor
    SingleShellTensor_phys(const U3xU4::SingleShellTensor& T,
      qcx_utils::Int8 LL, qcx_utils::Int8 SS,
      qcx_utils::Int8 TT, qcx_utils::Int8 MMT,
      qcx_utils::Int8 JJ, qcx_utils::Int8 MMJ,
      const HO::Table_nljmjmt& labels)
        : SingleShellTensor(T),
          LL_(LL),
          SS_(SS),
          TT_(TT),
          MMT_(MMT),
          JJ_(JJ),
          MMJ_(MMJ),
          labels_(labels)
    {
      kappamax_ = qcx_utils::itou8(u3::Kappa(u3(), LL_ / 2));
      zetamax_ = qcx_utils::itou8(u4::Zeta(u4(), SS_, TT_));
    }
    SingleShellTensor_phys(
      qcx_utils::UInt8 n, 
      qcx_utils::UInt8 ads, qcx_utils::UInt8 ats,
      const U3xU4::irreps& Omega_ads,
      const U3xU4::irreps& Omega_ats,
      const U3xU4::irrep& T, 
      qcx_utils::UInt8 rho0max, qcx_utils::UInt8 eta0max,
      qcx_utils::Int8 LL, qcx_utils::Int8 SS,
      qcx_utils::Int8 TT, qcx_utils::Int8 MMT,
      qcx_utils::Int8 JJ, qcx_utils::Int8 MMJ,
      const HO::Table_nljmjmt& labels)
        : SingleShellTensor(n, ads, ats, Omega_ads, Omega_ats, 
                                            T, rho0max, eta0max),
          LL_(LL),
          SS_(SS),
          TT_(TT),
          MMT_(MMT),
          JJ_(JJ),
          MMJ_(MMJ),
          labels_(labels)
    {
      kappamax_ = qcx_utils::itou8(u3::Kappa(u3(), LL_ / 2));
      zetamax_ = qcx_utils::itou8(u4::Zeta(u4(), SS_, TT_));
    }

    // Getters
    int kappamax() const { return static_cast<int>(kappamax_); }
    int zetamax() const { return static_cast<int>(zetamax_); }
    int LL() const { return static_cast<int>(LL_); }
    int SS() const { return static_cast<int>(SS_); }
    int TT() const { return static_cast<int>(TT_); }
    int MMT() const { return static_cast<int>(MMT_); }
    int JJ() const { return static_cast<int>(JJ_); }
    int MMJ() const { return static_cast<int>(MMJ_); }

    // Label table this tensor's operators are expressed against (needed to
    // rebuild ad_phys/at_phys when loading a cached expansion).
    const HO::Table_nljmjmt& labels() const { return labels_; }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const SingleShellTensor_phys& x);

    // Expand in terms of uncoupled products of ad's and a's
    // (not at's because a phase is added)
    // coeffs have rhomax x etamax x kappamax x zetamax columns, index of each
    // column is
    // ((rho * etamax + eta) * kappamax + kappa) * zetamax + zeta
    // OR, if we switch index: alphamax = rhomax x etamax, then index of each
    // column is
    // (alpha * kappamax + kappa) * zetamax + zeta
    std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
               std::vector<U3xU4::ads_phys>,    // ad's
               std::vector<U3xU4::ats_phys>>    // a's
    expand() const;

    // Get two-body strength from the cached expansion (db.get) contracted
    // against the matrix elements.
    std::vector<double>
      getStrengths(const HO::TwBME_scalar& TBME, SSTDecomp_phys_db& db) const;

  };
  using SingleShellTensors_phys = std::vector<SingleShellTensor_phys>;
  using ads_ats_phys = std::vector<std::variant<ad_phys, at_phys>>;

} // namespace U3xU4

#ifdef HAVE_FMT

// formatter for SingleShellTensor:

template <>
class fmt::formatter<U3xU4::SingleShellTensor>
{
public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename Context>
  constexpr auto format(const U3xU4::SingleShellTensor& t, Context& ctx) const
  {
    return fmt::format_to(ctx.out(), "{}({},{}){}{}{}{}", t.shell(), t.num_ads(),
                          t.num_ats(), t.rho0max(), t.u3(), t.eta0max(), t.u4());
  }
};

#endif
