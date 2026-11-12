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

#include <U3xU4Creators.h>
#include <U3xU4Annihilators.h>

namespace U3xU4
{
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

    // name of binary file that stores RMEs; a function of the labels above,
    // set by every constructor and by readBinary()
    std::optional<std::string> filename_;
    void set_filename();

  public:
    // Constructor
    SingleShellTensor() : U3xU4::GammaElement(0, 0, {}, 0, {})
    { }

    SingleShellTensor(utils::UInt8 n,
                      utils::UInt8 ads, utils::UInt8 ats,
                      const U3xU4::irreps& Omega_ads,
                      const U3xU4::irreps& Omega_ats,
                      const U3xU4::irrep& T, 
                      utils::UInt8 rho0max, utils::UInt8 eta0max)
        : U3xU4::GammaElement(n, rho0max, T.u3(), eta0max, T.u4()),
          ads_(ads),
          ats_(ats),
          Omega_ads_(Omega_ads),
          Omega_ats_(Omega_ats)
    {
      set_filename();
    }

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
      utils::UInt8 n, 
      const u3::gelfands& sps3, const u3::gelfands& sps3_conj,
      const u4::gelfands& sps4, const u4::gelfands& sps4_conj, 
      utils::UInt8 ads, utils::UInt8 ats,
      const U3xU4::irreps& Omega_ads,
      const U3xU4::irreps& Omega_ats,
      const U3xU4::irrep& T, 
      utils::UInt8 rho0max, utils::UInt8 eta0max,
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
  // Decomposition cache (defined in U3xU4SingleShellTensorsDecomposition_Database.h,
  // which includes this header). Forward-declared here so SingleShellTensor_phys
  // methods can take it by reference without a circular include.
  class SSTDecomp_phys_db;

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
      utils::Int8 LL, utils::Int8 SS,
      utils::Int8 TT, utils::Int8 MMT,
      utils::Int8 JJ, utils::Int8 MMJ,
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
      kappamax_ = utils::itou8(u3::Kappa(u3(), LL_ / 2));
      zetamax_ = utils::itou8(u4::Zeta(u4(), SS_, TT_));
    }
    SingleShellTensor_phys(
      utils::UInt8 n, 
      utils::UInt8 ads, utils::UInt8 ats,
      const U3xU4::irreps& Omega_ads,
      const U3xU4::irreps& Omega_ats,
      const U3xU4::irrep& T, 
      utils::UInt8 rho0max, utils::UInt8 eta0max,
      utils::Int8 LL, utils::Int8 SS,
      utils::Int8 TT, utils::Int8 MMT,
      utils::Int8 JJ, utils::Int8 MMJ,
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
      kappamax_ = utils::itou8(u3::Kappa(u3(), LL_ / 2));
      zetamax_ = utils::itou8(u4::Zeta(u4(), SS_, TT_));
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

    // Expand in terms of uncoupled products of ad's and a's (not at's because a
    // phase is added). Returns the shared flat Expansion; coeff rows have
    // rhomax x etamax x kappamax x zetamax columns, index of each column is
    // ((rho * etamax + eta) * kappamax + kappa) * zetamax + zeta.
    Expansion expand() const;

    // Get two-body strength from the cached expansion (db.get) contracted
    // against the matrix elements.
    std::vector<double>
      getStrengths(const HO::TwBME_scalar& TBME, SSTDecomp_phys_db& db) const;
    std::vector<double>
      getStrengths(const HO::TwBME_general& TBME, SSTDecomp_phys_db& db) const;
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
