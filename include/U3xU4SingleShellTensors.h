#pragma once

#include <cstdint>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include <Databases.h>
#include <HOutilities.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4irrep.h>
#include <RME.h>

#include <qcx_convert.h>

#ifdef HAVE_FMT
#include <fmt/format.h>
#endif

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
    uint8_t r, // U(4) Gelfand for [1,0,0,0]
            c; // U(3) gelfand for [n,0,0]

    const std::vector<u3::gelfand>& sps3; // Gelfand array for [n,0,0]
    const std::vector<u4::gelfand>& sps4; // Gelfand array for [1,0,0,0]

  public:
    // Constructor
    ad_gelfand(int c_, int r_,
               const std::vector<u3::gelfand>& sps3_,
               const std::vector<u4::gelfand>& sps4_)
        : c(qcx_utils::itou8(c_)), 
          r(qcx_utils::itou8(r_)), 
          sps3(sps3_), sps4(sps4_)
    { }

    // Get tensor characters
    std::pair<uint8_t, uint8_t> getTensorCharacters() const
    {
      return std::make_pair(r, c);
    }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const ad_gelfand& x);
  };
  using ads_gelfand = std::vector<U3xU4::ad_gelfand>;

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
    const HO::Table_nljmjmt& labels;
    uint16_t index;

  public:
    // Constructor
    
    ad_phys(int8_t n_,                // shell number
            int8_t ll_,               // twice orbital angular momentum
            int8_t jj_, int8_t mmj_,  // twice total angular momentum
            int8_t mmt_,              // twice isospin projection
            const HO::Table_nljmjmt& labels_)
        : labels(labels_)
    {
      index = labels.getIndex(n_, ll_, jj_, mmj_, mmt_);
    }

    // Get labels
    auto getLabels() const { return labels.getLabels(index); }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const ad_phys& x);
  };
  using ads_phys = std::vector<U3xU4::ad_phys>;

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
    uint8_t r, // U(4) Gelfand for [1,1,1,0]
            c; // U(3) gelfand for [n,n,0]

    const std::vector<u3::gelfand>& sps3;      // Gelfand array for [n,0,0]
    const std::vector<u3::gelfand>& sps3_conj; // Gelfand array for [n,n,0]

    const std::vector<u4::gelfand>& sps4;      // Gelfand array for [1,0,0,0]
    const std::vector<u4::gelfand>& sps4_conj; // Gelfand array for [1,1,1,0]

    bool tensor = true; // flag indicating if this object is tensor or operator

  public:
    // Constructor
    at_gelfand(int c_, int r_,
               const std::vector<u3::gelfand>& sps3_,
               const std::vector<u3::gelfand>& sps3_conj_,
               const std::vector<u4::gelfand>& sps4_,
               const std::vector<u4::gelfand>& sps4_conj_)
        : c(qcx_utils::itou8(c_)),
          r(qcx_utils::itou8(r_)),
          sps3(sps3_),
          sps3_conj(sps3_conj_),
          sps4(sps4_),
          sps4_conj(sps4_conj_)
    { 
      assert(c_ < 256);
    }

    // Get tensor characters
    std::pair<uint8_t, uint8_t> getTensorCharacters() const
    {
      return std::make_pair(r, c);
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
  using ats_gelfand = std::vector<U3xU4::at_gelfand>;

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
    const HO::Table_nljmjmt& labels;
    uint16_t index;

  public:
    // Constructor
    at_phys(int8_t n_,                // shell number
            int8_t ll_,               // twice orbital angular momentum
            int8_t jj_, int8_t mmj_,  // twice total angular momentum
            int8_t mmt_,              // twice isospin projection
            const HO::Table_nljmjmt& labels_)
        : labels(labels_)
    {
      index = labels.getIndex(n_, ll_, jj_, mmj_, mmt_);
    }

    // Get labels
    auto getLabels() const { return labels.getLabels(index); }

    // Turn it to a by conjugating, this method returns a phase
    // and change mmj to -mmj, mmt to -mmt
    // After this method is executed, tensor is turned to false
    double turnToAnnihilation();

    // Check if it is annihilation tensor or annihilation operator
    bool isTensor() const { return tensor; }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const at_phys& x);
  };
  using ats_phys = std::vector<U3xU4::at_phys>;

  /////////////// PRODUCT OF CREATION OPERATORS/TENSORS /////////////////

  /* Class for holding coupled products of ad's, it can have up to 3 ad's
     ad, [ad x ad], [[ad x ad] x ad]
     Tensor components are represented by Gelfand patterns
  */
  class Tad_gelfand
  {
  private:
    // tensor character of ad [n,0,0] x [1,0,0,0]
    int8_t n;

    const std::vector<u3::gelfand>& sps3; // Gelfand array for [n,0,0]
    const std::vector<u4::gelfand>& sps4; // Gelfand array for [1,0,0,0]

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U30 x U40) x ad](U3 x U4)
    std::vector<U3xU4::irrep> Omegas;

    // tensor component represented by two Gelfand patterns
    // which is associated with the last element of prod_characters
    u3::gelfand g3;
    u4::gelfand g4;

  public:
    // Constructor
    Tad_gelfand(int8_t n_, const std::vector<u3::gelfand>& sps3_,
                const std::vector<u4::gelfand>& sps4_,
                const std::vector<U3xU4::irrep>& Omegas_, 
                const u3::gelfand& g3_, const u4::gelfand& g4_)
        : n(n_), sps3(sps3_), sps4(sps4_), Omegas(Omegas_), g3(g3_), g4(g4_)
    { }

    // Get the number of ad's
    uint8_t num_ads() const { return Omegas.size() + 1; }

    // Get shell number
    int8_t shell() const { return n; }

    // Expand in terms of uncoupled products of ad's
    std::pair<std::vector<double>, std::vector<U3xU4::ads_gelfand>>
    expand(sam::Databases& dbs) const;
  };

  /* Class for holding coupled products of ad's, it can have up to 3 ad's
     ad, [ad x ad], [[ad x ad] x ad]
     Tensor components are represented by kappa LL zeta SS TT MMT JJ MMJ
  */
  class Tad_phys
  {
  private:
    // A lookup table for the labels
    const HO::Table_nljmjmt& labels;

  public:
    // tensor character of ad [n,0,0] x [1,0,0,0]
    int8_t n;

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U30 x U40) x ad](U3 x U4)
    std::vector<U3xU4::irrep> Omegas;

    // tensor components
    int8_t kappamax, LL, SS, TT, MMT, JJ, MMJ;

    // Constructor
    Tad_phys(int8_t n_, const std::vector<U3xU4::irrep>& Omegas_, int8_t LL_,
             int8_t SS_, int8_t TT_, int8_t MMT_, int8_t JJ_, int8_t MMJ_,
             const HO::Table_nljmjmt& labels_)
        : n(n_), Omegas(Omegas_), LL(LL_), SS(SS_), TT(TT_),
          MMT(MMT_), JJ(JJ_), MMJ(MMJ_), labels(labels_)
    {
      if (Omegas.empty())
        kappamax = 1;
      else
        kappamax = qcx_utils::ito8(
                   u3::Kappa(Omegas.back().u3(), static_cast<int>(LL) / 2));
    }

    // Get the number of ad's
    uint8_t num_ads() const { return Omegas.size() + 1; }

    // Get shell number
    int8_t shell() const { return n; }

    // Expand in terms of uncoupled products of ad's
    std::pair<std::vector<Eigen::RowVectorXd>, // each element has size ==
                                               // kappamax
              std::vector<U3xU4::ads_phys>>
    expand(sam::Databases& dbs) const;
  };

  /////////////// PRODUCT OF ANNIHILATION OPERATORS/TENSORS ///////////////

  /* Class for holding coupled products of at's, it can have up to 3 at's
     at, [at x at], [[at x at] x at]
     Tensor components are represented by Gelfand patterns
  */
  class Tat_gelfand
  {
  private:
    // tensor character of at [n,n,0] x [1,1,1,0]
    int8_t n;

    const std::vector<u3::gelfand>& sps3;      // Gelfand array for [n,0,0]
    const std::vector<u3::gelfand>& sps3_conj; // Gelfand array for [n,n,0]

    const std::vector<u4::gelfand>& sps4;      // Gelfand array for [1,0,0,0]
    const std::vector<u4::gelfand>& sps4_conj; // Gelfand array for [1,1,1,0]

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U30 x U40) x at](U3 x U4)
    std::vector<U3xU4::irrep> Omegas;

    // tensor component represented by two Gelfand patterns
    u3::gelfand g3;
    u4::gelfand g4;

  public:
    // Constructor
    Tat_gelfand(int8_t n_, const std::vector<u3::gelfand>& sps3_,
                const std::vector<u3::gelfand>& sps3_conj_,
                const std::vector<u4::gelfand>& sps4_,
                const std::vector<u4::gelfand>& sps4_conj_,
                const std::vector<U3xU4::irrep>& Omegas_, 
                const u3::gelfand& g3_, const u4::gelfand& g4_)
        : n(n_), sps3(sps3_), sps3_conj(sps3_conj_), sps4(sps4_), 
          sps4_conj(sps4_conj_), Omegas(Omegas_), g3(g3_), g4(g4_)
    { }

    // Get the number of at's
    uint8_t num_ats() const { return Omegas.size() + 1; }

    // Get shell number
    int8_t shell() const { return n; }

    // Expand in terms of uncoupled products of a's
    // (not at's because a phase is added)
    std::pair<std::vector<double>, std::vector<U3xU4::ats_gelfand>>
    expand(sam::Databases& dbs) const;
  };

  /* Class for holding coupled products of at's, it can have up to 3 at's
     at, [at x at], [[at x at] x at]
     Tensor components are represented by kappa LL zeta SS TT MMT JJ MMJ
  */
  class Tat_phys
  {
  private:
    // A lookup table for the labels
    const HO::Table_nljmjmt& labels;

  public:
    // tensor character of at [n,n,0] x [1,1,1,0]
    int8_t n;

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U30 x U40) x at](U3 x U4)
    std::vector<U3xU4::irrep> Omegas;

    // tensor components
    int8_t kappamax, LL, SS, TT, MMT, JJ, MMJ;

    // Constructor
    Tat_phys(int8_t n_, const std::vector<U3xU4::irrep>& Omegas_, int8_t LL_,
             int8_t SS_, int8_t TT_, int8_t MMT_, int8_t JJ_, int8_t MMJ_,
             const HO::Table_nljmjmt& labels_)
        : n(n_), Omegas(Omegas_), LL(LL_), SS(SS_), TT(TT_),
          MMT(MMT_), JJ(JJ_), MMJ(MMJ_), labels(labels_)
    {
      if (Omegas.empty())
        kappamax = 1;
      else
        kappamax = qcx_utils::ito8(
                   u3::Kappa(Omegas.back().u3(), static_cast<int>(LL) / 2));
    }

    // Get the number of at's
    uint8_t num_ats() const { return Omegas.size() + 1; }

    // Get shell number
    int8_t shell() const { return n; }

    // Expand in terms of uncoupled products of at's
    std::pair<std::vector<Eigen::RowVectorXd>, // each element has size ==
                                               // kappamax
              std::vector<U3xU4::ats_phys>>
    expand(sam::Databases& dbs) const;
  };

  ////// PRODUCT OF CREATION AND ANNIHILATION OPERATORS/TENSORS ///////

  /* Class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
     Tensor components are not specified
  */
  class SingleShellTensor : public U3xU4::GammaElement
  {
  public:
    // tensor character of ad [n,0,0] x [1,0,0,0]
    // tensor character of at [n,n,0] x [1,1,1,0]

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U3 x U4) x ad](U3_1 x U4_1)
    std::vector<U3xU4::irrep> Omega_ads;
    // number of ad's
    uint8_t ads;

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U3 x U4) x at](U3_2 x U4_2)
    std::vector<U3xU4::irrep> Omega_ats;
    // number of at's
    uint8_t ats;

    // final tensor coupling stored in GammaElement base class

    // Constructor
    SingleShellTensor() : U3xU4::GammaElement(0, 0, {}, 0, {})
    { }

    SingleShellTensor(uint8_t n_, uint8_t ads_, uint8_t ats_,
                      const std::vector<U3xU4::irrep>& Omega_ads_,
                      const std::vector<U3xU4::irrep>& Omega_ats_,
                      const U3xU4::irrep& T_, uint8_t rho0max_, uint8_t eta0max_)
        : U3xU4::GammaElement(n_, rho0max_, T_.u3(), eta0max_, T_.u4()),
          ads(ads_),
          ats(ats_),
          Omega_ads(Omega_ads_),
          Omega_ats(Omega_ats_)
    { }

    // Get the number of ad's
    uint8_t num_ads() const { return ads; }

    // Get the number of at's
    uint8_t num_ats() const { return ats; }

    // Check if it's identity
    bool identity() const { return (ads == 0) && (ats == 0); }

    const U3xU4::irrep& T() const { return *this; }

    int rho0max() const { return rhomax(); }
    int eta0max() const { return etamax(); }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const SingleShellTensor& x);

    // Write to binary file
    void writeBinary(std::ostream& s) const;

    // Read from binary file
    void readBinary(std::istream& s);

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
    sam::RME RME(const U3xU4::GammaElement& f, const U3xU4::GammaElement& i,
                 sam::Databases& dbs) const;
  };

  /* Class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
     Tensor components are represented by Gelfand patterns
  */
  class SingleShellTensor_gelfand : public U3xU4::SingleShellTensor
  {
  private:
    const std::vector<u3::gelfand>& sps3;      // Gelfand array for [n,0,0]
    const std::vector<u3::gelfand>& sps3_conj; // Gelfand array for [n,n,0]

    const std::vector<u4::gelfand>& sps4;      // Gelfand array for [1,0,0,0]
    const std::vector<u4::gelfand>& sps4_conj; // Gelfand array for [1,1,1,0]

    // tensor component represented by two Gelfand patterns
    u3::gelfand g3; // associated with T.u3
    u4::gelfand g4; // associated with T.u4

  public:
    // Constructor
    SingleShellTensor_gelfand(
      uint8_t n_, 
      const std::vector<u3::gelfand>& sps3_,
      const std::vector<u3::gelfand>& sps3_conj_,
      const std::vector<u4::gelfand>& sps4_,
      const std::vector<u4::gelfand>& sps4_conj_, 
      uint8_t ads_, uint8_t ats_,
      const std::vector<U3xU4::irrep>& Omega_ads_,
      const std::vector<U3xU4::irrep>& Omega_ats_,
      const U3xU4::irrep& T_, uint8_t rho0max_, uint8_t eta0max_,
      const u3::gelfand& g3_, const u4::gelfand& g4_)
        : SingleShellTensor(n_, ads_, ats_, Omega_ads_, Omega_ats_, 
                                            T_, rho0max_, eta0max_),
          sps3(sps3_), sps3_conj(sps3_conj_),
          sps4(sps4_), sps4_conj(sps4_conj_),
          g3(g3_),
          g4(g4_)
    { }

    // Expand in terms of uncoupled products of ad's and a's
    // (not at's because a phase is added)
    // coeffs have rhomax x etamax columns, index of each column is
    // rho * etamax + eta
    std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
               std::vector<U3xU4::ads_gelfand>, // ad's
               std::vector<U3xU4::ats_gelfand>> // a's
    expand(sam::Databases& dbs) const;
  };

  /* Class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
     Tensor components are represented by kappa LL zeta SS TT MMT JJ MMJ
  */
  class SingleShellTensor_phys : public U3xU4::SingleShellTensor
  {
  private:
    // A lookup table for the labels
    const HO::Table_nljmjmt& labels;

  public:
    // tensor components
    int8_t kappamax, LL, zetamax, SS, TT, MMT, JJ, MMJ;

    // Constructor
    SingleShellTensor_phys(
      uint8_t n_, uint8_t ads_, uint8_t ats_,
      const std::vector<U3xU4::irrep>& Omega_ads_,
      const std::vector<U3xU4::irrep>& Omega_ats_,
      const U3xU4::irrep& T_, uint8_t rho0max_, uint8_t eta0max_,
      int8_t LL_, int8_t SS_, int8_t TT_, int8_t MMT_, int8_t JJ_,
      int8_t MMJ_, const HO::Table_nljmjmt& labels_)
        : SingleShellTensor(n_, ads_, ats_, Omega_ads_, Omega_ats_, 
                                            T_, rho0max_, eta0max_),
          LL(LL_),
          SS(SS_),
          TT(TT_),
          MMT(MMT_),
          JJ(JJ_),
          MMJ(MMJ_),
          labels(labels_)
    {
      kappamax = qcx_utils::ito8(u3::Kappa(u3(), LL / 2));
      zetamax = qcx_utils::ito8(u4::Zeta(u4(), SS, TT));
    }

    // Expand in terms of uncoupled products of ad's and a's
    // (not at's because a phase is added)
    // coeffs have rhomax x etamax x kappamax x zetamax columns, index of each
    // column is
    // ((rho * etamax + eta) * kappamax + kappa) * zetamax + zeta
    std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
               std::vector<U3xU4::ads_phys>,    // ad's
               std::vector<U3xU4::ats_phys>>    // a's
    expand(sam::Databases& dbs) const;
  };
  using SingleShellTensors_phys = std::vector<U3xU4::SingleShellTensor_phys>;
  using ads_ats_phys =
      std::vector<std::variant<U3xU4::ad_phys, U3xU4::at_phys>>;

  // Find all single-shell tensors composed of
  // num_ads creation tensors and num_ats annihilation tensors
  std::vector<U3xU4::SingleShellTensor> 
    GenerateSingleShellTensors(int n, int num_ads, int num_ats);

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
    return fmt::format_to(ctx.out(), "{}({},{}){}{}{}{}", t.shell(), t.ads,
                          t.ats, t.rho0max(), t.u3(), t.eta0max(), t.u4());
  }
};

#endif
