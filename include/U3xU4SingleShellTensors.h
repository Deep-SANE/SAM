#pragma once

#include <cstdint>
#include <ostream>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

#include <U3SO3xU3SO3.h>
#include <U3U2xU3U2.h>
#include <U4SU2SU2xU4SU2SU2.h>
#include <U4U3xU4U3.h>

#include <U3xU4Irrep.h>
#include <HOutilities.h>

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
  */
  class ad_gelfand
  {
  private:
    std::size_t r, // U(4) Gelfand for [1,0,0,0]
        c;         // U(3) gelfand for [n,0,0]

    const std::vector<u3::gelfand>& sps3; // Gelfand array for [n,0,0]
    const std::vector<u4::gelfand>& sps4; // Gelfand array for [1,0,0,0]

  public:
    // Constructor
    ad_gelfand(std::size_t c_, std::size_t r_,
               const std::vector<u3::gelfand>& sps3_,
               const std::vector<u4::gelfand>& sps4_)
        : c(c_), r(r_), sps3(sps3_), sps4(sps4_)
    { }

    // Get tensor characters
    std::pair<std::size_t, std::size_t> getTensorCharacters() const
    {
      return std::make_pair(r, c);
    }

    // Print
    /* void print() const
    {
      std::cout << "[" << sps3[c] << "] x [" << sps4[r] << "]";
    } */
    friend std::ostream& operator<<(std::ostream& s, const ad_gelfand& x)
    {
      return s << "ad[" << x.sps3[x.c] << "][" << x.sps4[x.r] << "]";
    }
  };
  using ads_gelfand = std::vector<U3xU4::ad_gelfand>;

  /* Class for defining creation operator a^dagger
     with tensor components given by
                     [n,0,0][1,0,0,0]
                    ad
                      ll jj mmj mmt
     Here ss = tt = 1 is automatically implied.
  */
  class ad_phys
  {
  private: 
    // A lookup table for the labels and location in the table
    HO::Table_nljmjmt& labels;
    int index;

  public:    
    // Constructor
    ad_phys(int16_t n_,                 // shell number
            int16_t ll_,                // twice orbital angular momentum
            int16_t jj_, int16_t mmj_,  // twice total angular momentum
            int16_t mmt_,               // twice isospin projection
            HO::Table_nljmjmt& labels_)
        : labels(labels_)
    { 
      index = labels.getIndex(n_, ll_, jj_, mmj_, mmt_);
    }

    // Get labels
    auto getLabels() const
    {
      return labels.getLabels(index);
    }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const ad_phys& x)
    {
      auto [n, ll, jj, mmj, mmt] = x.getLabels();
      return s << "ad(" << n << "," << ll << "," << jj << "," << mmj
               << "," << mmt << ")";
    }
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
    std::size_t r, // U(4) Gelfand for [1,1,1,0]
        c;         // U(3) gelfand for [n,n,0]

    const std::vector<u3::gelfand>& sps3;      // Gelfand array for [n,0,0]
    const std::vector<u3::gelfand>& sps3_conj; // Gelfand array for [n,n,0]

    const std::vector<u4::gelfand>& sps4;      // Gelfand array for [1,0,0,0]
    const std::vector<u4::gelfand>& sps4_conj; // Gelfand array for [1,1,1,0]

    bool tensor = true; // flag indicating if this object is tensor or operator

  public:
    // Constructor
    at_gelfand(std::size_t c_, std::size_t r_,
               const std::vector<u3::gelfand>& sps3_,
               const std::vector<u3::gelfand>& sps3_conj_,
               const std::vector<u4::gelfand>& sps4_,
               const std::vector<u4::gelfand>& sps4_conj_)
        : c(c_),
          r(r_),
          sps3(sps3_),
          sps3_conj(sps3_conj_),
          sps4(sps4_),
          sps4_conj(sps4_conj_)
    { }

    // Get tensor characters
    std::pair<std::size_t, std::size_t> getTensorCharacters() const
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
    /* void print() const
    {
      if (tensor == true)
        std::cout << "[" << sps3_conj[c] << "] x [" << sps4_conj[r] << "]";
      else
        std::cout << "[" << sps3[c] << "] x [" << sps4[r] << "]";
    } */
    friend std::ostream& operator<<(std::ostream& s, const at_gelfand& x)
    {
      if (x.tensor == true)
        return s << "at[" << x.sps3_conj[x.c] << "][" << x.sps4_conj[x.r]
                 << "]";
      else
        return s << "a[" << x.sps3[x.c] << "][" << x.sps4[x.r] << "]";
    }
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
    HO::Table_nljmjmt& labels;
    int index;

  public:
    // Constructor
    at_phys(int16_t n_,                 // shell number
            int16_t ll_,                // twice orbital angular momentum
            int16_t jj_, int16_t mmj_,  // twice total angular momentum
            int16_t mmt_,               // twice isospin projection
            HO::Table_nljmjmt& labels_)
        : labels(labels_)
    { 
      index = labels.getIndex(n_,ll_,jj_,mmj_,mmt_);
    }

    // Get labels
    auto getLabels() const
    {
      return labels.getLabels(index);
    }

    // Turn it to a by conjugating, this method returns a phase
    // and change mmj to -mmj, mmt to -mmt
    // After this method is executed, tensor is turned to false
    double turnToAnnihilation()
    {
      auto [n, ll, jj, mmj, mmt] = labels.getLabels(index);
      int phi = (n + ll / 2 + 1 - jj - mmj - mmt) / 2;
      mmj *= -1;
      mmt *= -1;
      tensor = false;
      index = labels.getIndex(n,ll,jj,mmj,mmt);
      return (phi % 2 == 0) ? 1.0 : -1.0;
    }

    // Check if it is annihilation tensor or annihilation operator
    bool isTensor() const { return tensor; }

    // Print
    friend std::ostream& operator<<(std::ostream& s, const at_phys& x)
    {
      auto [n, ll, jj, mmj, mmt] = x.getLabels();
      if (x.tensor)
        return s << "at(" << n << "," << ll << "," << jj << "," << mmj
                 << "," << mmt << ")";
      else
        return s << "a(" << n << "," << ll << "," << jj << "," << mmj
                 << "," << mmt << ")";
    }
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
    std::size_t n;

    const std::vector<u3::gelfand>& sps3; // Gelfand array for [n,0,0]
    const std::vector<u4::gelfand>& sps4; // Gelfand array for [1,0,0,0]

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U30 x U40) x ad](U3 x U4)
    std::vector<U3xU4::irrep> prod_characters;

    // tensor component represented by two Gelfand patterns
    // which is associated with the last element of prod_characters
    u3::gelfand g3;
    u4::gelfand g4;

  public:
    // Constructor
    Tad_gelfand(std::size_t n_, const std::vector<u3::gelfand>& sps3_,
                const std::vector<u4::gelfand>& sps4_,
                const std::vector<U3xU4::irrep>& prod, const u3::gelfand& b,
                const u4::gelfand& p)
        : n(n_), sps3(sps3_), sps4(sps4_), prod_characters(prod), g3(b), g4(p)
    { }

    // Get the number of ad's
    std::size_t num_ads() const { return prod_characters.size() + 1; }

    // Get shell number
    std::size_t shell() const { return n; }

    // Expand in terms of uncoupled products of ad's
    std::pair<std::vector<double>, std::vector<U3xU4::ads_gelfand>>
    expand(qcx_utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
           u4::WigU4U3_db& wigU4U3db) const;
  };

  /* Class for holding coupled products of ad's, it can have up to 3 ad's
     ad, [ad x ad], [[ad x ad] x ad]
     Tensor components are represented by kappa LL zeta SS TT MMT JJ MMJ
  */
  class Tad_phys
  {
  private:
    // A lookup table for the labels 
    HO::Table_nljmjmt& labels;

  public:
    // tensor character of ad [n,0,0] x [1,0,0,0]
    int16_t n;

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U30 x U40) x ad](U3 x U4)
    std::vector<U3xU4::irrep> prod_characters;

    // tensor components
    int16_t kappamax, LL, SS, TT, MMT, JJ, MMJ;

    // Constructor
    Tad_phys(int16_t n_, const std::vector<U3xU4::irrep>& prod, int16_t LL_,
             int16_t SS_, int16_t TT_, int16_t MMT_, int16_t JJ_, int16_t MMJ_,
             HO::Table_nljmjmt& labels_)
        : n(n_),
          prod_characters(prod),
          LL(LL_),
          SS(SS_),
          TT(TT_),
          MMT(MMT_),
          JJ(JJ_),
          MMJ(MMJ_),
          labels(labels_)
    {
      if (prod.empty())
        kappamax = 1;
      else
        kappamax = u3::Kappa(prod.back().u3, LL / 2);
    }

    // Get the number of ad's
    std::size_t num_ads() const { return prod_characters.size() + 1; }

    // Get shell number
    std::size_t shell() const { return n; }

    // Expand in terms of uncoupled products of ad's
    std::pair<std::vector<Eigen::RowVectorXd>, // each element has size ==
                                               // kappamax
              std::vector<U3xU4::ads_phys>>
    expand(qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah9& racah9,
           u3::WigU3SO3_db& wigU3SO3db,
           u4::WigU4SU2SU2_db& wigU4SU2SU2db) const;
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
    std::size_t n;

    const std::vector<u3::gelfand>& sps3;      // Gelfand array for [n,0,0]
    const std::vector<u3::gelfand>& sps3_conj; // Gelfand array for [n,n,0]

    const std::vector<u4::gelfand>& sps4;      // Gelfand array for [1,0,0,0]
    const std::vector<u4::gelfand>& sps4_conj; // Gelfand array for [1,1,1,0]

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U30 x U40) x at](U3 x U4)
    std::vector<U3xU4::irrep> prod_characters;

    // tensor component represented by two Gelfand patterns
    u3::gelfand g3;
    u4::gelfand g4;

  public:
    // Constructor
    Tat_gelfand(std::size_t n_, const std::vector<u3::gelfand>& sps3_,
                const std::vector<u3::gelfand>& sps3_conj_,
                const std::vector<u4::gelfand>& sps4_,
                const std::vector<u4::gelfand>& sps4_conj_,
                const std::vector<U3xU4::irrep>& prod, const u3::gelfand& b,
                const u4::gelfand& p)
        : n(n_),
          sps3(sps3_),
          sps3_conj(sps3_conj_),
          sps4(sps4_),
          sps4_conj(sps4_conj_),
          prod_characters(prod),
          g3(b),
          g4(p)
    { }

    // Get the number of at's
    std::size_t num_ats() const { return prod_characters.size() + 1; }

    // Get shell number
    std::size_t shell() const { return n; }

    // Expand in terms of uncoupled products of a's
    // (not at's because a phase is added)
    std::pair<std::vector<double>, std::vector<U3xU4::ats_gelfand>>
    expand(qcx_utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
           u4::WigU4U3_db& wigU4U3db) const;
  };

  /* Class for holding coupled products of at's, it can have up to 3 at's
     at, [at x at], [[at x at] x at]
     Tensor components are represented by kappa LL zeta SS TT MMT JJ MMJ
  */
  class Tat_phys
  {
  private:
    // A lookup table for the labels 
    HO::Table_nljmjmt& labels;

  public:
    // tensor character of at [n,n,0] x [1,1,1,0]
    int16_t n;

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U30 x U40) x at](U3 x U4)
    std::vector<U3xU4::irrep> prod_characters;

    // tensor components
    int16_t kappamax, LL, SS, TT, MMT, JJ, MMJ;

    // Constructor
    Tat_phys(int16_t n_, const std::vector<U3xU4::irrep>& prod, int16_t LL_,
             int16_t SS_, int16_t TT_, int16_t MMT_, int16_t JJ_, int16_t MMJ_,
             HO::Table_nljmjmt& labels_)
        : n(n_),
          prod_characters(prod),
          LL(LL_),
          SS(SS_),
          TT(TT_),
          MMT(MMT_),
          JJ(JJ_),
          MMJ(MMJ_),
          labels(labels_)
    {
      if (prod.empty())
        kappamax = 1;
      else
        kappamax = u3::Kappa(prod.back().u3, LL / 2);
    }

    // Get the number of at's
    std::size_t num_ats() const { return prod_characters.size() + 1; }

    // Get shell number
    std::size_t shell() const { return n; }

    // Expand in terms of uncoupled products of at's
    std::pair<std::vector<Eigen::RowVectorXd>, // each element has size ==
                                               // kappamax
              std::vector<U3xU4::ats_phys>>
    expand(qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah9& racah9,
           u3::WigU3SO3_db& wigU3SO3db,
           u4::WigU4SU2SU2_db& wigU4SU2SU2db) const;
  };

  ////// PRODUCT OF CREATION AND ANNIHILATION OPERATORS/TENSORS ///////

  /* Class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
     Tensor components are represented by Gelfand patterns
  */
  class Tadat_gelfand
  {
  private:
    // tensor character of ad [n,0,0] x [1,0,0,0]
    // tensor character of at [n,n,0] x [1,1,1,0]
    int16_t n;

    const std::vector<u3::gelfand>& sps3;      // Gelfand array for [n,0,0]
    const std::vector<u3::gelfand>& sps3_conj; // Gelfand array for [n,n,0]

    const std::vector<u4::gelfand>& sps4;      // Gelfand array for [1,0,0,0]
    const std::vector<u4::gelfand>& sps4_conj; // Gelfand array for [1,1,1,0]

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U3 x U4) x ad](U3_1 x U4_1)
    std::vector<U3xU4::irrep> prod_ads;
    // number of ad's
    uint8_t ads;
    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U3 x U4) x at](U3_2 x U4_2)
    std::vector<U3xU4::irrep> prod_ats;
    // number of at's
    uint8_t ats;

    // final tensor coupling
    U3xU4::irrep T;
    uint8_t rhomax, etamax;

    // tensor component represented by two Gelfand patterns
    u3::gelfand g3; // associated with T.u3
    u4::gelfand g4; // associated with T.u4

  public:
    // Constructor
    Tadat_gelfand(int16_t n_, const std::vector<u3::gelfand>& sps3_,
                  const std::vector<u3::gelfand>& sps3_conj_,
                  const std::vector<u4::gelfand>& sps4_,
                  const std::vector<u4::gelfand>& sps4_conj_, uint8_t ads_,
                  uint8_t ats_, const std::vector<U3xU4::irrep>& prod_ads_,
                  const std::vector<U3xU4::irrep>& prod_ats_,
                  const U3xU4::irrep& t, uint8_t Rhomax, uint8_t Etamax,
                  const u3::gelfand& b, const u4::gelfand& p)
        : n(n_),
          sps3(sps3_),
          sps3_conj(sps3_conj_),
          sps4(sps4_),
          sps4_conj(sps4_conj_),
          ads(ads_),
          ats(ats_),
          prod_ads(prod_ads_),
          prod_ats(prod_ats_),
          T(t),
          rhomax(Rhomax),
          etamax(Etamax),
          g3(b),
          g4(p)
    { }

    // Get the number of ad's
    uint8_t num_ads() const { return ads; }

    // Get the number of at's
    uint8_t num_ats() const { return ats; }

    // Get shell number
    int16_t shell() const { return n; }

    // Expand in terms of uncoupled products of ad's and a's
    // (not at's because a phase is added)
    // coeffs have rhomax x etamax columns, index of each column is
    // rho * etamax + eta
    std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
               std::vector<U3xU4::ads_gelfand>, // ad's
               std::vector<U3xU4::ats_gelfand>> // a's
    expand(qcx_utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
           u4::WigU4U3_db& wigU4U3db) const;
  };

  /* Class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
     Tensor components are represented by kappa LL zeta SS TT MMT JJ MMJ
  */
  class Tadat_phys
  {
  private:
    // A lookup table for the labels 
    HO::Table_nljmjmt& labels;

  public:
    // tensor character of ad [n,0,0] x [1,0,0,0]
    // tensor character of at [n,n,0] x [1,1,1,0]
    int16_t n;

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U3 x U4) x ad](U3_1 x U4_1)
    std::vector<U3xU4::irrep> prod_ads;
    // number of ad's
    uint8_t ads;

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U3 x U4) x at](U3_2 x U4_2)
    std::vector<U3xU4::irrep> prod_ats;
    // number of at's
    uint8_t ats;

    // final tensor coupling
    U3xU4::irrep T;
    uint8_t rhomax, etamax;

    // tensor components
    int16_t kappamax, LL, zetamax, SS, TT, MMT, JJ, MMJ;

    // Constructor
    Tadat_phys(int16_t n_, uint8_t ads_, uint8_t ats_,
               const std::vector<U3xU4::irrep>& prod_ads_,
               const std::vector<U3xU4::irrep>& prod_ats_,
               const U3xU4::irrep& t, uint8_t Rhomax, uint8_t Etamax,
               int16_t LL_, int16_t SS_, int16_t TT_, int16_t MMT_, int16_t JJ_,
               int16_t MMJ_, HO::Table_nljmjmt& labels_)
        : n(n_),
          ads(ads_),
          ats(ats_),
          prod_ads(prod_ads_),
          prod_ats(prod_ats_),
          T(t),
          rhomax(Rhomax),
          etamax(Etamax),
          LL(LL_),
          SS(SS_),
          TT(TT_),
          MMT(MMT_),
          JJ(JJ_),
          MMJ(MMJ_),
          labels(labels_)
    {
      kappamax = u3::Kappa(T.u3, LL / 2);
      zetamax = u4::Zeta(T.u4, SS, TT);
    }

    // Get the number of ad's
    uint8_t num_ads() const { return ads; }

    // Get the number of at's
    uint8_t num_ats() const { return ats; }

    // Get shell number
    int16_t shell() const { return n; }

    // Expand in terms of uncoupled products of ad's and a's
    // (not at's because a phase is added)
    // coeffs have rhomax x etamax x kappamax x zetamax columns, index of each
    // column is
    // ((rho * etamax + eta) * kappamax + kappa) * zetamax + zeta
    std::tuple<std::vector<Eigen::RowVectorXd>, // coeffs
               std::vector<U3xU4::ads_phys>,    // ad's
               std::vector<U3xU4::ats_phys>>    // a's
    expand(qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah9& racah9,
           u3::WigU3SO3_db& wigU3SO3db,
           u4::WigU4SU2SU2_db& wigU4SU2SU2db) const;
  };
  using Tadats_phys = std::vector<U3xU4::Tadat_phys>;
  using ads_ats_phys =
      std::vector<std::variant<U3xU4::ad_phys, U3xU4::at_phys>>;

  /* Class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
     Tensor components are not specified
  */
  class Tadat
  {
  public:
    // tensor character of ad [n,0,0] x [1,0,0,0]
    // tensor character of at [n,n,0] x [1,1,1,0]
    int16_t n;

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U3 x U4) x ad](U3_1 x U4_1)
    std::vector<U3xU4::irrep> prod_ads;
    // number of ad's
    uint8_t ads;

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U3 x U4) x at](U3_2 x U4_2)
    std::vector<U3xU4::irrep> prod_ats;
    // number of at's
    uint8_t ats;

    // final tensor coupling
    U3xU4::irrep T;
    uint8_t rho0max, eta0max;

    // Constructor
    Tadat(int16_t n_, uint8_t ads_, uint8_t ats_,
          const std::vector<U3xU4::irrep>& prod_ads_,
          const std::vector<U3xU4::irrep>& prod_ats_, const U3xU4::irrep& t,
          uint8_t rho0max_, uint8_t eta0max_)
        : n(n_),
          ads(ads_),
          ats(ats_),
          prod_ads(prod_ads_),
          prod_ats(prod_ats_),
          T(t),
          rho0max(rho0max_),
          eta0max(eta0max_)
    { }

    // Get the number of ad's
    uint8_t num_ads() const { return ads; }

    // Get the number of at's
    uint8_t num_ats() const { return ats; }

    // Get shell number
    int16_t shell() const { return n; }

    // Print out
    friend std::ostream& operator<<(std::ostream& s, const Tadat& x)
    {
      s << "[ (n=" << x.n << ") ";
      if (x.ads > 0)
      {
        s << static_cast<int>(x.ads) << "ads ";
        for (auto ad : x.prod_ads)
          s << ad << " ";
        if (x.ats > 0)
          s << "x ";
      }
      if (x.ats > 0)
      {
        s << static_cast<int>(x.ats) << "ats ";
        for (auto at : x.prod_ats)
          s << at << " ";
      }
      s << "] rho0max" << static_cast<int>(x.rho0max) << " eta0max"
        << static_cast<int>(x.eta0max) << x.T;
      return s;
    }

    /* Compute RMEs of single-shell U(3)xU(4) tensors
                              rho_0 U(3)_0
    < a_f U(3)_f U(4)_f ||| Tadat         ||| a_i U(3)_i U(4)_i >_{rho,eta}
                              eta_0 U(4)_0
      where rho_0 and eta_0 are from the structure of the tensor itself,
      whereas U(3)_i x U(3)_0 --> U(3)_f with mutliplicity rho
              U(4)_i x U(4)_0 --> U(4)_f with multiplicity eta

      These RMEs are stored in an array of matrices, the index of the matrix is
                  alpha_i * max(alpha_f) + alpha_f
      For each matrix:
                  row = rho * max(eta) + eta
                  column = rho_0 * max(eta_0) + eta_0

      Inputs:
      i3, i4: initial irreps
      f3, f4: final irreps
      cgcs, wigU3U2db, wigU4U3db: databases for coupling coefficients

      Outputs:
      RMEs stored in an array of matrices
      ai_max: outer multiplicity of i3
      af_max: outer multiplicity of f3
      rhomax, etamax: multiplicities of RMEs
    */
    std::vector<Eigen::MatrixXd> RME(std::size_t& ai_max, const U3xU4::irrep& i,
                                     std::size_t& af_max, const U3xU4::irrep& f,
                                     std::size_t& rhomax, std::size_t& etamax,
                                     qcx_utils::SO3_CGC& cgcs,
                                     u3::WigU3U2_db& wigU3U2db,
                                     u4::WigU4U3_db& wigU4U3db);
  };

  // Find all single-shell tensors composed of
  // num_ads creation tensors and num_ats annihilation tensors
  std::vector<U3xU4::Tadat> GenerateSingleShellTensors(int n, int num_ads,
                                                       int num_ats);

} // namespace U3xU4
