#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <so3Coeffs/SO3_CGC.h>

#include <U3basis.h>
#include <U3U2xU3U2.h>

#include <U4basis.h>
#include <U4U3xU4U3.h>

namespace U3xU4
{
  /* Class for defining creation operator a^dagger
     with tensor characters given by Gelfand pattern
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
    void print() const
    {
      std::cout << "[" << sps3[c] << "] x [" << sps4[r] << "]";
    }
  };

  /* Class for defining annihilation tensor a^tilde
     with tensor characters given by Gelfand pattern
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
    void print() const
    {
      if (tensor == true)
        std::cout << "[" << sps3_conj[c] << "] x [" << sps4_conj[r] << "]";
      else
        std::cout << "[" << sps3[c] << "] x [" << sps4[r] << "]";
    }
  };

  /* class for holding coupled products of ad's, it can have up to 3 ad's
     ad, [ad x ad], [[ad x ad] x ad]
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
    std::vector<std::pair<u3::irrep, u4::irrep>> prod_characters;

    // tensor component represented by two Gelfand patterns
    // which is associated with the last element of prod_characters
    u3::gelfand g3;
    u4::gelfand g4;

  public:
    // Constructor
    Tad_gelfand(std::size_t n_, const std::vector<u3::gelfand>& sps3_,
                const std::vector<u4::gelfand>& sps4_,
                const std::vector<std::pair<u3::irrep, u4::irrep>>& prod,
                const u3::gelfand& b, const u4::gelfand& p)
        : n(n_), sps3(sps3_), sps4(sps4_), prod_characters(prod), g3(b), g4(p)
    { }

    // Get the number of ad's
    std::size_t num_ads() const { return prod_characters.size() + 1; }

    // Get shell number
    std::size_t shell() const { return n; }

    // Expand in terms of uncoupled products of ad's
    std::pair<std::vector<double>, std::vector<std::vector<U3xU4::ad_gelfand>>>
    expand(u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
           u4::WigU4U3_db& wigU4U3db) const;
  };

  /* class for holding coupled products of at's, it can have up to 3 at's
     at, [at x at], [[at x at] x at]
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
    std::vector<std::pair<u3::irrep, u4::irrep>> prod_characters;

    // tensor component represented by two Gelfand patterns
    u3::gelfand g3;
    u4::gelfand g4;

  public:
    // Constructor
    Tat_gelfand(std::size_t n_, const std::vector<u3::gelfand>& sps3_,
                const std::vector<u3::gelfand>& sps3_conj_,
                const std::vector<u4::gelfand>& sps4_,
                const std::vector<u4::gelfand>& sps4_conj_,
                const std::vector<std::pair<u3::irrep, u4::irrep>>& prod,
                const u3::gelfand& b, const u4::gelfand& p)
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
    std::pair<std::vector<double>, std::vector<std::vector<U3xU4::at_gelfand>>>
    expand(u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
           u4::WigU4U3_db& wigU4U3db) const;
  };

  /* class for holding single-shell tensors, which are coupled products
     of ad's and at's. It can have up to 3 ad's and 3 at's
  */
  class Tadat_gelfand
  {
  private:
    // tensor character of ad [n,0,0] x [1,0,0,0]
    // tensor character of at [n,n,0] x [1,1,1,0]
    std::size_t n;

    const std::vector<u3::gelfand>& sps3;      // Gelfand array for [n,0,0]
    const std::vector<u3::gelfand>& sps3_conj; // Gelfand array for [n,n,0]

    const std::vector<u4::gelfand>& sps4;      // Gelfand array for [1,0,0,0]
    const std::vector<u4::gelfand>& sps4_conj; // Gelfand array for [1,1,1,0]

    // coupled tensor characters
    // 0 element for ad
    // 1 element for [ad x ad](U3 x U4)
    // 2 elements for [[ad x ad](U3 x U4) x ad](U3_1 x U4_1)
    std::vector<std::pair<u3::irrep, u4::irrep>> prod_ads;
    std::size_t ads;

    // coupled tensor characters
    // 0 element for at
    // 1 element for [at x at](U3 x U4)
    // 2 elements for [[at x at](U3 x U4) x at](U3_2 x U4_2)
    std::vector<std::pair<u3::irrep, u4::irrep>> prod_ats;
    std::size_t ats;

    // final tensor coupling
    u3::irrep T3;
    u4::irrep T4;
    std::size_t rhomax, etamax;

    // tensor component represented by two Gelfand patterns
    u3::gelfand g3; // associated with T3
    u4::gelfand g4; // associated with T4

  public:
    // Constructor
    Tadat_gelfand(std::size_t n_, const std::vector<u3::gelfand>& sps3_,
                  const std::vector<u3::gelfand>& sps3_conj_,
                  const std::vector<u4::gelfand>& sps4_,
                  const std::vector<u4::gelfand>& sps4_conj_, std::size_t ads_,
                  std::size_t ats_,
                  const std::vector<std::pair<u3::irrep, u4::irrep>>& prod_ads_,
                  const std::vector<std::pair<u3::irrep, u4::irrep>>& prod_ats_,
                  const u3::irrep& t3, const u3::gelfand& b, std::size_t Rhomax,
                  const u4::irrep& t4, const u4::gelfand& p, std::size_t Etamax)
        : n(n_),
          sps3(sps3_),
          sps3_conj(sps3_conj_),
          sps4(sps4_),
          sps4_conj(sps4_conj_),
          ads(ads_),
          ats(ats_),
          prod_ads(prod_ads_),
          prod_ats(prod_ats_),
          T3(t3),
          g3(b),
          rhomax(Rhomax),
          T4(t4),
          g4(p),
          etamax(Etamax)
    { }

    // Get the number of ad's
    std::size_t num_ads() const { return ads; }

    // Get the number of at's
    std::size_t num_ats() const { return ats; }

    // Get shell number
    std::size_t shell() const { return n; }

    // Expand in terms of uncoupled products of ad's and a's
    // (not at's because a phase is added)
    // coeffs have rhomax x etamax columns, index of each column is
    // rho * etamax + eta
    std::tuple<std::vector<Eigen::RowVectorXd>,             // coeffs
               std::vector<std::vector<U3xU4::ad_gelfand>>, // ad's
               std::vector<std::vector<U3xU4::at_gelfand>>> // a's
    expand(u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
           u4::WigU4U3_db& wigU4U3db);
  };
} // namespace U3xU4