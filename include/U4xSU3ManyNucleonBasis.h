#ifndef SAM_U4XSU3_MANY_NUCLEON_BASIS_H
#define SAM_U4XSU3_MANY_NUCLEON_BASIS_H

#include <cassert>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <U3irrep.h>
#include <U4irrep.h>
#include <U4xSU3Irrep.h>

#ifdef HAVE_FMT
#include <fmt/format.h>
#endif

namespace U4xSU3
{
  /**
   * Class that represents the many nucleon basis.
   */
  class ManyNucleonBasis
  {
  public:
    // **** PUBLIC MEMBER TYPES ****

    // (1) distributions:

    /**
     * Type for storing a number of particles in a shell.
     */
    using DistributionElement = uint16_t;

    // (2) gammas:

    /**
     * Class that represent a single basis gamma element.
     */
    class GammaElement : public U4xSU3::irrep
    {
      uint16_t alphamax_; // U(N)->U(3) reduction multiplicity

    public:
      /**
       * Default constructor.
       *
       * Creates a vaccum gamma element, where:
       *   u4 = [0,0,0,0],
       *   alphamax = 1,
       *   su3 = (0, 0).
       */
      GammaElement();

      /**
       * Consctuctor that creates a gamma element with given parameters.
       *
       * @param u4 U(4) irrep.
       * @param alphamax U(N)->U(3) reduction multiplicity.
       * @param su3 SU(3) irrep.
       */
      GammaElement(const u4::irrep& u4, int alphamax, const su3::irrep& su3);

      /**
       * Sets U(N)->(3) reduction multiplicity.
       *
       * @param am U(N)->U(3) reduction multiplicity.
       */
      void alphamax(int am);

      /**
       * Returns U(N)->U(3) recution multiplicity.
       *
       * @return U(N)->U(3) recuction multiplicity.
       */
      int alphamax() const;
    };

    // (3) omegas:

  private:
    // the following declarations must precede OmegaElement class:
    class OmegaElementU4;
    class OmegaElementSU3;

  public:
    /**
     * Class that represent a single basis omega element.
     */
    class OmegaElement : public U4xSU3::irrep
    {
      uint16_t etamax_; // U(4) coupling multiplicity
      uint16_t rhomax_; // SU(3) coupling multiplicity

    public:
      /**
       * Default constructor.
       *
       * Creates a vaccum omega element, where:
       *   etamax = 1,
       *   u4 = [0,0,0,0],
       *   rhomax = 1,
       *   su3 = (0, 0).
       */
      OmegaElement();

      /**
       * Consctuctor that creates an omega element with given parameters.
       *
       * @param etamax U(4) couling multiplicity.
       * @param u4 U(4) irrep.
       * @param rhomax SU(3) couling multiplicity.
       * @param su3 SU(3) irrep.
       */
      OmegaElement(int etamax, const u4::irrep& u4, int rhomax,
                   const su3::irrep& su3);

      /**
       * Consctuctor that creates an omega element from its U(4) and SU(3)
       * parts.
       *
       * @param oe_u4 U(4) part.
       * @param oe_su3 SU(3) part.
       */
      OmegaElement(const OmegaElementU4& oe_u4, const OmegaElementSU3& oe_su3);

      /**
       * Consctuctor that creates an omega element from a gamma element.
       *
       * Irreps are copied, multiplicities are set to 1.
       *
       * @param ge Gamma element.
       */
      OmegaElement(const GammaElement& ge);

      /**
       * Sets U(4) coupling multiplicity.
       *
       * @param em U(4) coupling multiplicity.
       */
      void etamax(int em);

      /**
       * Sets SU(3) coupling multiplicity.
       *
       * @param rm SU(3) coupling multiplicity.
       */
      void rhomax(int rm);

      /**
       * Returns U(4) coupling multiplicity.
       *
       * @return U(4) coupling multiplicity.
       */
      int etamax() const;

      /**
       * Returns SU(3) coupling multiplicity.
       *
       * @return SU(3) coupling multiplicity.
       */
      int rhomax() const;
    };

    // (4) irreps:

    /**
     * Single basis irrep.
     */
    struct Irrep
    {
      std::span<const DistributionElement> dist;
      std::span<const GammaElement> gamma;
      std::span<const OmegaElement> omega;
      uint64_t offset;
    };

    /**
     * Database of basis irreps.
     */
    class IrrepDB
    {
      const ManyNucleonBasis& basis_;

      // storage for resulting combined omega elements:
      std::vector<OmegaElement> omega_;

    public:
      /**
       * Constructor.
       *
       * @param basis Basis to which irrep database is related.
       */
      IrrepDB(const ManyNucleonBasis& basis);

      /**
       * Size of the irrep database.
       *
       * @return Number of basis irreps.
       */
      auto size() const { return basis_.n_irreps(); }

      /**
       * Gets a basis irrep.
       *
       * @param irrep_id ID of the irrep to be returned.
       *
       * @return A required basis irrep.
       */
      Irrep operator[](size_t irrep_id);
    };

    // **** PUBLIC MEMBER FUNCTIONS  ****

    /**
     * Constructor that sets the model space parameters.
     *
     * @param Z Number of protons.
     * @param N Number of neutrons.
     * @param Nmax Nmax parameter.
     * @param JJ Total angular momemtum J multiplied by 2.
     */
    ManyNucleonBasis(int Z, int N, int Nmax, int JJ);

    /**
     * Basis construction.
     */
    void construct();

    /**
     * Database of basis irreps.
     *
     * @returns Database of basis irreps.
     */
    auto irreps() const { return IrrepDB(*this); }

    /**
     * Basis dimension.
     *
     * @returns Dimension of the basis.
     */
    auto dim() const { return dim_; }

  private:
    // **** PRIVATE MEMBER TYPES ****

    class OmegaElementU4
    {
      uint16_t etamax_; // U(4) coupling multiplicity
      u4::irrep u4_;

    public:
      OmegaElementU4();
      OmegaElementU4(int etamax, const u4::irrep& u4);

      void etamax(int em);
      int etamax() const;

      void u4(const u4::irrep& u4);
      const u4::irrep& u4() const;
    };

    class OmegaElementSU3
    {
      uint16_t rhomax_; // SU(3) coupling multiplicity
      su3::irrep su3_;

    public:
      OmegaElementSU3();
      OmegaElementSU3(int rhomax, const su3::irrep& su3);

      void rhomax(int rm);
      int rhomax() const;

      void su3(const su3::irrep& su3);
      const su3::irrep& su3() const;
    };

    struct InternalIrrep
    {
      uint32_t dist_id;
      uint32_t gamma_id;
      uint32_t omega_u4_id;
      uint32_t omega_su3_id;
      uint64_t offset;
    };

    // **** PRIVATE MEMBER VARIABLES ****

    int A_;   // number of nucleons; equal to (Z+N)
    int MMT_; // proton-neutron "difference"; equal to (abs(Z-N))
    int Nmax_;
    int JJ_;

    int n0hw_; // lowest considered HO shell; 0 for even Nmax, 1 for odd Nmax

    size_t dim_; // basis dimension

    // (1) distributions:

    int n0max_; // highest occupied HO shell for 0hw

    int nmax_; // highest occupied HO shell (for Nmaxhw); equal to
               // (n0max_+Nmax_)

    size_t D_; // number of shells in each distribution; equal to (nmax_+1)

    std::vector<DistributionElement>
        shells_; // occupancy of shells for distributions

    // (2) gammas:

    std::vector<GammaElement> gamma_elements_;
    std::vector<std::pair<size_t, size_t>> gammas_;

    // (3) omegas:

    std::vector<OmegaElementU4> omega_elements_u4_;
    std::vector<std::pair<size_t, size_t>> omegas_u4_;

    std::vector<OmegaElementSU3> omega_elements_su3_;
    std::vector<std::pair<size_t, size_t>> omegas_su3_;

    inline static const uint32_t INVALID_OMEGA_ID =
        std::numeric_limits<uint32_t>::max();

    // (4) irreps:

    std::vector<InternalIrrep> irreps_;

    // **** PRIVATE MEMBER FUNCTIONS ****

    // (1) distributions:

    /**
     * Number of basis distributions.
     *
     * @return Number of distributions.
     */
    auto n_distributions() const
    {
      assert(shells_.size() % D_ == 0);
      return shells_.size() / D_;
    }

    /**
     * Returns a distribution by ID.
     *
     * @param dist_id ID of the required distribution.
     *
     * @return Required distribution.
     */
    auto distribution(size_t dist_id) const
    {
      auto end_index = D_ * (dist_id + 1) - 1;
      assert(end_index < shells_.size());

      return std::span(shells_.data() + D_ * dist_id, D_);
    }

    // (2) gammas:

    /**
     * Number of gammas in the basis.
     *
     * @return Number of gammas.
     */
    auto n_gammas() const { return gammas_.size(); }

    /**
     * Returns a gamma by ID.
     *
     * @param gamma_id ID of the required gamma.
     * 
     * @return Required gamma.
     */
    auto gamma(size_t gamma_id) const
    {
      assert(gamma_id < gammas_.size());

      auto begin_index = gammas_[gamma_id].first;
      auto end_index = gammas_[gamma_id].second;

      assert(end_index <= gamma_elements_.size());

      return std::span(gamma_elements_.data() + begin_index,
                       gamma_elements_.data() + end_index);
    }

    // (3) omegas:

    /**
     * Returns a U(4) omega by ID.
     *
     * @param omega_u4_id ID of the required U(4) omega.
     * 
     * @return Required U(4) omega.
     */
    auto omega_u4(size_t omega_u4_id) const
    {
      assert(omega_u4_id != INVALID_OMEGA_ID);
      assert(omega_u4_id < omegas_u4_.size());

      auto begin_index = omegas_u4_[omega_u4_id].first;
      auto end_index = omegas_u4_[omega_u4_id].second;

      assert(end_index <= omega_elements_u4_.size());

      return std::span(omega_elements_u4_.data() + begin_index,
                       omega_elements_u4_.data() + end_index);
    }

    /**
     * Returns a SU(3) omega by ID.
     *
     * @param omega_su3_id ID of the required SU(3) omega.
     * 
     * @return Required SU(3) omega.
     */
    auto omega_su3(size_t omega_su3_id) const
    {
      assert(omega_su3_id != INVALID_OMEGA_ID);
      assert(omega_su3_id < omegas_su3_.size());

      auto begin_index = omegas_su3_[omega_su3_id].first;
      auto end_index = omegas_su3_[omega_su3_id].second;

      assert(end_index <= omega_elements_su3_.size());

      return std::span(omega_elements_su3_.data() + begin_index,
                       omega_elements_su3_.data() + end_index);
    }

    // (4) irreps:

    /**
     * Number of basis irreps.
     * 
     * @return Number of irreps in the basis.
     */
    size_t n_irreps() const;

    // (5) basis construction distributions:

    /**
     * Generates model space distributions.
     *
     * Generate all possible distributions of the model space according to A and
     * Nmax.
     */
    void generate_distributions();

    /**
     * Generates all the basis gammas.
     */
    void generate_gammas();

    /**
     * Generates gammas (and omegas) for a specific basis distribution.
     *
     * @param dist_id ID of the required distribution for which to generate
     * gammas.
     */
    void generate_gammas_for_distribution(size_t dist_id);

    /**
     * Generates gamma elements for a single shell.
     *
     * @param n Shell number.
     * @param A Number of particles in the shell.
     * @param gamma_elements Resulting vector of gamma elements.
     *
     * @note To avoid repetitive memory allocations, the vector where the
     * generated gamma elements are stored is passed by reference.
     */
    void generate_single_shell_gamma_elements(
        int n, int A, std::vector<GammaElement>& gamma_elements);

    /**
     * Generates U(4) irreps for A particles in nth shell.
     *
     * @param n Shell number.
     * @param A Number of particles in teh shell.
     * @param u4_irreps Resulting vector of U(4) irreps.
     *
     * @note To avoid repetitive memory allocations, the vector where the
     * generated U(4) irreps are stored is passed by reference.
     */
    void generate_U4_irreps(int n, int A, std::vector<u4::irrep>& u4_irreps);

    /**
     * Calculates the dimension of a pair of U(4) and SU(3) irreps.
     *
     * @param u4 U(4) irrep.
     * @param su3 SU(3) irrep.
     *
     * @return Dimension.
     */
    int compute_dim(const u4::irrep& u4, const su3::irrep& su3);

    /**
     * Calculates the dimension of an SU(3) irrep and spin.
     *
     * @param su3 SU(3) irrep.
     * @param SS Spin multiplied by 2.
     *
     * @return Dimension.
     */
    int compute_dim(const su3::irrep& su3, int SS);

    /**
     * Generates omegas for a gamma.
     *
     * @param dist_id ID of the gamma distribution.
     * @param gamma_id ID of the required gamma for which to generate omegas.
     */
    void generate_omegas_for_gamma(size_t dist_id, size_t gamma_id);
  };

  // structure for irrep used outside of the basis class:
  struct Irrep
  {
    std::vector<ManyNucleonBasis::DistributionElement> dist;
    std::vector<ManyNucleonBasis::GammaElement> gamma;
    std::vector<ManyNucleonBasis::OmegaElement> omega;
    std::vector<uint16_t> gamma_shells;

    inline static const size_t INVALID_INDEX =
        std::numeric_limits<size_t>::max();

    Irrep() = default;

    Irrep(const std::vector<ManyNucleonBasis::DistributionElement>& dist_,
          const std::vector<ManyNucleonBasis::GammaElement>& gamma_,
          const std::vector<ManyNucleonBasis::OmegaElement>& omega_)
      : dist(dist_), gamma(gamma_), omega(omega_)
    {}

    size_t gamma_shell(size_t gamma_idx) const
    {
      assert(gamma_idx < gamma.size());

      return gamma_shells[gamma_idx];
    }

    size_t shell_gamma(size_t shell_idx) const
    {
      assert(shell_idx < dist.size());

      auto iter = std::ranges::find(gamma_shells, shell_idx);

      return (iter == gamma_shells.end() ? INVALID_INDEX : *iter);
    }

    void clear()
    {
      dist.clear();
      gamma.clear();
      omega.clear();
    }
  };

  // class for unifying 2 irreps wrt to their active shells:
  class IrrepUnifier
  {
    Irrep o1_, o2_;

  public:
    auto unify(const auto& i1, const auto& i2)
    {
      // (1) setup:

      auto& o1 = o1_;
      auto& o2 = o2_;

      o1.clear();
      o2.clear();

      size_t g1 = 0; // igamma1 element index
      size_t g2 = 0; // igamma2 element index

      static const ManyNucleonBasis::GammaElement ge_vacuum;
      assert(ge_vacuum.vacuum());

      static const ManyNucleonBasis::OmegaElement oe_vacuum;
      assert(oe_vacuum.vacuum());

      // vacuum-aware coupling:
      auto couple = [](const auto& first, const auto& second,
                       const auto& iomega,
                       size_t g) -> ManyNucleonBasis::OmegaElement
      {
        if (first.vacuum() && second.vacuum())
          return oe_vacuum;
        else if (first.vacuum())
        {
          // couplig with vacuum results in rhomax = etamax = 1:
          ManyNucleonBasis::OmegaElement copy = second;
          copy.rhomax(1);
          copy.etamax(1);
          return copy;
        }
        else if (second.vacuum())
        {
          // ditto:
          ManyNucleonBasis::OmegaElement copy = first;
          copy.rhomax(1);
          copy.etamax(1);
          return copy;
        }
        else
          return iomega[g - 2];
      };

      // (2) unify distribution sizes:

      std::copy(i1.dist.begin(), i1.dist.end(), std::back_inserter(o1.dist));
      std::copy(i2.dist.begin(), i2.dist.end(), std::back_inserter(o2.dist));

      auto D = std::max(o1.dist.size(), o2.dist.size());

      assert(D > 0);

      o1.dist.resize(D);
      o2.dist.resize(D);

      // (3) unify gammas and omegas:

      for (size_t d = 0; d < D; d++)
      {
        // nothing to do if both shells are empty:
        if ((o1.dist[d] == 0) && (o2.dist[d] == 0))
          continue;

        // add element to gamma1:
        o1.gamma.push_back(o1.dist[d] == 0 ? ge_vacuum : i1.gamma[g1++]);
        o1.gamma_shells.push_back(d);

        // add element to omega1:
        if (o1.gamma.size() == 2)
          // couple first 2 gamma elements:
          o1.omega.push_back(couple(o1.gamma[0], o1.gamma[1], i1.omega, g1));
        else if (o1.gamma.size() > 2)
          // couple last gamma element with last omega element:
          o1.omega.push_back(
              couple(o1.gamma.back(), o1.omega.back(), i1.omega, g1));

        // add element to gamma2:
        o2.gamma.push_back(o2.dist[d] == 0 ? ge_vacuum : i2.gamma[g2++]);
        o2.gamma_shells.push_back(d);

        // add element to omega2:
        if (o2.gamma.size() == 2)
          // couple first 2 gamma elements:
          o2.omega.push_back(couple(o2.gamma[0], o2.gamma[1], i2.omega, g2));
        else if (o2.gamma.size() > 2)
          // couple last gamma element with last omega element:
          o2.omega.push_back(
              couple(o2.gamma.back(), o2.omega.back(), i2.omega, g2));
      }

      // (4) return result:

      return std::tie(std::as_const(o1_), std::as_const(o2_));
    }
  };
} // namespace U4xSU3

#ifdef HAVE_FMT

// formatter for GammaElement:

template <>
class fmt::formatter<U4xSU3::ManyNucleonBasis::GammaElement>
{
public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename Context>
  constexpr auto format(const U4xSU3::ManyNucleonBasis::GammaElement& ge,
                        Context& ctx) const
  {
    return format_to(ctx.out(), "[{},{},{},{}]{}({},{})", ge.u4().n14(),
                     ge.u4().n24(), ge.u4().n34(), ge.u4().n44(), ge.alphamax(),
                     ge.su3().lambda(), ge.su3().mu());
  }
};

/*
// formatter for OmegaElementU4:

template <>
class fmt::formatter<U4xSU3::ManyNucleonBasis::OmegaElementU4>
{
public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename Context>
  constexpr auto format(const U4xSU3::ManyNucleonBasis::OmegaElementU4& oe,
                        Context& ctx) const
  {
    return format_to(ctx.out(), "{}[{},{},{},{}]", oe.etamax(), oe.u4().n14(),
                     oe.u4().n24(), oe.u4().n34(), oe.u4().n44());
  }
};

// formatter for OmegaElementSU3:

template <>
class fmt::formatter<U4xSU3::ManyNucleonBasis::OmegaElementSU3>
{
public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename Context>
  constexpr auto format(const U4xSU3::ManyNucleonBasis::OmegaElementSU3& oe,
                        Context& ctx) const
  {
    return format_to(ctx.out(), "{}({},{})", oe.rhomax(), oe.su3().lambda(),
                     oe.su3().mu());
  }
};
*/

    // formatter for OmegaElement:

    template <>
    class fmt::formatter<U4xSU3::ManyNucleonBasis::OmegaElement>
    {
    public:
      constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

      template <typename Context>
      constexpr auto format(const U4xSU3::ManyNucleonBasis::OmegaElement& oe,
                            Context& ctx) const
      {
        return format_to(ctx.out(), "{}[{},{},{},{}]{}({},{})", oe.etamax(),
                         oe.u4().n14(), oe.u4().n24(), oe.u4().n34(),
                         oe.u4().n44(), oe.rhomax(), oe.su3().lambda(),
                         oe.su3().mu());
      }
};

#endif

#endif
