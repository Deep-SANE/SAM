#ifndef SAM_U4XSU3_MANY_NUCLEON_BASIS_H
#define SAM_U4XSU3_MANY_NUCLEON_BASIS_H

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <qcx_convert.h>
#include <qcx_Int.h>

#include <U3irrep.h>
#include <U4irrep.h>

#include <HOutilities.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4gelfand.h>
#include <U3xU4irrep.h>

namespace U3xU4
{
  /**
   * Class that represents a particular hw sub model space.
   */
  struct hw_subspace
  {
    // Members
    HO::qn_t hw;

    u3::irreps u3s;
    u4::irreps u4s;
    std::vector<HO::qn_t> SSs, TTs;

    // 2D array for storing probabilities of multiple states
    // only non-empty after analyzing w.f.
    std::vector<std::vector<double>> probs; 

    // Constructors
    hw_subspace() = default;
    hw_subspace(qcx_utils::Int8 hw_, const u3::irreps& u3s_, const u4::irreps& u4s_,
      const std::vector<HO::qn_t>& SSs_, const std::vector<HO::qn_t>& TTs_)
        : hw(hw_), u3s(u3s_), u4s(u4s_), SSs(SSs_), TTs(TTs_)
    {}
    hw_subspace(qcx_utils::Int8 hw_, const u3::irreps& u3s_, const u4::irreps& u4s_,
      const std::vector<HO::qn_t>& SSs_, const std::vector<HO::qn_t>& TTs_,
                const std::vector<std::vector<double>>& probs_)
        : hw(hw_), u3s(u3s_), u4s(u4s_), SSs(SSs_), TTs(TTs_), probs(probs_)
    {}

    // selection rules: used during truncation
    // if u3s is empty, no selection is performed
    bool U3_selection(const u3::irrep& irr) const;
    // if u4s is empty, no selection is performed
    bool U4_selection(const u4::irrep& irr) const;
    // if SSs is empty, no selection is performed
    bool SS_selection(qcx_utils::Int8 SS) const;
    // if TTs is empty, no selection is performed
    bool TT_selection(qcx_utils::Int8 TT) const;

    // sort u3s in decreasing order of C2; u4s, SSs, TTs must be empty
    // if probs is not empty, it will be sorted too
    void sort_u3_by_C2_decreasing();

    // sort u4s in increasing order of C2; u3s, SSs, TTs must be empty
    // if probs is not empty, it will be sorted too
    void sort_u4_by_C2_increasing();

    // sort SSs in increasing order; u3s, u4s, TTs must be empty
    // if probs is not empty, it will be sorted too
    void sort_SS();

    // sort TTs in increasing order; u3s, u4s, SSs must be empty
    // if probs is not empty, it will be sorted too
    void sort_TT();

    // print to terminal
    friend std::ostream& operator<<(std::ostream& os, const hw_subspace& s);

    // print to file
    void print_to_file(std::ostream& os) const;
  };

  /**
   * Class that represents the many nucleon basis.
   */
  class ManyNucleonBasis
  {
  public:
    // **** PUBLIC MEMBER TYPES ****

    /**
     * Single basis irrep.
     */
    struct Irrep
    {
      std::vector<U3xU4::GammaElement> gamma;
      std::vector<U3xU4::OmegaElement> omega;

      size_t alphamax() const
      {
        size_t res = 1;
        for (const auto& g : gamma) res *= g.alphamax();
        for (const auto& o : omega) res *= (o.rhomax() * o.etamax());
        return res;
      }
    };

    /**
     * Database of basis irreps.
     */
    class IrrepDB
    {
      const ManyNucleonBasis& basis_;

      // storage for resulting gamma and omega elements:
      std::vector<uint16_t> active_shells_;
      std::vector<U3xU4::GammaElement> gamma_;
      std::vector<U3xU4::OmegaElement> omega_;

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

      size_t offset(size_t irrep_id) const;
    };

    // **** PUBLIC MEMBER FUNCTIONS  ****

    /**
     * Constructor that sets the model space parameters.
     *
     * @param N Number of neutrons.
     * @param Z Number of protons.
     * @param Nmax Nmax parameter.
     * @param JJ Total angular momemtum J multiplied by 2.
     */
    ManyNucleonBasis(int N, int Z, int Nmax, int JJ);

    /**
     * Constructor that sets the model space parameters.
     *
     * @param N Number of neutrons.
     * @param Z Number of protons.
     * @param Nmax Nmax parameter.
     * @param JJ Total angular momemtum J multiplied by 2.
     * @param 
     */
    ManyNucleonBasis(int N, int Z, int Nmax, int JJ,
                     const std::vector<hw_subspace>& hw_subspaces);

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

    using DistributionElement = uint16_t;

     // class that represents a single basis gamma element:
    struct GammaElement
    {
      u4::irrep u4;
      uint16_t alphamax; // U(N)->U(3) reduction multiplicity
      su3::irrep su3;

    public:
      GammaElement() : alphamax(1) { }

      GammaElement(const u4::irrep& u4, int alphamax, const su3::irrep& su3)
          : u4(u4), alphamax(qcx_utils::itou16(alphamax)), su3(su3)
      { }
    };

    // class that represents an SU(3) part of a single basis omega element:
    struct OmegaElementSU3
    {
      uint16_t rhomax; // SU(3) coupling multiplicity
      su3::irrep su3;

    public:
      OmegaElementSU3() : rhomax(1) { }

      OmegaElementSU3(int rhomax, const su3::irrep& su3)
          : rhomax(qcx_utils::itou16(rhomax)), su3(su3)
      { }
    };

    // class that represents a U(4) part of a single basis omega element:
    struct OmegaElementU4
    {
      uint16_t etamax; // U(4) coupling multiplicity
      u4::irrep u4;

    public:
      OmegaElementU4() : etamax(1) { }

      OmegaElementU4(int etamax, const u4::irrep& u4)
          : etamax(qcx_utils::itou16(etamax)), u4(u4)
      { }
    };

    // internal representation of basis irreps:
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
    int MMT_; // proton-neutron "difference"; N-Z
    int Nmax_;
    int JJ_;

    int n0hw_; // lowest considered HO shell; 0 for even Nmax, 1 for odd Nmax

    size_t dim_; // basis dimension

    // (1) distributions:

    int n0max_; // highest occupied HO shell for 0hw

    int nmax_; // highest occupied HO shell (for Nmaxhw); equal to
               // (n0max_ + Nmax_)

    int quanta_0hw; // total quanta at 0hw

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

    // (5) subgroup labels:

    std::vector<std::vector<std::array<int,6>>> LLSSTTs_;

    // (6) sub-model spaces:

    std::vector<hw_subspace> hw_subspaces_;

    // (7) grouping ID's into distribution spaces

    std::vector<                                      // index = dist_id
      std::vector<                      
        std::pair<uint32_t,                           // gamma_id
                  std::vector<size_t>>>> group_dists; // list of irrep_id's


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
     * Calculates the dimension of a pair of U(4) and SU(3) irreps and
     * returns subgroup labels (LL, SS, TT).
     *
     * @param hw subspace Nhw
     * @param u4 U(4) irrep.
     * @param su3 SU(3) irrep.
     *
     * @return Dimension.
     */
    std::pair<std::vector<std::array<int,6>>, int> 
      decomposition(int hw, const u4::irrep& u4, const su3::irrep& su3);

    /**
     * Generates omegas for a gamma.
     *
     * @param dist_id ID of the gamma distribution.
     * @param gamma_id ID of the required gamma for which to generate omegas.
     */
    void generate_omegas_for_gamma(size_t dist_id, size_t gamma_id,
                                   std::vector<size_t>& irrep_ids);
  
  public:
    // get twice angular momentum
    int JJ() const { return static_cast<int>(JJ_); }

    // get twice isospin projection
    int MMT() const { return static_cast<int>(MMT_); }

    // get valence shell
    int valenceShell() const { return n0max_; }

    // get maxShell
    int maxShell() const { return nmax_; }

    // determine if this model space is truncated
    bool isTruncated() { return !hw_subspaces_.empty(); }

    // perform decomposition to LL,SS,TT
    // std::vector<std::vector<std::array<int,6>>> decomposition() const;
    const std::vector<std::vector<std::array<int,6>>>& decomposition() const
    {
      return LLSSTTs_;
    }
    std::vector<std::array<int,6>> decomposition(const Irrep& irr) const;

    /////////////////// COMPUTING MATRIX ELEMENTS ///////////////////

    // Symmetric matrix
    /*
    friend Eigen::SparseMatrix<double> MatrixElements(
      const std::vector<MultiShellTensor_phys>& tensors, 
      const U3xU4::ManyNucleonBasis& basis,
      const U3xU4::Table_gelfand& gelfand_labels);

    friend Eigen::SparseMatrix<double> MatrixElements(
      const std::vector<MultiShellTensor_ShareGammas_phys>& tensors, 
      const U3xU4::ManyNucleonBasis& basis,
      const U3xU4::Table_gelfand& gelfand_labels);
    */
    template <typename TensorType>
    friend Eigen::SparseMatrix<double> MatrixElements(
      const std::vector<TensorType>& tensors, 
      const U3xU4::ManyNucleonBasis& basis,
      const U3xU4::Table_gelfand& gelfand_labels);

    // Rectangular matrix
    template <typename TensorType>
    friend Eigen::SparseMatrix<double> MatrixElements(
      const std::vector<TensorType>& tensors, 
      const U3xU4::ManyNucleonBasis& bras,
      const U3xU4::ManyNucleonBasis& kets,
      const U3xU4::Table_gelfand& gelfand_labels);
  };

  // class for unifying 2 irreps wrt to their shells:
  class IrrepUnifier
  {
    ManyNucleonBasis::Irrep o1_, o2_;

  public:
    auto unify(const ManyNucleonBasis::Irrep& i1,
               const ManyNucleonBasis::Irrep& i2)
    {
      // (1) setup:

      auto& o1 = o1_;
      auto& o2 = o2_;

      o1.gamma.clear();
      o1.omega.clear();

      o2.gamma.clear();
      o2.omega.clear();

      // (2) unify gamma elements:

      assert(i1.gamma.size() > 0);
      assert(i2.gamma.size() > 0);

      size_t g1 = 0; // igamma1 element index
      size_t g2 = 0; // igamma2 element index

      while ((g1 < i1.gamma.size()) || (g2 < i2.gamma.size()))
      {
        if ((g1 < i1.gamma.size()) && (g2 < i2.gamma.size()))
        {
          assert(i1.gamma[g1].vacuum() == false);
          assert(i2.gamma[g2].vacuum() == false);

          if (i1.gamma[g1].shell() == i2.gamma[g2].shell())
          {
            o1.gamma.push_back(i1.gamma[g1++]);
            o2.gamma.push_back(i2.gamma[g2++]);
          }
          else if (i1.gamma[g1].shell() < i2.gamma[g2].shell())
          {
            o2.gamma.emplace_back(i1.gamma[g1].shell(), 1, u3::irrep{}, u4::irrep{});
            o1.gamma.push_back(i1.gamma[g1++]);
          }
          else // if (i1.gamma[g1].shell() > i2.gamma[g2].shell())
          {
            o1.gamma.emplace_back(i2.gamma[g2].shell(), 1, u3::irrep{}, u4::irrep{});
            o2.gamma.push_back(i2.gamma[g2++]);
          }
        }
        else if (g1 < i1.gamma.size())
        {
          assert(i1.gamma[g1].vacuum() == false);

          o2.gamma.emplace_back(i1.gamma[g1].shell(), 1, u3::irrep{}, u4::irrep{});
          o1.gamma.push_back(i1.gamma[g1++]);
        }
        else // if (g2 < i2.gamma.size())
        {
          assert(i2.gamma[g2].vacuum() == false);

          o1.gamma.emplace_back(i2.gamma[g2].shell(), 1, u3::irrep{}, u4::irrep{});
          o2.gamma.push_back(i2.gamma[g2++]);
        }
      }

      assert(o1.gamma.size() == o2.gamma.size());

      // check unification of shells:
      for (size_t i = 0; i < o1.gamma.size(); i++)
        assert(o1.gamma[i].shell() == o2.gamma[i].shell());

      // (3) setup coupling for omega elements:

      auto couple = [](const U3xU4::irrep A, const U3xU4::irrep B,
                       const U3xU4::OmegaElement* poe) -> U3xU4::OmegaElement
      {
        if ((A.vacuum()) && (B.vacuum()))
          return {1, {}, 1, {}};
        else if (A.vacuum())
        {
          u3::irrep u3 = B.u3();
          u3.normalize();
          return {1, u3, 1, B.u4()};
        }
        else if (B.vacuum())
        {
          u3::irrep u3 = A.u3();
          u3.normalize();
          return {1, u3, 1, A.u4()};
        }
        else
        {
          assert(poe != nullptr);
          return *poe;
        }
      };

      // (4) reconstruct omega elements for bra:

      size_t om1 = 0;

      for (size_t g1 = 1; g1 < o1.gamma.size(); g1++)
      {
        assert(o1.omega.size() == g1 - 1);

        const U3xU4::irrep& A =
            (g1 == 1) ? static_cast<const U3xU4::irrep&>(o1.gamma[0])
                      : static_cast<const U3xU4::irrep&>(o1.omega.back());

        const U3xU4::irrep& B = o1.gamma[g1];

        const U3xU4::OmegaElement* poe = nullptr;
        if ((A.vacuum() == false) && (B.vacuum() == false))
        {
          assert(om1 < i1.omega.size());
          poe = i1.omega.data() + om1++;
        }

        o1.omega.push_back(couple(A, B, poe));
      }

      // (5) reconstruct omega elements for ket:

      size_t om2 = 0;

      for (size_t g2 = 1; g2 < o2.gamma.size(); g2++)
      {
        assert(o2.omega.size() == g2 - 1);

        const U3xU4::irrep& A =
            (g2 == 1) ? static_cast<const U3xU4::irrep&>(o2.gamma[0])
                      : static_cast<const U3xU4::irrep&>(o2.omega.back());

        const U3xU4::irrep& B = o2.gamma[g2];

        const U3xU4::OmegaElement* poe = nullptr;
        if ((A.vacuum() == false) && (B.vacuum() == false))
        {
          assert(om2 < i2.omega.size());
          poe = i2.omega.data() + om2++;
        }

        o2.omega.push_back(couple(A, B, poe));
      }

      assert(o1.gamma.size() == o1.omega.size() + 1);
      assert(o1.omega.size() == o2.omega.size());

      // (6) return result:

      return std::tie(std::as_const(o1_), std::as_const(o2_));
    }
  };
} // namespace U3xU4

#endif
