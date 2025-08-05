#ifndef SAM_U4XSU3_MANY_NUCLEON_BASIS_H
#define SAM_U4XSU3_MANY_NUCLEON_BASIS_H

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <ranges>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <U3basis.h>
#include <U3decomposition.h>
#include <U3irrep.h>

#include <U4basis.h>
#include <U4decomposition.h>
#include <U4irrep.h>

#include <UNtoU3.h>

#include <U4xSU3Irrep.h>

#include <qcx_convert.h>
#include <qcx_ND.h>
#include <qcx_tree_generator.h>

#ifdef HAVE_FMT
#include <fmt/format.h>
#endif

namespace U4xSU3
{
  class ManyNucleonBasis
  {
  public:
    ManyNucleonBasis(int A, int MMT, int Nmax, int JJ)
        : A_(A), MMT_(MMT), Nmax_(Nmax), JJ_(JJ), n0hw_(Nmax % 2), dim_(0)
    { }

    void construct()
    {
      dim_ = 0;

      shells_.clear();
      gamma_elements_.clear();
      gammas_.clear();
      omega_elements_u4_.clear();
      omegas_u4_.clear();
      omega_elements_su3_.clear();
      omegas_su3_.clear();
      irreps_.clear();

      generate_distributions();
      generate_gammas();
    }
    
    // (1) distributions:

    using DistributionElement = uint16_t;

  private:
    // the following definitions need to be here:

    auto n_distributions() const
    {
      assert(shells_.size() % D_ == 0);
      return shells_.size() / D_;
    }

    auto distribution(size_t dist_id) const
    {
      auto end_index = D_ * (dist_id + 1) - 1;
      assert(end_index < shells_.size());

      return std::span(shells_.data() + D_ * dist_id, D_);
    }

  public:

    // (2) gammas:

    class GammaElement : public U4xSU3::irrep
    {
      uint16_t alphamax_; // U(N)->U(3) reduction multiplicity

    public:
      GammaElement() : alphamax_(1) { }

      GammaElement(const u4::irrep& u4, int alphamax, const su3::irrep& su3)
          : U4xSU3::irrep(u4, su3), alphamax_(qcx_utils::itou16(alphamax))
      { }

      void alphamax(int am) { alphamax_ = qcx_utils::itou16(am); }
      int alphamax() const { return alphamax_; }
    };

  private:
    // the following definitions need to be here:

    auto n_gammas() const { return gammas_.size(); }

    auto gamma(size_t gamma_id) const
    {
      assert(gamma_id < gammas_.size());

      auto begin_index = gammas_[gamma_id].first;
      auto end_index = gammas_[gamma_id].second;

      assert(end_index <= gamma_elements_.size());

      return std::span(gamma_elements_.data() + begin_index,
                       gamma_elements_.data() + end_index);
    }

  public:

    // (3) omegas:

  private:
    // the following definitions need to be here:
    
    class OmegaElementU4
    {
      uint16_t etamax_; // U(4) coupling multiplicity
      u4::irrep u4_;

    public:
      OmegaElementU4() : etamax_(1) { }

      OmegaElementU4(int etamax, const u4::irrep& u4)
          : etamax_(qcx_utils::itou16(etamax)), u4_(u4)
      { }

      void etamax(int em) { etamax_ = qcx_utils::itou16(em); }
      int etamax() const { return etamax_; }

      void u4(const u4::irrep& u4) { u4_ = u4; }
      const u4::irrep& u4() const { return u4_; }

      // bool vacuum() const { return u4_.zero(); }
    };

    class OmegaElementSU3
    {
      uint16_t rhomax_; // SU(3) coupling multiplicity
      su3::irrep su3_;

    public:
      OmegaElementSU3() : rhomax_(1) { }

      OmegaElementSU3(int rhomax, const su3::irrep& su3)
          : rhomax_(qcx_utils::itou16(rhomax)), su3_(su3)
      { }

      void rhomax(int rm) { rhomax_ = qcx_utils::itou16(rm); }
      int rhomax() const { return rhomax_; }

      void su3(const su3::irrep& su3) { su3_ = su3; }
      const su3::irrep& su3() const { return su3_; }

      // bool vacuum() const { return su3_.zero(); }
    };

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

  public:
    class OmegaElement : public U4xSU3::irrep
    {
      uint16_t etamax_; // U(4) coupling multiplicity
      uint16_t rhomax_; // SU(3) coupling multiplicity

    public:
      OmegaElement(int etamax, const u4::irrep& u4, int rhomax,
                   const u3::irrep& su3)
          : U4xSU3::irrep(u4, su3),
            etamax_(qcx_utils::itou16(etamax)),
            rhomax_(qcx_utils::itou16(rhomax))
      { }

      OmegaElement(const OmegaElementU4& oe_u4, const OmegaElementSU3& oe_su3)
          : U4xSU3::irrep(oe_u4.u4(), oe_su3.su3()),
            etamax_(oe_u4.etamax()),
            rhomax_(oe_su3.rhomax())
      { }

      OmegaElement(const GammaElement& ge)
          : U4xSU3::irrep(ge), etamax_(1), rhomax_(1)
      { }

      void etamax(int em) { etamax_ = qcx_utils::itou16(em); }
      void rhomax(int rm) { rhomax_ = qcx_utils::itou16(rm); }

      int etamax() const { return etamax_; }
      int rhomax() const { return rhomax_; }
    };

  private:
    // the following definition need to be here:

    // class for combining U(4) and SU(3) omega parts:
    class OmegaCombiner
    {
      // storage for resulting combined omega elements:
      std::vector<OmegaElement> omega_;

    public:
      auto combine(std::span<const OmegaElementU4> omega_u4,
                   std::span<const OmegaElementSU3> omega_su3)
      {
        assert(omega_u4.size() == omega_su3.size());

        omega_.clear();

        for (size_t i = 0; i < omega_u4.size(); i++)
          omega_.emplace_back(omega_u4[i], omega_su3[i]);

        return std::span(std::as_const(omega_));
      }
    };

  public:

    // (4) irreps:

    auto n_irreps() const { return irreps_.size(); }

    struct Irrep
    {
      std::span<const DistributionElement> dist;
      std::span<const GammaElement> gamma;
      std::span<const OmegaElement> omega;
      uint64_t offset;
    };

    class IrrepDB
    {
      const ManyNucleonBasis& basis_;
      OmegaCombiner oc_;

    public:
      IrrepDB(const ManyNucleonBasis& basis) : basis_(basis) { }

      auto size() const { return basis_.n_irreps(); }

      Irrep operator[](size_t irrep_id) 
      {
        assert(irrep_id < basis_.irreps_.size());

        auto [dist_id, gamma_id, omega_u4_id, omega_su3_id, offset] =
            basis_.irreps_[irrep_id];

        auto dist = basis_.distribution(dist_id);
        auto gamma = basis_.gamma(gamma_id);

        assert(gamma.size() > 0);

        if (gamma.size() == 1)
        {
          assert(omega_u4_id == INVALID_OMEGA_ID);
          assert(omega_su3_id == INVALID_OMEGA_ID);

          return {dist, gamma, std::span<const OmegaElement>(), offset};
        }
        else
        {
          auto omega_u4 = basis_.omega_u4(omega_u4_id);
          auto omega_su3 = basis_.omega_su3(omega_su3_id);

          assert(omega_u4.size() == gamma.size() - 1);
          assert(omega_su3.size() == gamma.size() - 1);

          auto omega = oc_.combine(omega_u4, omega_su3);

          return {dist, gamma, omega, offset};
        }
      }

   // friend class IrrepDB;
    };

    auto irreps() const { return IrrepDB(*this); }

    // (5) others:

    auto dim() const { return dim_; }

  private:
    int A_;   // number of nucleons; equal to (Z+N)
    int MMT_; // proton-neutron "difference"; equal to (abs(Z-N))
    int Nmax_;
    int JJ_;

    int n0hw_; // lowest considered HO shell; 0 for even Nmax, 1 for odd Nmax

    size_t dim_;

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

    struct InternalIrrep
    {
      uint32_t dist_id;
      uint32_t gamma_id;
      uint32_t omega_u4_id;
      uint32_t omega_su3_id;
      uint64_t offset;
    };

    std::vector<InternalIrrep> irreps_;

    // (1) distributions:

    void generate_distributions()
    {
      // configuration of a single shell:
      using ShellConfig = std::vector<DistributionElement>;

      // shell configurations for all subspaces:
      std::vector<std::set<ShellConfig>> subspaces_configs(Nmax_ + 1);

      // (1) generate the (unique) shell configuration for 0hw:

      ShellConfig v0hw_config;

      assert(A_ > 0);

      int nParticlesLeft = A_;
      int n = 0; // shell index

      while (nParticlesLeft > 0)
      {
        // number of nucleons that can fit into nth HO shell:
        int N = 2 * (n + 1) * (n + 2);

        auto place = std::min(nParticlesLeft, N);
        assert(place > 0);

        v0hw_config.push_back(place);
        nParticlesLeft -= place;
        n++;
      }

      n--; // n: last occuppied shell for 0hw subspace
      assert(n >= 0);

      n0max_ = n;
      nmax_ = n0max_ + Nmax_;
      D_ = nmax_ + 1;

      v0hw_config.resize(D_);

      // there is only one 0hw shell configuration:
      subspaces_configs[0].insert(std::move(v0hw_config));

      // (2) generate shell configurations for subspaces 1hw, 2hw, ..., Nmaxhw:

      for (size_t Nsubspace = 1; Nsubspace <= Nmax_; Nsubspace++)
      {
        auto& subspace_configs = subspaces_configs[Nsubspace];
        const auto& seed_subspace = subspaces_configs[Nsubspace - 1];

        for (const auto& seed_config : seed_subspace)
        {
          // try to raise 1 nucleon for each occupied shell:
          for (size_t q = 0; q < nmax_; q++)
          {
            auto new_config(seed_config);

            // number of nucleons that can fit into next HO shell:
            auto N = 2 * (q + 2) * (q + 3);

            // if it is possible to raise a nucleon:
            if (new_config[q] && new_config[q + 1] + 1 <= N)
            {
              // raise a nucleon:
              new_config[q] -= 1;
              new_config[q + 1] += 1;

              // add to subspace configurations:
              subspace_configs.insert(std::move(new_config));
            }
          }
        }
      }

      // (3) add the generated Nmax-relevant configurations to the basis:

      for (size_t Nsubspace = n0hw_; Nsubspace <= Nmax_; Nsubspace += 2)
        for (const auto& subspace_config : subspaces_configs[Nsubspace])
          std::ranges::copy(subspace_config, std::back_inserter(shells_));
    }

    // (2) gammas:
    
    void generate_gammas()
    {
      for (size_t dist_id = 0; dist_id < n_distributions(); dist_id++)
        generate_gammas_for_distribution(dist_id);
    }

    void generate_gammas_for_distribution(size_t dist_id)
    {
      assert(dist_id < n_distributions());

      const auto dist = distribution(dist_id);

      // (1) generate gamma elements for all non-vacant shells:

      // gamma elements for non-vacant shells:
      std::vector<std::vector<GammaElement>> shells_elements;

      for (int n = 0; n < dist.size(); n++)
      {
        int A = dist[n];

        if (A == 0)
          continue;

        std::vector<GammaElement> shell_elements;
        generate_single_shell_gamma_elements(n, A, shell_elements);
        shells_elements.push_back(std::move(shell_elements));
      }

      // (2) gammas = Cartesian product of single-shell gamma elements:

      qcx_utils::NDSpan<GammaElement> nds;

      for (const auto& shell_elements : shells_elements)
        nds.add_dimension(shell_elements);

      auto n_combinations = nds.cardinality();
      auto n_active_shells = shells_elements.size();

      assert(n_active_shells > 0);

      std::vector<GammaElement> gamma(n_active_shells);

      // add gammas to basis:
      for (unsigned index = 0; index < n_combinations; index++)
      {
        nds.extract_multi_element(index, gamma);

        // add gamma to basis:

        auto gamma_id = gammas_.size();

        auto begin = gamma_elements_.size();
        auto end = begin + gamma.size();
        gammas_.emplace_back(begin, end);

        std::ranges::copy(gamma, std::back_inserter(gamma_elements_));

        // a single gamma element case:
        if (n_active_shells == 1)
        {
          const auto& ge = gamma[0];

          auto dim_gamma = ge.alphamax() * compute_dim(ge.u4(), ge.su3());

          if (dim_gamma > 0)
          {
            // add irrep to basis:
            irreps_.emplace_back(dist_id, gamma_id, INVALID_OMEGA_ID,
                                 INVALID_OMEGA_ID, dim_);

            dim_ += dim_gamma;
          }
        }
        else
        {
          // generate omegas for gamma:
          generate_omegas_for_gamma(dist_id, gamma_id);
        }
      }
    }

    void generate_single_shell_gamma_elements(
        int n, int A, std::vector<GammaElement>& gamma_elements)
    {
      std::vector<u4::irrep> u4_irreps;
      generate_U4_irreps(n, A, u4_irreps);

      untou3::UNtoU3<4> gen;
      gen.set_shell(n);
      int N = (n + 1) * (n + 2) / 2;

      gamma_elements.clear();

      for (auto u4 : u4_irreps)
      {
        gen.reduceU4(u4, N);
        auto res = gen.irreps();

        for (const auto& [u3, alphamax] : res)
          gamma_elements.emplace_back(u4, alphamax, u3);
      }
    }

    void generate_U4_irreps(int n, int A, std::vector<u4::irrep>& u4_irreps)
    {
      int nsps = (n + 1) * (n + 2) / 2;

      // if (A > 4 * nsps) return;
      assert(A <= 4 * nsps);

      for (int n14 = std::min(A, nsps); n14 >= 0; --n14)
      {
        for (int n24 = n14; n24 >= 0; --n24)
        {
          if (n14 + n24 > A)
            continue;

          for (int n34 = n24; n34 >= 0; --n34)
          {
            if (n14 + n24 + n34 > A)
              continue;

            int n44 = A - (n14 + n24 + n34);

            if (n44 <= n34 && n44 >= 0)
              u4_irreps.emplace_back(n14, n24, n34, n44);
          }
        }
      }
    }

    int compute_dim(const u4::irrep& u4, const su3::irrep& su3)
    {
      int dim = 0;

      auto u4_basis = u4::physicalBasis(u4);

      for (auto const& labels : u4_basis)
      {
        if (labels.TT >= MMT_)
        {
          int zeta = labels.zeta;
          int SS = labels.SS;

          dim += zeta * compute_dim(su3, SS);
        }
      }

      return dim;
    }

    int compute_dim(const su3::irrep& su3, int SS)
    {
      int dim = 0;

      for (int L = 0; L <= su3.lambda() + su3.mu(); L++)
      {
        int k0max = su3::kmax(su3, L);

        if (k0max)
        {
          for (size_t jj = std::abs(2 * L - SS); jj <= (2 * L + SS); jj += 2)
            if ((jj == JJ_) || (JJ_ == -1))
              dim += k0max;
        }
      }

      return dim;
    }

    // (3) omegas:

    void generate_omegas_for_gamma(size_t dist_id, size_t gamma_id)
    {
      auto gamma = this->gamma(gamma_id);
      auto gamma_size = gamma.size();
      assert(gamma_size > 1);
      auto omega_size = gamma_size - 1;

      // (1) gamma2omega for U(4):

      std::vector<std::vector<OmegaElementU4>> omegas_u4;

      auto couple_u4 = [](const auto& first, const auto& second, auto& stack)
      {
        auto [u4s, mults] = u4::decomposition(first.u4(), second.u4());

        for (size_t i = 0; i < u4s.size(); i++)
          stack.emplace(mults[i], u4s[i]);
      };

      qcx_utils::tree_generator<OmegaElementU4> te_u4;

      te_u4.generate(gamma, couple_u4, omegas_u4);

      // add generated omegas into basis:

      auto first_omega_u4_id = omegas_u4_.size();

      for (const auto& omega_u4 : omegas_u4)
      {
        auto begin = omega_elements_u4_.size();
        auto end = begin + omega_size;

        omegas_u4_.emplace_back(begin, end);

        std::ranges::copy(omega_u4, std::back_inserter(omega_elements_u4_));
      }

      // (2) gamma2omega for SU(3):

      std::vector<std::vector<OmegaElementSU3>> omegas_su3;

      auto couple_su3 = [](const auto& first, const auto& second, auto& stack)
      {
        auto [su3s, mults] = su3::decomposition(first.su3(), second.su3());

        for (size_t i = 0; i < su3s.size(); i++)
          stack.emplace(mults[i], su3s[i]);
      };

      qcx_utils::tree_generator<OmegaElementSU3> te_su3;

      te_su3.generate(gamma, couple_su3, omegas_su3);

      // add generated omegas into basis:

      auto first_omega_su3_id = omegas_su3_.size();

      for (const auto& omega_su3 : omegas_su3)
      {
        auto begin = omega_elements_su3_.size();
        auto end = begin + omega_size;

        omegas_su3_.emplace_back(begin, end);

        std::ranges::copy(omega_su3, std::back_inserter(omega_elements_su3_));
      }

      // (3) add irreps:

      // total gamma multiplicity:
      uint32_t mult_gamma = 1;
      for (const auto& ge : gamma)
        mult_gamma *= ge.alphamax();

      for (size_t i = 0; i < omegas_u4.size(); i++)
      {
        const auto& omega_u4 = omegas_u4[i];
        const auto& last_oe_u4 = omega_u4.back();
        auto omega_id_u4 = first_omega_u4_id + i;

        uint32_t mult_u4 = 1;
        for (const auto& oe_u4 : omega_u4)
          mult_u4 *= oe_u4.etamax();

        for (size_t j = 0; j < omegas_su3.size(); j++)
        {
          const auto& omega_su3 = omegas_su3[j];
          const auto& last_oe_su3 = omega_su3.back();
          auto omega_id_su3 = first_omega_su3_id + j;

          uint32_t mult_su3 = 1;
          for (const auto& oe_su3 : omega_su3)
            mult_su3 *= oe_su3.rhomax();

          auto mult = mult_gamma * mult_u4 * mult_su3;
          auto dim_irrep = compute_dim(last_oe_u4.u4(), last_oe_su3.su3());
          auto dim_omega = mult * dim_irrep;

          if (dim_omega > 0)
          {
            // add irrep to basis:
            irreps_.emplace_back(dist_id, gamma_id, omega_id_u4, omega_id_su3,
                                 dim_);

            dim_ += dim_omega;
          }
        }
      }
    }
  };

  /*
  // structure for irrep used outside of the basis class:
  struct Irrep
  {
    std::vector<ManyNucleonBasis::DistributionElement> dist;
    std::vector<ManyNucleonBasis::GammaElement> gamma;
    std::vector<ManyNucleonBasis::OmegaElementU4> omega_u4;
    std::vector<ManyNucleonBasis::OmegaElementSU3> omega_su3;

    void clear()
    {
      dist.clear();
      gamma.clear();
      omega_u4.clear();
      omega_su3.clear();
    }
  };

  // class for unifying 2 irreps wrt to their active shells:
  class IrrepUnifier
  {
    Irrep o1_, o2_;
    OmegaCombiner oc1_, oc2_;
    std::vector<ManyNucleonBasis::OmegaElement> oomega1_, oomega2_;

  public:
    auto combine(const auto& i1, const auto& i2)
    {
      // (1) setup:

      const auto& idist1 = i1.dist;
      const auto& igamma1 = i1.gamma;
      auto iomega1 = oc1_.combine(i1.omega_u4(), i1.omega_su3());

      const auto& idist2 = i2.dist;
      const auto& igamma2 = i2.gamma;
      auto iomega2 = oc2_.combine(i2.omega_u4(), i2.omega_su3());

      o1_.clear();
      oomega1_.clear();
      auto& odist1 = o1_.dist;
      auto& ogamma1 = o1_.gamma;
      auto& oomega1 = oomega1_;

      o2_.clear();
      oomega2_.clear();
      auto& odist2 = o2_.dist;
      auto& ogamma2 = o2_.gamma;
      auto& oomega2 = oomega2_;

      size_t g1 = 0; // igamma1 element index
      size_t g2 = 0; // igamma2 element index

      static const u4::irrep u4_vacuum(0, 0, 0, 0);
      static const su3::irrep su3_vacuum(0, 0);
      static const ManyNucleonBasis::GammaElement ge_vacuum(u4_vacuum, 1,
                                                            su3_vacuum);
      static const ManyNucleonBasis::OmegaElement oe_vacuum({1, u4_vacuum},
                                                            {1, su3_vacuum});

      // (2) unify distribution sizes:

      std::copy(idist1.begin(), idist1.end(), std::back_inserter(odist1));
      std::copy(idist2.begin(), idist2.end(), std::back_inserter(odist2));

      auto D = std::max(odist1.size(), odist2.size());

      assert(D > 0);

      odist1.resize(D);
      odist2.resize(D);

      // (3) unify gammas and omegas:

      for (size_t d = 0; d < D; d++)
      {
        // nothing to do if both shells are empty:
        if ((odist1[d] == 0) && (odist2[d] == 0))
          continue;

        // add element to gamma1:
        ogamma1.push_back(odist1[d] == 0 ? ge_vacuum : igamma1[g1++]);

        // add element to omega1:
        if (ogamma1.size() > 1)
        {
          if (oomega1.empty())
          {
            // couple last 2 gamma elements:
            if (ogamma1[0].vacuum() && ogamma1[1].vacuum())
              oomega1.push_back(oe_vacuum);
            else if (ogamma1[0].vacuum())
              oomega1.push_back(ogamma1[1]);
            else if (ogamma1[1].vacuum())
              oomega1.push_back(ogamma1[0]);
            else
              oomega1.push_back(iomega1[0]);
          }
          else
          {
            // couple last gamma element with last omega element:
            if (ogamma1.back().vacuum() && oomega1.back().vacuum())
              oomega1.push_back(oe_vacuum);
            else if (ogamma1.back().vacuum())
              oomega1.push_back(oomega1.back());
            else if (oomega1.back().vacuum())
              oomega1.push_back(ogamma1.back());
            else
              oomega1.push_back(iomega1[g1 - 2]);
          }
        }

        // add element to gamma2:
      }

      // (4) split resulting omegas to U(4) and SU(3) parts:

      // (5) return result:

      return std::tie(std::as_const(o1_), std::as_const(o2_));
    }
  };
  */
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
                     oe.u4().n14(), oe.u4().n24(), oe.u4().n34(), oe.u4().n44(),
                     oe.rhomax(), oe.su3().lambda(), oe.su3().mu());
  }
};

#endif

#endif
