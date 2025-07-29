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
#include <vector>

#include <U3basis.h>
#include <U3decomposition.h>
#include <U3irrep.h>

#include <U4basis.h>
#include <U4decomposition.h>
#include <U4irrep.h>

#include <UNtoU3.h>

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

    // (2) gammas:

    struct GammaElement
    {
      u4::irrep u4;
      uint16_t alphamax; // U(N)->U(3) reduction multiplicity
      su3::irrep su3;
    };

    // auto n_gammas() const { return gammas_.size(); }

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

    struct OmegaElementU4
    {
      uint16_t etamax; // U(4) coupling multiplicity
      u4::irrep u4;
    };

    struct OmegaElementSU3
    {
      uint16_t rhomax; // SU(3) coupling multiplicity
      su3::irrep su3;
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

    struct OmegaElement
    {
      OmegaElementU4 mu4;
      OmegaElementSU3 msu3;
    };

    static void combine_omega_u4su3(std::span<const OmegaElementU4> omega_u4,
                                    std::span<const OmegaElementSU3> omega_su3,
                                    std::vector<OmegaElement>& result)
    {
      assert(omega_u4.size() == omega_su3.size());

      result.clear();

      for (size_t i = 0; i < omega_u4.size(); i++)
        result.emplace_back(omega_u4[i], omega_su3[i]);
    }

    // (4) irreps:

    struct Irrep
    {
      uint32_t dist_id; 
      uint32_t gamma_id;
      uint32_t omega_u4_id;
      uint32_t omega_su3_id;
      uint32_t offset;
    };

    inline static const uint32_t INVALID_OMEGA_ID =
        std::numeric_limits<uint32_t>::max();

    auto n_irreps() const { return irreps_.size(); }

    /*
    const auto& irrep(size_t irrep_id) const
    {
      assert(irrep_id < irreps_.size());
      return irreps_[irrep_id];
    }
    */

    auto irrep(size_t irrep_id) const
    {
      assert(irrep_id < irreps_.size());

      auto [dist_id, gamma_id, omega_u4_id, omega_su3_id, offset] =
          irreps_[irrep_id];

      auto dist = distribution(dist_id);
      auto gamma = this->gamma(gamma_id);

      assert(gamma.size() > 0);

      if (gamma.size() == 1)
      {
        assert(omega_u4_id == INVALID_OMEGA_ID);
        assert(omega_su3_id == INVALID_OMEGA_ID);

        return std::tuple(dist, gamma, std::span<const OmegaElementU4>(),
                          std::span<const OmegaElementSU3>(), offset);
      }
      else
      {
        auto omega_u4 = this->omega_u4(omega_u4_id);
        auto omega_su3 = this->omega_su3(omega_su3_id);

        assert(omega_u4.size() == gamma.size() - 1);
        assert(omega_su3.size() == gamma.size() - 1);

        return std::tuple(dist, gamma, omega_u4, omega_su3, offset);
      }
    }

    // (5) others:

    auto dim() const { return dim_; }

  private:
    int A_; // number of nucleons; equal to (Z+N)
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

    std::vector<uint16_t> shells_; // occupancy of shells for distributions

    // (2) gammas:

    std::vector<GammaElement> gamma_elements_;
    std::vector<std::pair<size_t, size_t>> gammas_;

    // (3) omegas:

    std::vector<OmegaElementU4> omega_elements_u4_;
    std::vector<std::pair<size_t, size_t>> omegas_u4_;

    std::vector<OmegaElementSU3> omega_elements_su3_;
    std::vector<std::pair<size_t, size_t>> omegas_su3_;

    // (4) irreps:

    std::vector<Irrep> irreps_;

    // (1) distributions:

    void generate_distributions()
    {
      // configuration of a single shell:
      using ShellConfig = std::vector<uint16_t>;

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

          auto dim_gamma = ge.alphamax * compute_dim(ge.u4, ge.su3);

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
        auto [u4s, mults] = u4::decomposition(first.u4, second.u4);

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
        auto [su3s, mults] = su3::decomposition(first.su3, second.su3);

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
        mult_gamma *= ge.alphamax;

      for (size_t i = 0; i < omegas_u4.size(); i++)
      {
        const auto& omega_u4 = omegas_u4[i];
        const auto& last_oe_u4 = omega_u4.back();
        auto omega_id_u4 = first_omega_u4_id + i;

        uint32_t mult_u4 = 1;
        for (const auto& oe_u4 : omega_u4)
          mult_u4 *= oe_u4.etamax;

        for (size_t j = 0; j < omegas_su3.size(); j++)
        {
          const auto& omega_su3 = omegas_su3[j];
          const auto& last_oe_su3 = omega_su3.back();
          auto omega_id_su3 = first_omega_su3_id + j;

          uint32_t mult_su3 = 1;
          for (const auto& oe_su3 : omega_su3)
            mult_su3 *= oe_su3.rhomax;

          auto mult = mult_gamma * mult_u4 * mult_su3;
          auto dim_irrep = compute_dim(last_oe_u4.u4, last_oe_su3.su3);
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
    return format_to(ctx.out(), "[{},{},{},{}]{}({},{})", ge.u4.n14(),
                     ge.u4.n24(), ge.u4.n34(), ge.u4.n44(), ge.alphamax,
                     ge.su3.lambda(), ge.su3.mu());
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
    return format_to(ctx.out(), "{}[{},{},{},{}]", oe.etamax, oe.u4.n14(),
                     oe.u4.n24(), oe.u4.n34(), oe.u4.n44());
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
    return format_to(ctx.out(), "{}({},{})", oe.rhomax, oe.su3.lambda(),
                     oe.su3.mu());
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
    return format_to(ctx.out(), "{}[{},{},{},{}]{}({},{})", oe.mu4.etamax,
                     oe.mu4.u4.n14(), oe.mu4.u4.n24(), oe.mu4.u4.n34(),
                     oe.mu4.u4.n44(), oe.msu3.rhomax, oe.msu3.su3.lambda(),
                     oe.msu3.su3.mu());
  }
};

#endif

#endif
