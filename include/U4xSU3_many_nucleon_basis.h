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

      generate_distributions();
      generate_gammas();
   // generate_omegas();
    }

    // (1) distributions:

    auto n_distributions() const
    {
      assert(shells_.size() % D_ == 0);
      return shells_.size() / D_;
    }

    auto distribution(size_t dist_id) const
    {
      return std::span(shells_.data() + D_ * dist_id, D_);
    }

    // (2) gammas:

    struct GammaElement
    {
      u4::irrep u4;
      uint16_t alpha; // U(N)->U(3) reduction multiplicity
      su3::irrep su3;
    };

    auto n_gammas() const { return gammas_.size(); }

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

    // (4) irreps:

    struct Irrep
    {
      uint32_t dist_id; 
      uint32_t gamma_id;
      uint32_t omega_id;
      uint32_t offset;
    };

    inline static const uint32_t INVALID_OMEGA_ID =
        std::numeric_limits<uint32_t>::max();

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
    std::vector<OmegaElementSU3> omega_elements_su3_;
    std::vector<std::pair<size_t, size_t>> omegas_;

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

      shells_.clear();

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

          auto dim = ge.alpha * compute_dim(ge.u4, ge.su3);

          if (dim > 0)
          {
            // add irrep to basis:
            irreps_.emplace_back(dist_id, gamma_id, INVALID_OMEGA_ID,
                                 INVALID_OMEGA_ID);

            dim_ += dim;
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

        for (const auto& [u3, alpha] : res)
          gamma_elements.emplace_back(u4, alpha, u3);
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
/*
      std::vector<MU4_VEC> omega_mu4;
      gamma2omega(gamma_u4, omega_mu4);

      std::vector<MSU3_VEC> omega_msu3;
      gamma2omega(gamma_su3, omega_msu3);

      uint32_t mult_gamma_u4 = 1;
      for (size_t i = 0; i < gamma_u4.size(); ++i)
        mult_gamma_u4 *= gamma_u4[i].mult;
      assert(mult_gamma_u4 == 1);

      uint32_t mult_gamma_su3 = 1;
      for (size_t i = 0; i < gamma_su3.size(); ++i)
        mult_gamma_su3 *= gamma_su3[i].mult;

      uint32_t mult_gamma = mult_gamma_su3 * mult_gamma_u4;

      assert(gamma_id < gamma_omegas_.size());

      auto& [omega_id_begin, n_omegas] = gamma_omegas_[gamma_id];
      omega_id_begin = omegas_.size();
      n_omegas = 0;

      for (const auto& mu4_vec : omega_mu4)
      {
        uint32_t mult_u4 = 1;
        for (size_t i = 0; i < mu4_vec.size(); ++i)
          mult_u4 *= mu4_vec[i].mult;

        for (const auto& msu3_vec : omega_msu3)
        {
          assert(mu4_vec.size() == msu3_vec.size());

          uint32_t mult_su3 = 1;
          for (size_t i = 0; i < msu3_vec.size(); ++i)
            mult_su3 *= msu3_vec[i].mult;
          uint32_t mult = mult_gamma * mult_u4 * mult_su3;

          MU4 mu4 = mu4_vec.back();
          MSU3 msu3 = msu3_vec.back();
       // u4::irrep u4 = mu4.u4;
       // su3::irerp su3 = msu3.su3;
       // int lm = msu3.lm;
       // int mu = msu3.mu;
          uint32_t dim_irrep = computeDim(msu3.su3, mu4.u4);

          auto dim_omega = mult * dim_irrep;
          dim += dim_omega;

          if (dim_omega > 0)
          {
            irreps_.emplace_back(nhw, dist_id, gamma_id, omegas_.size(), dim_);

            omegas_.emplace_back(omegas_u4_.size(), mu4_vec.size(), mult,
                                 mult * dim_irrep);

            for (std::size_t i = 0; i < mu4_vec.size(); i++)
            {
              omegas_u4_.push_back(mu4_vec[i]);
              omegas_su3_.push_back(msu3_vec[i]);
            }

            n_omegas++;

            dim_ += dim_omega;
          }
        }
      }

      // dim_ += dim;
*/
    }



#if 0
    /////////////   DISTRIBUTIONS  /////////////////

    ////////////////////////// U4 /////////////////////////////
    struct MU4
    {
      uint16_t mult;
      u4::irrep u4;
    };

    using MU4_VEC = std::vector<MU4>;

    ////////////////////////// SU3 /////////////////////////////
    struct MSU3
    {
      uint16_t mult;
      su3::irrep su3;
    };

    using MSU3_VEC = std::vector<MSU3>;

  private:

    // distributions:
    std::vector<tSHELLSCONFIG> distributions_;
    // indices of distributions for `nhw` are from [nhw] to [nhw+1]-1:
    std::vector<std::size_t> nhw_dist_ids_;

    // for each distribution: index of first gamma, #of gammas:
    std::vector<std::pair<std::size_t, std::size_t>> dist_gammas_;
    // for each gamma: index of first irrep, #of irreps
    std::vector<std::pair<std::size_t, std::size_t>> gammas_;
    // gammas irreps:
    std::vector<MU4> gammas_u4_;
    std::vector<MSU3> gammas_su3_;

    // for each gamma: index of first omega, #of omegas:
    std::vector<std::pair<std::size_t, std::size_t>> gamma_omegas_;
    // for each omega: index of first irrep, #of irreps, multiplicity,
    // dimension:
    std::vector<std::tuple<std::size_t, std::size_t, int, int>> omegas_;
    // omegas irreps:
    std::vector<MU4> omegas_u4_;
    std::vector<MSU3> omegas_su3_;

    // for each irrep: nhw, dist_id, gamma_id, omega_id, offset:
    std::vector<
        std::tuple<int, std::size_t, std::size_t, std::size_t, std::size_t>>
        irreps_;

  public:

    void construct()
    {
      dim_ = 0;

      generate_distributions();
      generate_gammas();
      generate_omegas();
    }

    // returns an index extent for distributions for particular `nhw`:
    auto distributions_ids(int nhw) const
    {
      assert(nhw >= 0);
      assert(nhw + 1 < nhw_dist_ids_.size());

      auto begin = nhw_dist_ids_[nhw];
      auto end = nhw_dist_ids_[nhw + 1];

      return std::pair(begin, end);
    }

    // return distribution by index:
    const auto& distribution(std::size_t dist_id)
    {
      assert(dist_id < distributions_.size());

      return distributions_[dist_id];
    }

    // returns a span of distributions for particular `nhw`:
    auto distributions(int nhw) const
    {
      auto [begin, end] = distributions_ids(nhw);

      return std::span(distributions_.data() + begin,
                       distributions_.data() + end);
    }

    // returns (first gamma id, #of gammas) for distribution:
    const auto& dist_gammas(std::size_t dist_id) const
    {
      assert(dist_id < dist_gammas_.size());

      return dist_gammas_[dist_id];
    }

    // returns irrpes for gamma:
    auto gamma_irreps(std::size_t gamma_id) const
    {
      assert(gamma_id < gammas_.size());

      auto [first, count] = gammas_[gamma_id];

      assert(first + count <= gammas_u4_.size());
      assert(first + count <= gammas_su3_.size());

      return std::pair(std::span(gammas_u4_.data() + first, count),
                       std::span(gammas_su3_.data() + first, count));
    }

    // returns (first omega id, #of omegas) for gamma:
    const auto& gamma_omegas(std::size_t gamma_id) const
    {
      assert(gamma_id < gamma_omegas_.size());

      return gamma_omegas_[gamma_id];
    }

    // returns irreps for omega plus their multiplicity and dimension:
    auto omega_irreps(std::size_t omega_id) const
    {
      assert(omega_id < omegas_.size());

      auto [first, count, mult, dim] = omegas_[omega_id];

      assert(first + count <= omegas_u4_.size());
      assert(first + count <= omegas_su3_.size());

      return std::tuple(std::span(omegas_u4_.data() + first, count),
                        std::span(omegas_su3_.data() + first, count), mult,
                        dim);
    }

    auto n_irreps() const { return irreps_.size(); }

    const auto& irrep(size_t irrep_id) const
    {
      assert(irrep_id < irreps_.size());

      return irreps_[irrep_id];
    }

    auto dim() const { return dim_; }

  private:

    ////////////////////////// U4 /////////////////////////////
    void Couple(const MU4& a, const MU4& b, MU4_VEC& result)
    {
      auto [u4s, mults] = u4::decomposition(a.u4, b.u4);
      for (int i = 0; i < u4s.size(); i++)
        result.emplace_back(mults[i], u4s[i]);
    }

    void generate_gammas()
    {
      dist_gammas_.clear();
      gammas_.clear();
      gammas_u4_.clear();
      gammas_su3_.clear();

      dist_gammas_.resize(distributions_.size());

      for (int nhw = n0hw_; nhw <= Nmax_; nhw += 2)
      {
        auto [dist_id_begin, dist_id_end] = distributions_ids(nhw);

        for (std::size_t dist_id = dist_id_begin; dist_id < dist_id_end;
             dist_id++)
          generate_gammas_for_dist(dist_id);
      }
    }


    void generate_omegas()
    {
      gamma_omegas_.clear();
      omegas_.clear();
      omegas_u4_.clear();
      omegas_su3_.clear();

      gamma_omegas_.resize(gammas_.size());

      for (int nhw = n0hw_; nhw <= Nmax_; nhw += 2)
      {
        auto [dist_id_begin, dist_id_end] = distributions_ids(nhw);

        for (std::size_t dist_id = dist_id_begin; dist_id < dist_id_end;
             dist_id++)
        {
          auto [gamma_id_begin, n_gammas] = dist_gammas_[dist_id];

          for (std::size_t i_gamma = 0; i_gamma < n_gammas; i_gamma++)
          {
            auto gamma_id = gamma_id_begin + i_gamma;
            generate_omegas_for_gamma(nhw, dist_id, gamma_id);
          }
        }
      }
    }

    void generate_omegas_for_gamma(int nhw, std::size_t dist_id,
                                   std::size_t gamma_id)
    {
      assert(gamma_id < gammas_.size());

      auto [begin, count] = gammas_[gamma_id];

      auto gamma_u4 = std::span(gammas_u4_.data() + begin, count);
      auto gamma_su3 = std::span(gammas_su3_.data() + begin, count);

      assert(gamma_u4.size() == gamma_su3.size());
      assert(gamma_u4.size() > 0);

      if (gamma_u4.size() == 1)
      {
        const auto& mu4 = gamma_u4[0];
        const auto& msu3 = gamma_su3[0];

        assert(mu4.mult == 1);

        uint32_t mult = msu3.mult;
     // auto dim = mult * computeDim(msu3.lm, msu3.mu, mu4.u4);
        auto dim = mult * computeDim(msu3.su3, mu4.u4);

        irreps_.emplace_back(nhw, dist_id, gamma_id, gamma_id, dim_);

        dim_ += dim;
      }
      else
        generate_omegas_for_gamma(nhw, dist_id, gamma_id, gamma_u4, gamma_su3);
    }


  private:
    int kmax(int lm, int mu, int L)
    {
      assert((lm + mu - L) >= 0);

      return (lm + mu - L) / 2 - std::max(0, (lm + 1 - L) / 2) -
             std::max(0, (mu + 1 - L) / 2) + 1;
    }



 // void couple(int lm1, int mu1, int lm2, int mu2, MSU3_VEC& ir3)
    void Couple(const MSU3& a, const MSU3& b, MSU3_VEC& result)
    {
      auto lm1 = a.su3.lambda();
      auto mu1 = a.su3.mu();

      auto lm2 = b.su3.lambda();
      auto mu2 = b.su3.mu();

      result.reserve(3 * (lm1 + mu1 + lm2 + mu2));

      int ix_max = std::min(mu1, lm2);
      for (int ix = 0; ix <= ix_max; ++ix)
      {
        int iy_max = std::min(lm1, lm2 + mu2 - ix);
        for (int iy = 0; iy <= iy_max; ++iy)
        {
          int iz_min = std::max(0, mu2 - mu1 + ix - iy);
          int iz_max = std::min(mu2, lm2 + mu2 - ix - iy);
          for (int iz = iz_min; iz <= iz_max; ++iz)
          {
            int lm3 = lm1 + lm2 + mu2 - 2 * iy - ix - iz;
            int mu3 = mu1 - mu2 - ix + iy + 2 * iz;
            int ip = (2 * (mu3 - mu1) + lm3 - lm1 - lm2 + mu2) / 3;
            int in = (2 * (lm1 + lm2 - lm3) + mu1 + mu2 - mu3) / 3;
            int irho = ix - std::max({0, -ip, in - lm1}) + 1;
            if (irho != 1)
              continue;
            result.push_back(MSU3(su3::Rho(lm1, mu1, lm2, mu2, lm3, mu3),
                                  su3::irrep(lm3, mu3)));
          }
        }
      }
    }

    template <typename T, std::size_t Extent>
    void gamma2omega(std::span<T, Extent> gamma,
                     std::vector<std::vector<T>>& omega)
    {
      assert(gamma.size() >= 2);

      size_t nColumns = gamma.size() - 1;
      std::vector<T> Row(nColumns);
      std::vector<std::vector<T>> vStack(nColumns);
      Couple(gamma[0], gamma[1], vStack[0]);

      for (int icol = 0;;)
      {
        if (icol == nColumns - 1)
        {
          for (size_t i = 0; i < vStack[icol].size(); ++i)
          {
            Row[icol] = vStack[icol][i];
            omega.push_back(Row);
          }
          vStack[icol].resize(0);
          if (icol > 0)
            icol--;
        }

        if (!vStack[icol].empty())
        {
          Row[icol] = vStack[icol].back();
          vStack[icol].pop_back();
          Couple(Row[icol], gamma[icol + 2], vStack[icol + 1]);
          icol++;
        }
        else
        {
          if (icol > 0)
            icol--;
          else // at this point icol == 0 and vStack[icol == 0] is empty
            break;
        }
      }
    }

#endif
  };
} // namespace U4xSU3
