#include <array>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <set>
#include <span>
#include <tuple>
#include <vector>

#include <U4basis.h>
#include <U4decomposition.h>
#include <UNtoU3.h>
#include <UNxU4SlaterDeterminant.h>

namespace U4xSU3
{
  class ManyNucleonBasis
  {
  public:
    /////////////   DISTRIBUTIONS  /////////////////
    typedef std::vector<uint32_t> tSHELLSCONFIG;

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
      int lm;
      int mu;
    };

    using MSU3_VEC = std::vector<MSU3>;

  private:
    int A_;
    int MMT_;
    int Nmax_;
    int JJ_;
    int n0hw_;

    std::size_t dim_;

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

  public:
    ManyNucleonBasis(int A, int MMT, int Nmax, int JJ)
        : A_(A), MMT_(MMT), Nmax_(Nmax), JJ_(JJ), n0hw_(Nmax % 2), dim_(0)
    { }

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

      return std::span(distributions_.data() + begin, distributions_.data() + end);
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

    auto dim() const { return dim_; }

  private:
    /////////////   DISTRIBUTIONS  /////////////////
    void generate_distributions()
    {
      // we generate 0hw, 1hw, 2hw, ... , Nmaxhw distribution of nucleons
      std::vector<std::set<tSHELLSCONFIG>> ConfRepository(Nmax_ + 1);

      tSHELLSCONFIG v0hwShellConfiguration;
      // setOfShellConfigurations stores a set of unique distributions for a
      // given Nhw subspace It is a set because the algorithm that is being used
      // can produce several identical distributions
      std::set<tSHELLSCONFIG> setOfShellConfigurations;

      int iNParticlesLeft = A_;
      unsigned nmax; // the last HO shell available
      unsigned n = 0;

      // This loop creates 0hw shell configuration
      while (iNParticlesLeft > 0)
      {
        // number of nucleons that can fit into n-th HO shell
        int N = 2 * (n + 1) * (n + 2);
        if ((iNParticlesLeft - N) > 0)
        {
          v0hwShellConfiguration.push_back(N);
          iNParticlesLeft -= N;
          n++;
        }
        else
        {
          v0hwShellConfiguration.push_back(iNParticlesLeft);
          iNParticlesLeft = 0;
        }
      }
      // This is maximal HO shell that can contain maximally 1 particle
      int last_shell = nmax = Nmax_ + n;

      // We need to resize vShellConfiguration up to nmax+1 elements
      v0hwShellConfiguration.resize(nmax + 1, 0);
      // There is only one 0hw shell configuration
      setOfShellConfigurations.insert(v0hwShellConfiguration);
      ConfRepository[0] = setOfShellConfigurations;

      // Iterate through N\hbar\Omega subspaces up to Nmax\hbar\Omega
      // and for each subspace (Nsubspace) generate allowed
      // shell configurations
      for (size_t Nsubspace = 1; Nsubspace <= Nmax_; Nsubspace++)
      {
        // We will reuse setOfShellConfigurations data structure over and over
        // again So we need to clean it first
        setOfShellConfigurations.clear();

        std::set<tSHELLSCONFIG>::const_iterator cit =
            ConfRepository[Nsubspace - 1].begin();
        for (; cit != ConfRepository[Nsubspace - 1].end(); cit++)
        {
          // vSeedConfiguration contains (Nsubspace-1) subspace distribution of
          // fermions over the shells
          tSHELLSCONFIG vSeedShellConfiguration(*cit);
          for (size_t q = 0; q < nmax; q++)
          {
            // vNewShellConfiguration contains Nsubspace subspace distribution
            // of fermions over the shells
            tSHELLSCONFIG vNewShellConfiguration(vSeedShellConfiguration);
            if (vNewShellConfiguration[q] &&
                (vNewShellConfiguration[q + 1] + 1) <= (2 * (q + 2) * (q + 3)))
            {
              vNewShellConfiguration[q] -= 1;
              vNewShellConfiguration[q + 1] += 1;
              // STL set<> datastructure takes care of duplicities
              // Thus we need not explicitly check whether
              // vNewShellConfiguration has been already encountered and stored
              setOfShellConfigurations.insert(vNewShellConfiguration);
            }
          }
        }
        // setOfShellConfigurations contains all (Nsubspace)hw shell
        // configurations let's store it in ConfRepository[Nsubspace]
        ConfRepository[Nsubspace] = setOfShellConfigurations;
      }

      nhw_dist_ids_.clear();
      nhw_dist_ids_.push_back(0);

      distributions_.clear();

      // iterate through all shell configurations belonging to N = Nsubspace
      // subspace Transfer data from ConfRepository into m_ShellConfigurations
      for (size_t Nsubspace = 0; Nsubspace <= Nmax_; Nsubspace++)
      {
        std::set<tSHELLSCONFIG>::const_iterator citShellConfs =
            ConfRepository[Nsubspace].begin();
        for (; citShellConfs != ConfRepository[Nsubspace].end();
             ++citShellConfs)
          distributions_.push_back(*citShellConfs);

        nhw_dist_ids_.push_back(distributions_.size());
      }
    }

    ////////////////////////// U4 /////////////////////////////
    void Couple(const MU4& a, const MU4& b, MU4_VEC& resulting_mult_u4)
    {
      auto [u4s, mults] = u4::decomposition(a.u4, b.u4);
      for (int i = 0; i < u4s.size(); i++)
      {
        MU4 mu4;
        mu4.mult = mults[i];
        mu4.u4 = u4s[i];
        resulting_mult_u4.push_back(mu4);
      }
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

        for (std::size_t dist_id = dist_id_begin; dist_id < dist_id_end; dist_id++)
          generate_gammas_for_dist(dist_id);
      }
    }

    void generate_gammas_for_dist(std::size_t dist_id)
    {
   // uint64_t dim = 0;

      assert(dist_id < distributions_.size());
      const auto& distribution = distributions_[dist_id];

      std::vector<std::vector<std::pair<MU4, MSU3>>> vvU4SU3;
      int n = 0; // HO shell
      for (int n = 0; n < distribution.size(); n++)
      {
        int A = distribution[n];
        if (A == 0)
          continue;
        std::vector<std::pair<MU4, MSU3>> allowedU4SU3irreps;
        generateSU3U4(n, A, allowedU4SU3irreps);
        vvU4SU3.push_back(allowedU4SU3irreps);
      }

      // Now we generate all possible combinations of coupling out of
      // vvU4SU3 irreps
      size_t NIrrepsSets = vvU4SU3.size();
      std::vector<unsigned> vElemsPerChange(
          NIrrepsSets, 1); // Fill with 1 cause vElemsPerChange[0] = 1;
      std::vector<size_t> vNSizes(NIrrepsSets);
      // uNAllowedCombinations = number of all possible coupling combinations
      // uNAllowedCombinations = vvSU3xSU2Irreps[0].size()* ...
      // *vvSU3xSU2Irreps[NIrrepsSets].size()
      unsigned uNAllowedCombinations = 1;

      for (size_t i = 0; i < NIrrepsSets; i++)
      {
        vNSizes[i] = vvU4SU3[i].size();
        uNAllowedCombinations *= vNSizes[i];
      }

      for (size_t i = 1; i < NIrrepsSets; i++)
      {
        // if i == 0 => vElemsPerChange[0]=1 since constructor
        vElemsPerChange[i] = vElemsPerChange[i - 1] * vNSizes[i - 1];
      }

      assert(dist_id < dist_gammas_.size());
      assert(gammas_u4_.size() == gammas_su3_.size());

      auto& pair = dist_gammas_[dist_id];
      std::get<0>(pair) = gammas_.size();
      std::get<1>(pair) = uNAllowedCombinations;

      for (unsigned index = 0; index < uNAllowedCombinations; index++)
      {
        // in each step create a set of SU3xSU2 irreps to couple
        std::vector<MU4> gammas_u4;
        std::vector<MSU3> gammas_su3;
        for (size_t i = 0; i < NIrrepsSets; i++)
        {
          size_t iElement = (index / vElemsPerChange[i]) % vNSizes[i];
          gammas_u4.push_back(vvU4SU3[i][iElement].first);
          gammas_su3.push_back(vvU4SU3[i][iElement].second);
        }

        /*
        for (size_t i = 0; i < gammas_u4.size(); ++i)
        {
          MSU3 msu3 = gammas_su3[i];
          MU4 mu4 = gammas_u4[i];
          std::cout << "\tgamma[" << i << "]:";
          std::cout << mu4.mult << "[" << mu4.u4.n14() << " " << mu4.u4.n24()
                    << " " << mu4.u4.n34() << " " << mu4.u4.n44() << "] ";
          std::cout << msu3.mult << "(" << msu3.lm << " " << msu3.mu << ")"
                    << std::endl;
        }
        */

        assert(gammas_u4.size() == gammas_su3.size());
        assert(gammas_u4.size() > 0);

        gammas_.emplace_back(gammas_u4_.size(), gammas_u4.size());

        gammas_u4_.insert(gammas_u4_.end(), gammas_u4.begin(), gammas_u4.end());
        gammas_su3_.insert(gammas_su3_.end(), gammas_su3.begin(),
                           gammas_su3.end());

        /*
        if (gammas_u4.size() > 1)
        {
          dim += dimGamma(gammas_u4, gammas_su3, MMT, JJ);
        }
        else
        {
          MSU3 msu3 = gammas_su3[0];
          MU4 mu4 = gammas_u4[0];
          assert(mu4.mult == 1);
          uint32_t mult = msu3.mult;
          dim += mult * computeDim(msu3.lm, msu3.mu, mu4.u4, MMT, JJ);
        }
        std::cout << std::endl;
        */
      }

   // dim_ += dim;
    }

    void generate_omegas()
    {
      gamma_omegas_.clear();
      omegas_.clear();
      omegas_u4_.clear();
      omegas_su3_.clear();

      gamma_omegas_.resize(gammas_.size());

      for (std::size_t gamma_id = 0; gamma_id < gammas_.size(); gamma_id++)
        generate_omegas_for_gamma(gamma_id);
    }

    void generate_omegas_for_gamma(std::size_t gamma_id)
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
        dim_ += mult * computeDim(msu3.lm, msu3.mu, mu4.u4);
      }
      else
        generate_omegas_for_gamma(gamma_id, gamma_u4, gamma_su3);
    }

    int computeDim(int lm, int mu, int SS)
    {
      int dim = 0;
      for (int l = 0; l <= lm + mu; ++l)
      {
        int k0max = kmax(lm, mu, l);
        if (k0max)
        {
          for (size_t jj = abs(2 * l - SS); jj <= (2 * l + SS); jj += 2)
            if ((jj == JJ_) || (JJ_ == -1))
              dim += k0max;
        }
      }
      return dim;
    }

    int computeDim(int lm, int mu, const u4::irrep& u4)
    {
      int dim = 0;
      auto u4_basis = u4::physicalBasis(u4);
      for (auto const& labels : u4_basis)
      {
        if (labels.TT >= MMT_)
        {
          int mult = labels.zeta;
          int SS = labels.SS;
          dim += mult * computeDim(lm, mu, SS);
        }
      }
      return dim;
    }

  private:
    int su3mult(int lm1, int mu1, int lm2, int mu2, int lm3, int mu3)
    {
      int result = 0;
      int phase = lm1 + lm2 - lm3 - mu1 - mu2 + mu3;

      if (phase % 3 == 0)
      {
        int redphase;
        int l1, l2, l3, m1, m2, m3;

        redphase = phase / 3;
        if (phase >= 0)
        {
          l1 = lm1;
          l2 = lm2;
          l3 = lm3;
          m1 = mu1;
          m2 = mu2;
          m3 = mu3;
        }
        else
        {
          l1 = mu1;
          l2 = mu2;
          l3 = mu3;
          m1 = lm1;
          m2 = lm2;
          m3 = lm3;
          redphase = -redphase;
        }
        phase = redphase + m1 + m2 - m3;

        int invl1 = std::min(l1 - redphase, m2);
        if (invl1 < 0)
          return result;

        int invl2 = std::min(l2 - redphase, m1);
        if (invl2 < 0)
          return result;
        result = std::max(
            std::min(phase, invl2) - std::max(phase - invl1, 0) + 1, 0);
      }
      return result;
    }

    int kmax(int lm, int mu, int L)
    {
      return (std::max(0, (lm + mu + 2 - L) / 2) -
              std::max(0, (lm + 1 - L) / 2) - std::max(0, (mu + 1 - L) / 2));
    }

    void generate_U4_irreps(int A, int n, std::vector<u4::irrep>& u4_irreps)
    {
      int nsps = (n + 1) * (n + 2) / 2;
      if (A > 4 * nsps)
        return;
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
              u4_irreps.push_back(u4::irrep(n14, n24, n34, n44));
          }
        }
      }
    }

    void generateSU3U4(int n, int A,
                       std::vector<std::pair<MU4, MSU3>>& allowedSU3U4irreps)
    {
      untou3::UNtoU3<4> gen;
      gen.set_shell(n);
      int N = (n + 1) * (n + 2) / 2;

      std::vector<u4::irrep> u4_irreps;
      generate_U4_irreps(A, n, u4_irreps);
      for (auto u4 : u4_irreps)
      {
        gen.reduceU4(u4, N);
        auto res = gen.irreps();
        for (const auto& [u3, mult] : res)
        {
          int lm = u3.lam();
          int mu = u3.mu();
          allowedSU3U4irreps.push_back(
              std::make_pair(MU4(1, u4), MSU3(mult, lm, mu)));
        }
      }
    }

    void couple(int lm1, int mu1, int lm2, int mu2, MSU3_VEC& ir3)
    {
      ir3.reserve(3 * (lm1 + mu1 + lm2 + mu2));

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
            ir3.push_back(
                MSU3(su3mult(lm1, mu1, lm2, mu2, lm3, mu3), lm3, mu3));
          }
        }
      }
    }

    void Couple(const MSU3& a, const MSU3& b, MSU3_VEC& result)
    {
      int lm1 = a.lm;
      int mu1 = a.mu;
      int lm2 = b.lm;
      int mu2 = b.mu;
      couple(lm1, mu1, lm2, mu2, result);
    }

    template <typename T, std::size_t Extent>
    void gamma2omega(std::span<T, Extent> gamma, std::vector<std::vector<T>>& omega)
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

    void generate_omegas_for_gamma(std::size_t gamma_id, auto gamma_u4, auto gamma_su3)
    {
      uint64_t dim = 0;

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

          /*
          std::cout << "\t\t";
          for (size_t i = 0; i < mu4_vec.size(); i++)
          {
            MSU3 msu3 = msu3_vec[i];
            MU4 mu4 = mu4_vec[i];
            std::cout << mu4.mult << "[" << mu4.u4.n14() << " " << mu4.u4.n24()
                      << " " << mu4.u4.n34() << " " << mu4.u4.n44() << "] ";
            std::cout << msu3.mult << "(" << msu3.lm << " " << msu3.mu
                      << ")   ";
          }
          */

          uint32_t mult_su3 = 1;
          for (size_t i = 0; i < msu3_vec.size(); ++i)
            mult_su3 *= msu3_vec[i].mult;
          uint32_t mult = mult_gamma * mult_u4 * mult_su3;
          MU4 mu4 = mu4_vec.back();
          MSU3 msu3 = msu3_vec.back();
          u4::irrep u4 = mu4.u4;
          int lm = msu3.lm;
          int mu = msu3.mu;
          uint32_t dim_irrep = computeDim(lm, mu, u4);
          dim += mult * dim_irrep;

          /*
          std::cout << " mult:" << mult << " dim:" << mult * dim_irrep
                    << std::endl;
          */

          if (mult * dim_irrep > 0)
          {
            omegas_.emplace_back(omegas_u4_.size(), mu4_vec.size(), mult,
                                 mult * dim_irrep);

            for (std::size_t i = 0; i < mu4_vec.size(); i++)
            {
              omegas_u4_.push_back(mu4_vec[i]);
              omegas_su3_.push_back(msu3_vec[i]);
            }

            n_omegas++;
          }
        }
      }

      dim_ += dim;
    }

    /*
    std::vector<int> UN_to_UN(const std::vector<size_t>& un5)
    {
      int N = std::reduce(un5.begin(), un5.end());
      std::vector<int> un(N, 0);

      size_t index = 0;
      for (size_t i = 0; i <= 4; ++i)
      {
        int A = 4 - i;
        // n ... number of labels with A particles
        int n = un5[i];
        for (size_t j = 1; j <= n; ++j)
          un[index++] = A;
      }
      return un;
    }
    */
  };
} // namespace U4xSU3
