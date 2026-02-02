#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include <qcx_convert.h>
#include <qcx_ND.h>
#include <qcx_print.h>
#include <qcx_tree_generator.h>
#include <so3Coeffs/SO3_CGC.h>

#include <U3basis.h>
#include <U3decomposition.h>
#include <U3irrep.h>

#include <U4basis.h>
#include <U4decomposition.h>
#include <U4irrep.h>

#include <UNtoU3.h>

#include <U3xU4GammaOmega.h>
#include <U3xU4irrep.h>
#include <U3xU4ManyNucleonBasis.h>

namespace U3xU4
{
  // **** hw_subspace ****

  // selection rules: used during truncation
  bool hw_subspace::U3_selection(const u3::irrep& irr) const
  {
    if (u3s.empty())
      return true;
    else
    {
      bool flag = false;
      for (const auto& ir : u3s)
        if (ir == irr)
        {
          flag = true;
          break;
        }
      return flag;
    }
  }

  bool hw_subspace::U4_selection(const u4::irrep& irr) const
  {
    if (u4s.empty())
      return true;
    else
    {
      bool flag = false;
      for (const auto& ir : u4s)
        if (ir == irr)
        {
          flag = true;
          break;
        }
      return flag;
    }
  }

  bool hw_subspace::SS_selection(qcx_utils::Int8 SS) const
  {
    if (SSs.empty())
      return true;
    else
    {
      bool flag = false;
      for (const auto& ir : SSs)
        if (ir == static_cast<int>(SS))
        {
          flag = true;
          break;
        }
      return flag;
    }
  }

  bool hw_subspace::TT_selection(qcx_utils::Int8 TT) const
  {
    if (TTs.empty())
      return true;
    else
    {
      bool flag = false;
      for (const auto& ir : TTs)
        if (ir == static_cast<int>(TT))
        {
          flag = true;
          break;
        }
      return flag;
    }
  }

  template <typename T>
  static void apply_permutation_by_indices(
    std::vector<T>& vec, const std::vector<size_t>& indices)
  {
    std::vector<T> tmp;
    tmp.reserve(vec.size());
    for (size_t idx : indices) tmp.push_back(vec[idx]);     
    vec = std::move(tmp);
  }

  // sort u3s in decreasing order of C2
  void hw_subspace::sort_u3_by_C2_decreasing()
  {
    assert(!u3s.empty() && u4s.empty() && SSs.empty() && TTs.empty());

    // build index array [0,1,2,...]
    std::vector<size_t> indices;
    for (size_t i = 0; i < u3s.size(); i++) indices.push_back(i);

    // sort indices by u3s[*].C2() decreasing
    std::stable_sort(
      indices.begin(), indices.end(), [&](size_t i, size_t j)
    { 
      const auto& a = u3s[i];
      const auto& b = u3s[j];

      return std::make_tuple(a.C2(), a.n13(), a.n23(), a.n33()) >
             std::make_tuple(b.C2(), b.n13(), b.n23(), b.n33());
    });

    // apply permutation consistently
    apply_permutation_by_indices(u3s, indices);
    if (!probs.empty()) apply_permutation_by_indices(probs, indices);
  }

  void hw_subspace::sort_u4_by_C2_increasing()
  {
    assert(u3s.empty() && !u4s.empty() && SSs.empty() && TTs.empty());

    // build index array [0,1,2,...]
    std::vector<size_t> indices;
    for (size_t i = 0; i < u4s.size(); i++) indices.push_back(i);

    // sort indices by u4s[*].C2() increasing
    std::stable_sort(
      indices.begin(), indices.end(), [&](size_t i, size_t j)
    { 
      const auto& a = u4s[i];
      const auto& b = u4s[j];

      return std::make_tuple(a.C2(), a.n14(), a.n24(), a.n34(), a.n44()) <
             std::make_tuple(b.C2(), b.n14(), b.n24(), b.n34(), b.n44());
    });

    // apply permutation consistently
    apply_permutation_by_indices(u4s, indices);
    if (!probs.empty()) apply_permutation_by_indices(probs, indices);
  }

  void hw_subspace::sort_SS()
  {
    assert(u3s.empty() && u4s.empty() && !SSs.empty() && TTs.empty());

    // build index array [0,1,2,...]
    std::vector<size_t> indices;
    for (size_t i = 0; i < SSs.size(); i++) indices.push_back(i);

    // sort indices by SS increasing
    std::stable_sort(indices.begin(), indices.end(), [
      &](size_t i, size_t j) { return SSs[i] < SSs[j]; });

    // apply permutation consistently
    apply_permutation_by_indices(SSs, indices);
    if (!probs.empty()) apply_permutation_by_indices(probs, indices);
  }

  void hw_subspace::sort_TT()
  {
    assert(u3s.empty() && u4s.empty() && SSs.empty() && !TTs.empty());

    // build index array [0,1,2,...]
    std::vector<size_t> indices;
    for (size_t i = 0; i < TTs.size(); i++) indices.push_back(i);

    // sort indices by SS increasing
    std::stable_sort(indices.begin(), indices.end(), [
      &](size_t i, size_t j) { return TTs[i] < TTs[j]; });

    // apply permutation consistently
    apply_permutation_by_indices(TTs, indices);
    if (!probs.empty()) apply_permutation_by_indices(probs, indices);
  }

  std::ostream& operator<<(std::ostream& os, const hw_subspace& s)
  {
    os << "subspace hw = " << static_cast<int>(s.hw) << "\n";
    if (!s.u3s.empty())
    {
      if (!s.probs.empty())
      {
        os << "u3::irrep,C2 ===> probabilities \n";
        for (size_t i = 0; i < s.u3s.size(); i++)
        {
          os << s.u3s[i] << s.u3s[i].C2() << " ===> ";
          for (const auto& p : s.probs[i]) os << p << "%\t";
          os << "\n";
        }
      }
      else
      {
        os << "u3::irrep,C2\n";
        for (const auto& irr : s.u3s) os << irr << irr.C2() << "\n";
      }
    }
    
    if (!s.u4s.empty())
    {
      if (!s.probs.empty())
      {
        os << "u4::irrep,C2 ===> probabilities \n";
        for (size_t i = 0; i < s.u4s.size(); i++)
        {
          os << s.u4s[i] << s.u4s[i].C2() << " ===> ";
          for (const auto& p : s.probs[i]) os << p << "%\t";
          os << "\n";
        }
      }
      else
      {
        os << "u4::irrep,C2\n";
        for (const auto& irr : s.u4s) os << irr << irr.C2() << "\n";
      }
    }

    if (!s.SSs.empty())
    {
      if (!s.probs.empty())
      {
        os << "SS ===> probabilities \n";
        for (size_t i = 0; i < s.SSs.size(); i++)
        {
          os << s.SSs[i] << " ===> ";
          for (const auto& p : s.probs[i]) os << p << "%\t";
          os << "\n";
        }
      }
      else
      {
        os << "SS\n";
        for (const auto& irr : s.SSs) os << irr << "\n";
      }
    }

    if (!s.TTs.empty())
    {
      if (!s.probs.empty())
      {
        os << "TT ===> probabilities \n";
        for (size_t i = 0; i < s.TTs.size(); i++)
        {
          os << s.TTs[i] << " ===> ";
          for (const auto& p : s.probs[i]) os << p << "%\t";
          os << "\n";
        }
      }
      else
      {
        os << "TT\n";
        for (const auto& irr : s.TTs) os << irr << "\n";
      }
    }

    return os;
  }

  // print to file
  void hw_subspace::print_to_file(std::ostream& os) const
  {
    os << static_cast<int>(hw) << "\n"
       << static_cast<int>(u3s.size()) << "\n"
       << static_cast<int>(u4s.size()) << "\n"
       << static_cast<int>(SSs.size()) << "\n"
       << static_cast<int>(TTs.size()) << "\n"
       << static_cast<int>(probs.size()) << "\n"
       << static_cast<int>(probs[0].size()) << "\n";

    if (!u3s.empty())
      for (size_t i = 0; i < u3s.size(); i++)
      {
        os << u3s[i].n13() << " " << u3s[i].n23() << " " << u3s[i].n33()
           << "\t" << u3s[i].C2() << "\t";

        for (const auto& p : probs[i]) os << p << "\t";
        os << "\n";
      }

    if (!u4s.empty())
      for (size_t i = 0; i < u4s.size(); i++)
      {
        os << u4s[i].n14() << " " << u4s[i].n24() << " " 
           << u4s[i].n34() << " " << u4s[i].n44()
           << "\t" << u4s[i].C2() << "\t";

        for (const auto& p : probs[i]) os << p << "\t";
        os << "\n";
      }

    if (!SSs.empty())
      for (size_t i = 0; i < SSs.size(); i++)
      {
        os << SSs[i] << "\t";
        for (const auto& p : probs[i]) os << p << "\t";
        os << "\n";
      }

    if (!TTs.empty())
      for (size_t i = 0; i < TTs.size(); i++)
      {
        os << TTs[i] << "\t";
        for (const auto& p : probs[i]) os << p << "\t";
        os << "\n";
      }

    return;
  }

  // **** ManyNucleonBasis::IrrepDB ****

  ManyNucleonBasis::IrrepDB::IrrepDB(const ManyNucleonBasis& basis)
      : basis_(basis)
  { }

  ManyNucleonBasis::Irrep ManyNucleonBasis::IrrepDB::operator[](size_t irrep_id)
  {
    assert(irrep_id < basis_.irreps_.size());

    auto [dist_id, gamma_id, omega_u4_id, omega_su3_id, offset] =
        basis_.irreps_[irrep_id];

    auto dist = basis_.distribution(dist_id);

    // qcx_utils::print("dist_id {}: {};\t", dist_id, dist);
    // qcx_utils::print("gamma_id {};\t irrep_id {}\n", gamma_id, irrep_id);

    // active shells:
    active_shells_.clear();
    for (size_t n = 0; n < dist.size(); n++)
      if (dist[n] > 0)
        active_shells_.push_back(n);

    auto gamma = basis_.gamma(gamma_id);

    assert(gamma.size() > 0);
    assert(gamma.size() == active_shells_.size());

    // create resulting gamma:
    gamma_.clear();
    for (size_t i = 0; i < gamma.size(); i++)
    {
      auto n = active_shells_[i];
      const auto& ge = gamma[i];

      auto A = ge.u4.quanta(); // particles
      auto N = A * n;          // quanta

      u3::irrep u3(ge.su3, N);

      gamma_.emplace_back(n, ge.alphamax, u3, ge.u4);
    }

    assert(gamma_.size() == gamma.size());

    if (gamma.size() == 1)
    {
      assert(omega_u4_id == INVALID_OMEGA_ID);
      assert(omega_su3_id == INVALID_OMEGA_ID);

      return {std::move(gamma_), {}};
    }
    else
    {
      auto omega_u4 = basis_.omega_u4(omega_u4_id);
      auto omega_su3 = basis_.omega_su3(omega_su3_id);

      assert(omega_u4.size() == gamma.size() - 1);
      assert(omega_su3.size() == gamma.size() - 1);

      // combine omega U(4) and SU(3) parts:
      omega_.clear();
      for (size_t i = 0; i < omega_u4.size(); i++)
      {
        const auto& oe_u4 = omega_u4[i];
        const auto& oe_su3 = omega_su3[i];

        omega_.emplace_back(oe_su3.rhomax, oe_su3.su3, oe_u4.etamax, oe_u4.u4);
      }

      return {std::move(gamma_), std::move(omega_)};
    }
  }

  size_t ManyNucleonBasis::IrrepDB::offset(size_t irrep_id) const
  {
    assert(irrep_id < basis_.irreps_.size());
    return basis_.irreps_[irrep_id].offset;
  }

  // **** ManyNucleonBasis ****

  ManyNucleonBasis::ManyNucleonBasis(int N, int Z, int Nmax, int JJ)
      : A_(Z + N),
        MMT_(N - Z),
        Nmax_(Nmax),
        JJ_(JJ),
        n0hw_(Nmax % 2),
        dim_(0)
  { }

  ManyNucleonBasis::ManyNucleonBasis(int N, int Z, int Nmax, int JJ,
                     const std::vector<hw_subspace>& hw_subspaces)
      : A_(Z + N),
        MMT_(N - Z),
        Nmax_(Nmax),
        JJ_(JJ),
        n0hw_(Nmax % 2),
        dim_(0),
        hw_subspaces_(hw_subspaces)
  { }

  void ManyNucleonBasis::construct()
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

  size_t ManyNucleonBasis::n_irreps() const { return irreps_.size(); }

  void ManyNucleonBasis::generate_distributions()
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
    quanta_0hw = 0;

    while (nParticlesLeft > 0)
    {
      // number of nucleons that can fit into nth HO shell:
      int N = 2 * (n + 1) * (n + 2);

      auto place = std::min(nParticlesLeft, N);
      assert(place > 0);

      quanta_0hw += n * place;
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

  void ManyNucleonBasis::generate_gammas()
  {
    group_dists.resize(n_distributions());

    for (size_t dist_id = 0; dist_id < n_distributions(); dist_id++)
      generate_gammas_for_distribution(dist_id);
  }

  void ManyNucleonBasis::generate_gammas_for_distribution(size_t dist_id)
  {
    assert(dist_id < n_distributions());

    const auto dist = distribution(dist_id);

    // (1) generate gamma elements for all non-vacant shells:

    // gamma elements for non-vacant shells:
    std::vector<std::vector<GammaElement>> shells_elements;

    int last_nonempty_shell;
    for (int n = 0; n < dist.size(); n++)
    {
      int A = dist[n];

      if (A == 0) continue;

      last_nonempty_shell = n;

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

        // total quanta is determined from one shell only
        int quanta = ge.u4.quanta() * last_nonempty_shell;
        int hw = quanta - quanta_0hw;

        bool select = true;
        if (isTruncated())
        {
          u4::irrep u4_(ge.u4.P1(), ge.u4.P2(), ge.u4.P3());
          for (const auto& subspace : hw_subspaces_)
          {
            if (subspace.hw == hw)
            {
              select = subspace.U3_selection(ge.su3) &&
                       subspace.U4_selection(u4_);
              break;
            }
          }
        }
        if (!select) continue;

        // auto dim_gamma = ge.alphamax * compute_dim(ge.u4, ge.su3);

        auto [LLSSTTs, dim_irrep] = decomposition(hw, ge.u4, ge.su3);
        auto dim_gamma = ge.alphamax * dim_irrep;

        if (dim_gamma > 0)
        {
          // add irrep to basis:
          group_dists[dist_id].push_back(
            {static_cast<uint32_t>(gamma_id), {irreps_.size()}});

          irreps_.emplace_back(dist_id, gamma_id, INVALID_OMEGA_ID,
                               INVALID_OMEGA_ID, dim_);

          LLSSTTs_.push_back(std::move(LLSSTTs));

          dim_ += dim_gamma;
        }
      }
      else
      {
        std::vector<size_t> irrep_ids;
        // generate omegas for gamma:
        generate_omegas_for_gamma(dist_id, gamma_id, irrep_ids);

        group_dists[dist_id].push_back(
          {static_cast<uint32_t>(gamma_id), std::move(irrep_ids)});
      }
    }
  }

  void ManyNucleonBasis::generate_single_shell_gamma_elements(
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

  void ManyNucleonBasis::generate_U4_irreps(int n, int A,
                                            std::vector<u4::irrep>& u4_irreps)
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

  int ManyNucleonBasis::compute_dim(const u4::irrep& u4, const su3::irrep& su3)
  {
    int dim = 0;

    auto u4_basis = u4::physicalBasis(u4);

    for (auto const& labels : u4_basis)
    {
      if (labels.TT() >= abs(MMT_))
      {
        int zeta = labels.zeta();
        int SS = labels.SS();

        dim += zeta * compute_dim(su3, SS);
      }
    }

    return dim;
  }

  int ManyNucleonBasis::compute_dim(const su3::irrep& su3, int SS)
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

  std::pair<std::vector<std::array<int,6>>, int> 
  ManyNucleonBasis::decomposition(int hw, const u4::irrep& u4, const su3::irrep& su3)
  {
    // int dim = 0;
    std::vector<std::array<int,6>> LLSSTTs;

    auto pv4s = u4::physicalBasis(u4);
    auto pv3s = u3::physicalBasis(su3);

    int offset = 0;
    for (const auto& pv4 : pv4s)
    {
      int z0max = pv4.zeta(), SS = pv4.SS(), TT = pv4.TT();

      if (TT < abs(MMT_)) continue;

      bool select = true;
      if (isTruncated())
      {
        for (const auto& subspace : hw_subspaces_)
        {
          if (subspace.hw == hw)
          {
            select = subspace.SS_selection(SS) &&
                     subspace.TT_selection(TT);
            break;
          }
        }
      }
      if (!select) continue;
      
      // int dim_su3 = 0;
      for (const auto& pv3 : pv3s)
      {
        int k0max = pv3.kappa(), LL = 2 * pv3.L();

        for (int jj = abs(LL - SS); jj <= (LL + SS); jj+= 2)
          if (jj == JJ_ || JJ_ == -1) 
          {
            // dim_su3 += k0max;
            LLSSTTs.push_back({k0max, LL, z0max, SS, TT, offset});
            offset += z0max * k0max;
          }
      }
      // dim += z0max * dim_su3;
    }
    return std::make_pair(LLSSTTs, offset);
    // return std::make_pair(LLSSTTs, dim);
  }

  void ManyNucleonBasis::generate_omegas_for_gamma(size_t dist_id,
                                                   size_t gamma_id,
                                                   std::vector<size_t>& irrep_ids)
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

    // (3) determine hw subspace & total gamma multiplicity:
    uint32_t mult_gamma = 1;
    int quanta = 0;

    auto dist = this->distribution(dist_id);
    std::vector<uint16_t> active_shells;
    for (size_t n = 0; n < dist.size(); n++)
      if (dist[n] > 0) active_shells.push_back(n);

    assert(gamma.size() == active_shells.size());

    for (size_t i = 0; i < gamma.size(); i++)
    {
      const auto& ge = gamma[i];
      mult_gamma *= ge.alphamax;
      quanta += active_shells[i] * ge.u4.quanta();
    }
    int hw = quanta - quanta_0hw;

    // (4) add irreps:

    for (size_t i = 0; i < omegas_u4.size(); i++)
    {
      const auto& omega_u4 = omegas_u4[i];
      const auto& last_oe_u4 = omega_u4.back();
      auto omega_id_u4 = first_omega_u4_id + i;

      bool select_u4 = true;
      if (isTruncated())
      {
        for (const auto& subspace : hw_subspaces_)
        {
          if (subspace.hw == hw)
          {
            select_u4 = subspace.U4_selection(last_oe_u4.u4);
            break;
          }
        }
      }
      if (!select_u4) continue;

      uint32_t mult_u4 = 1;
      for (const auto& oe_u4 : omega_u4)
        mult_u4 *= oe_u4.etamax;

      for (size_t j = 0; j < omegas_su3.size(); j++)
      {
        const auto& omega_su3 = omegas_su3[j];
        const auto& last_oe_su3 = omega_su3.back();
        auto omega_id_su3 = first_omega_su3_id + j;

        bool select_su3 = true;
        if (isTruncated())
        {
          for (const auto& subspace : hw_subspaces_)
          {
            if (subspace.hw == hw)
            {
              select_su3 = subspace.U3_selection(last_oe_su3.su3);
              break;
            }
          }
        }
        if (!select_su3) continue;

        uint32_t mult_su3 = 1;
        for (const auto& oe_su3 : omega_su3)
          mult_su3 *= oe_su3.rhomax;

        auto mult = mult_gamma * mult_u4 * mult_su3;

        // auto dim_irrep = compute_dim(last_oe_u4.u4, last_oe_su3.su3);
        auto [LLSSTTs, dim_irrep] = 
                      decomposition(hw, last_oe_u4.u4, last_oe_su3.su3);

        auto dim_omega = mult * dim_irrep;

        if (dim_omega > 0)
        {
          // add irrep to basis:
          irrep_ids.push_back(irreps_.size());

          irreps_.emplace_back(dist_id, gamma_id, omega_id_u4, omega_id_su3,
                               dim_);

          LLSSTTs_.push_back(std::move(LLSSTTs));

          dim_ += dim_omega;
        }
      }
    }
  }

/*
  std::vector<std::vector<std::array<int,6>>> ManyNucleonBasis::decomposition() const
  {
    auto irrs = irreps();

    std::vector<std::vector<std::array<int,6>>> labels(irrs.size());

    for (size_t irr_id = 0; irr_id < irrs.size(); irr_id++)
    {
      const auto& irr = irrs[irr_id];

      u3::physicalvectors u3_basis = (irr.omega.empty())
        ? u3::physicalBasis(irr.gamma.back().u3())
        : u3::physicalBasis(irr.omega.back().u3());

      u4::physicalvectors u4_basis = (irr.omega.empty())
        ? u4::physicalBasis(irr.gamma.back().u4())
        : u4::physicalBasis(irr.omega.back().u4());

      int offset = 0;
      for (const auto& u4_labels : u4_basis)
      {
        if (u4_labels.TT() < abs(MMT_)) continue;

        for (const auto& u3_labels : u3_basis)
        {
          if (u2::mult(2 * u3_labels.L(), u4_labels.SS(), JJ()) == 1)
          {
            labels[irr_id].push_back({u3_labels.kappa(), 2 * u3_labels.L(),
              u4_labels.zeta(), u4_labels.SS(), u4_labels.TT(),offset});
            offset += (u3_labels.kappa() * u4_labels.zeta());
          }
        }
      }
    }

    return labels;
  }
*/

  std::vector<std::array<int,6>> 
    ManyNucleonBasis::decomposition(const Irrep& irr) const
  {
    std::vector<std::array<int,6>> labels;

    u3::physicalvectors u3_basis = (irr.omega.empty())
        ? u3::physicalBasis(irr.gamma.back().u3())
        : u3::physicalBasis(irr.omega.back().u3());

    u4::physicalvectors u4_basis = (irr.omega.empty())
        ? u4::physicalBasis(irr.gamma.back().u4())
        : u4::physicalBasis(irr.omega.back().u4());

    int offset = 0;
    for (const auto& u4_labels : u4_basis)
    {
      if (u4_labels.TT() < abs(MMT_)) continue;

      for (const auto& u3_labels : u3_basis)
      {
        if (u2::mult(2 * u3_labels.L(), u4_labels.SS(), JJ()) == 1)
        {
          labels.push_back({u3_labels.kappa(), 2 * u3_labels.L(),
            u4_labels.zeta(), u4_labels.SS(), u4_labels.TT(),offset});
            offset += (u3_labels.kappa() * u4_labels.zeta());
        }
      }
    }
    
    return labels;
  }

} // namespace U3xU4
