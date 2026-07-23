#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <U3xU4GammaOmega.h>
#include <U3xU4irrep.h>
#include <U3xU4ManyNucleonBasis.h>
#include <U3xU4MultiShellTensors.h>
#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  std::optional<MultiShellTensor>
  unify_tensor(const MultiShellTensor& T, const ManyNucleonBasis::Irrep& irrep)
  {
    // std::cout << "\nTensor before unifying: " << T << std::endl;

    // (1) setup:

    const auto& Gamma = T.Gammas();
    const auto& Omega = T.Omegas();

    // std::cout << "Before unifying Gamma:" << std::endl;
    // for (const auto& G : Gamma) std::cout << G << std::endl;

    assert(Omega.size() + 1 == Gamma.size());

    const auto& gamma = irrep.gamma;
    const auto& omega = irrep.omega;

    assert(omega.size() + 1 == gamma.size());

    // (2) check if unification is possible:

    if (Gamma.size() > gamma.size())
        return {};

    auto tensor_shells = extract_shells_from_gamma(Gamma);
    auto irrep_shells = extract_shells_from_gamma(gamma);

    for (auto shell : tensor_shells)
      if (std::ranges::find(irrep_shells, shell) == irrep_shells.end())
        return {};

    // (3) setup continued:

    U3xU4::SingleShellTensors oGamma;
    U3xU4::OmegaElements oOmega;

    oGamma.reserve(gamma.size());
    oOmega.reserve(omega.size());

    /*
    std::vector<uint8_t> added;
    added.reserve(gamma.size());
    */

    // (4) unify tensor gamma:

    size_t i_Gamma = 0;

    for (const auto& g : gamma)
    {
      auto shell = g.shell();

      if ((i_Gamma < Gamma.size()) && (Gamma[i_Gamma].shell() == shell))
      {
        // (1) there is a corresponding tensor Gamma element:
        oGamma.push_back(Gamma[i_Gamma]);

        // std::cout << "Gamma[i_Gamma]: " << Gamma[i_Gamma] << std::endl;
        // std::cout << "back of oGamma: " << oGamma.back() << std::endl;

        // added.push_back(false);
        i_Gamma++;
      }
      else
      {
        // (2) there is no corresponding tensor Gamma element:
        SingleShellTensor Identity(shell, 0, 0, {}, {},
                                   {{0, 0, 0}, {0, 0, 0, 0}}, 1, 1);
        oGamma.push_back(Identity);
        // added.push_back(true);
      }
    }

    assert(oGamma.size() == gamma.size());

    // std::cout << "After unifying Gamma:" << std::endl;
    // for (const auto& G : oGamma) std::cout << G << std::endl;

    // (5) reconstruct tensor omega:

    size_t i_Omega = 0;
/*
    if (oGamma.size() > 1)
    {
      // if (added[0] && added[1])
      if (oGamma[0].identity() && oGamma[1].identity())
        oOmega.emplace_back(1, u3::irrep{0, 0, 0}, 1, u4::irrep{0, 0, 0, 0});
      // else if (added[0])
      else if (oGamma[0].identity())
        oOmega.emplace_back(1, oGamma[1].u3(), 1, oGamma[1].u4());
      // else if (added[1])
      else if (oGamma[1].identity())
        oOmega.emplace_back(1, oGamma[0].u3(), 1, oGamma[0].u4());
      else
        oOmega.push_back(Omega[i_Omega++]);

      for (i_Gamma = 2; i_Gamma < oGamma.size(); i_Gamma++)
      {
        // if (added[i_Gamma])
        if (oGamma[i_Gamma].identity())
        {
          auto Oe = oOmega.back();
          oOmega.emplace_back(1, Oe.u3(), 1, Oe.u4());
        }
        else if (i_Omega == 0)
        {
          // if (added[0] && added[1])
          if (oGamma[0].identity() && oGamma[1].identity())
            oOmega.emplace_back(1, oGamma[i_Gamma].u3(), 1, oGamma[i_Gamma].u4());
          else
          {
            oOmega.emplace_back(Omega[i_Omega]);
            i_Omega++;
          }
        }
        else
        {
          // std::cout << i_Omega << std::endl;
          oOmega.push_back(Omega[i_Omega++]);
        }
      }
    }
*/

    if (oGamma.size() > 1)
    {
      size_t count_preceding_identities = 0;
      for (const auto& g : oGamma)
      {
        if (g.identity())
          count_preceding_identities++;
        else
          break;
      }

      if (count_preceding_identities >= 2)
      {
        for (size_t cnt = 0; cnt < count_preceding_identities - 1; cnt++)
          oOmega.emplace_back(1, u3::irrep{0, 0, 0}, 1, u4::irrep{0, 0, 0, 0});

        oOmega.emplace_back(1, oGamma[count_preceding_identities].u3(), 
                            1, oGamma[count_preceding_identities].u4());

        for (i_Gamma = count_preceding_identities + 1; 
             i_Gamma < oGamma.size(); i_Gamma++)
        {
          if (oGamma[i_Gamma].identity())
            oOmega.emplace_back(1, oOmega.back().u3(), 1, oOmega.back().u4());
          else
            oOmega.push_back(Omega[i_Omega++]);
        }
      }
      else
      {
        if (oGamma[0].identity())
          oOmega.emplace_back(1, oGamma[1].u3(), 1, oGamma[1].u4());
        else if (oGamma[1].identity())
          oOmega.emplace_back(1, oGamma[0].u3(), 1, oGamma[0].u4());
        else
          oOmega.push_back(Omega[i_Omega++]);

        for (i_Gamma = 2; i_Gamma < oGamma.size(); i_Gamma++)
        {
          if (oGamma[i_Gamma].identity())
            oOmega.emplace_back(1, oOmega.back().u3(), 1, oOmega.back().u4());
          else
            oOmega.push_back(Omega[i_Omega++]);
        }
      }
    }

    assert(oOmega.size() + 1 == oGamma.size());

    return std::optional<MultiShellTensor>(std::in_place, oGamma, oOmega);
  }

  std::optional<MultiShellTensor_ShareGammas_phys> 
    unify_tensor(const MultiShellTensor_ShareGammas_phys& T, 
                 const ManyNucleonBasis::Irrep& bra,
                 const ManyNucleonBasis::Irrep& ket)
  {
    // (1) setup:

    // tensor
    const auto& Gamma = T.Gammas();
    const auto& Omega_LLSSTTstrengths = T.Omegas_LLSSTTstrengths();

    // bra
    const auto& b_gamma = bra.gamma;
    const auto& b_omega = bra.omega;

    assert(b_omega.size() + 1 == b_gamma.size());

    // ket
    const auto& k_gamma = ket.gamma;
    const auto& k_omega = ket.omega;

    assert(k_omega.size() + 1 == k_gamma.size());
    assert(k_gamma.size() == b_gamma.size());

    // (2) check if unification is possible:

    if (Gamma.size() > b_gamma.size()) return {};

    auto tensor_shells = extract_shells_from_gamma(Gamma);
    auto irrep_shells = extract_shells_from_gamma(b_gamma); 
    // same for bra & ket because they are unified

    for (auto shell : tensor_shells)
      if (std::ranges::find(irrep_shells, shell) == irrep_shells.end())
        return {};

    // (3) setup continued:

    U3xU4::SingleShellTensors oGamma;
    oGamma.reserve(b_gamma.size());

    // strengths are shared_ptr-backed (see MultiShellTensor_ShareGammas_phys):
    // this builds each surviving branch by MOVING/copying the source branch's
    // strengths pointer -- a refcount bump, not a double-vector copy.
    using labels_type = std::tuple<uint8_t,              // kappamax
                                   HO::qn_t,             // LL
                                   uint8_t,              // zetamax
                                   HO::qn_t,             // SS
                                   HO::qn_t,             // TT
                                   std::shared_ptr<std::vector<double>>>; // strengths

    std::vector<std::pair<U3xU4::OmegaElements,
                          std::vector<labels_type>>> oOmega_LLSSTTstrengths;
    oOmega_LLSSTTstrengths.reserve(Omega_LLSSTTstrengths.size());

    // (4) unify tensor gamma:

    size_t i_Gamma = 0;

    for (size_t i_gamma = 0; i_gamma < b_gamma.size(); i_gamma++)
    {
      auto shell = b_gamma[i_gamma].shell();

      if ((i_Gamma < Gamma.size()) && (Gamma[i_Gamma].shell() == shell))
      {
        // (a) there is a corresponding tensor Gamma element:

        // if RME doesn't exist on single-shell tensor level, nothing to do
        if (Gamma[i_Gamma].RME_exist(b_gamma[i_gamma], k_gamma[i_gamma]) == false) 
          return {};

        oGamma.push_back(Gamma[i_Gamma]);

        // std::cout << "Gamma[i_Gamma]: " << Gamma[i_Gamma] << std::endl;
        // std::cout << "back of oGamma: " << oGamma.back() << std::endl;

        i_Gamma++;
      }
      else
      {
        // (b) there is no corresponding tensor Gamma element:

        // identity only connect identical irreps
        if (b_gamma[i_gamma] != k_gamma[i_gamma])
          return {};

        SingleShellTensor Identity(shell, 0, 0, {}, {},
                                   {{0, 0, 0}, {0, 0, 0, 0}}, 1, 1);
        oGamma.push_back(Identity);
      }
    }

    if (oGamma.size() == 0) return {};
    if (oGamma.size() == 1) return T; // nothing has changed
    assert(oGamma.size() == b_gamma.size());

    // (5) reconstruct tensor omega:

    // std::cout << "After unifying Gamma:" << std::endl;
    size_t count_preceding_identities = 0;
    for (const auto& G : oGamma)
    {
      // std::cout << G << std::endl;
      if (G.identity()) count_preceding_identities++;
      else break;
    }

    for (const auto& [Omega, LLSSTTstrengths] : Omega_LLSSTTstrengths)
    {
      assert(Omega.size() + 1 == Gamma.size());

      U3xU4::OmegaElements oOmega;
      oOmega.reserve(b_omega.size());

      size_t i_Omega = 0;

      if (count_preceding_identities >= 2)
      {
        for (size_t cnt = 0; cnt < count_preceding_identities - 1; cnt++)
          oOmega.emplace_back(1, u3::irrep{0, 0, 0}, 1, u4::irrep{0, 0, 0, 0});

        oOmega.emplace_back(1, oGamma[count_preceding_identities].u3(), 
                            1, oGamma[count_preceding_identities].u4());

        for (i_Gamma = count_preceding_identities + 1; 
             i_Gamma < oGamma.size(); i_Gamma++)
        {
          if (oGamma[i_Gamma].identity())
            oOmega.emplace_back(1, oOmega.back().u3(), 1, oOmega.back().u4());
          else
            oOmega.push_back(Omega[i_Omega++]);
        }
      }
      else
      {
        if (oGamma[0].identity())
          oOmega.emplace_back(1, oGamma[1].u3(), 1, oGamma[1].u4());
        else if (oGamma[1].identity())
          oOmega.emplace_back(1, oGamma[0].u3(), 1, oGamma[0].u4());
        else
          oOmega.push_back(Omega[i_Omega++]);

        for (i_Gamma = 2; i_Gamma < oGamma.size(); i_Gamma++)
        {
          if (oGamma[i_Gamma].identity())
              oOmega.emplace_back(1, oOmega.back().u3(), 1, oOmega.back().u4());
          else
            oOmega.push_back(Omega[i_Omega++]);
        }
      }

      assert(oOmega.size() + 1 == oGamma.size());

      bool couple = true;
      for (size_t i = 0; i < oOmega.size(); i++)
      {
        if (u3::Rho(k_omega[i].u3(), oOmega[i].u3(), b_omega[i].u3()) == 0 ||
            u4::Eta(k_omega[i].u4(), oOmega[i].u4(), b_omega[i].u4()) == 0)
        {
          couple = false;
          break;
        }
      }
      if (couple)
        oOmega_LLSSTTstrengths.push_back({std::move(oOmega), LLSSTTstrengths});
    }

    if (oOmega_LLSSTTstrengths.size() == 0) return {};

    return std::optional<MultiShellTensor_ShareGammas_phys>(std::in_place,
                std::move(oGamma), std::move(oOmega_LLSSTTstrengths),
                T.MMT(), T.JJ(), T.MMJ());
  }
} // namespace U3xU4
