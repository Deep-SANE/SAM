#include <algorithm>
#include <cstddef>
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
    // (1) setup:

    const auto& Gamma = T.Gamma;
    const auto& Omega = T.Omega;

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

    std::vector<U3xU4::SingleShellTensor> oGamma;
    std::vector<U3xU4::OmegaElement> oOmega;

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

    // (5) reconstruct tensor omega:

    size_t i_Omega = 0;

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

    assert(oOmega.size() + 1 == oGamma.size());

    return std::optional<MultiShellTensor>(std::in_place, oGamma, oOmega);
  }

} // namespace U3xU4
