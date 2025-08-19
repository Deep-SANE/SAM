#include <cstddef>
#include <vector>

#include <U3xU4Irrep.h>
#include <U3xU4MultiShellTensors.h>
#include <U4xSU3ManyNucleonBasis.h>

namespace U3xU4
{
  MultiShellTensor unify_tensor(const MultiShellTensor& T,
                                const U4xSU3::Irrep& irrep)
  {
    const auto& Gamma = T.single_shell_tensors;
    const auto& Omega = T.coupled_irreps;

    assert(Omega.size() + 1 == Gamma.size());

    const auto& gamma = irrep.gamma;
    const auto& omega = irrep.omega;

    assert(omega.size() + 1 == gamma.size());

    static const MultiShellTensor::GammaElement Identity(
        0, 0, 0, {}, {}, {{0, 0, 0}, {0, 0, 0, 0}}, 1, 1);

    std::vector<MultiShellTensor::GammaElement> oGamma;
    std::vector<MultiShellTensor::OmegaElement> oOmega;

    oGamma.reserve(gamma.size());
    oOmega.reserve(omega.size());

    std::vector<uint8_t> added;
    added.reserve(gamma.size());

    size_t i_Gamma = 0;

    for (size_t i_gamma = 0; i_gamma < gamma.size(); i_gamma++)
    {
      auto shell = irrep.gamma_shell(i_gamma);

      if ((i_Gamma < Gamma.size()) && (Gamma[i_Gamma].n == shell))
      {
        // (1) there is a corresponding Gamma element:
        oGamma.push_back(Gamma[i_Gamma]);
        added.push_back(false);
        i_Gamma++;
      }
      else
      {
        // (2) there is no corresponding Gamma element:
        oGamma.push_back(Identity);
        oGamma.back().n = shell;
        added.push_back(true);
      }
    }

    assert(oGamma.size() == gamma.size());

    size_t i_Omega = 0;

    if (oGamma.size() > 1)
    {
      if (added[0] && added[1])
        oOmega.emplace_back(U3xU4::irrep({0, 0, 0}, {0, 0, 0, 0}), 1, 1);
      else if (added[0])
        oOmega.push_back({{oGamma[1].T.u3, oGamma[1].T.u4}, 1, 1});
      else if (added[1])
        oOmega.push_back({{oGamma[0].T.u3, oGamma[0].T.u4}, 1, 1});
      else
        oOmega.push_back(Omega[i_Omega++]);

      for (i_Gamma = 2; i_Gamma < oGamma.size(); i_Gamma++)
        if (added[i_Gamma])
        {
          auto Oe = oOmega.back();
          std::get<1>(Oe) = 1; // rho0max
          std::get<2>(Oe) = 1; // eta0max
          oOmega.push_back(Oe);
        }
        else if (i_Omega == 0)
          oOmega.push_back(
              {{oGamma[i_Gamma].T.u3, oGamma[i_Gamma].T.u4}, 1, 1});
        else
          oOmega.push_back(Omega[i_Omega++]);
    }

    assert(oOmega.size() + 1 == oGamma.size());

    return {oGamma, oOmega};
  }

} // namespace U3xU4
