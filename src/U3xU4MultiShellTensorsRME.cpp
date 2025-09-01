#include <numeric>

#include <qcx_convert.h>
#include <qcx_ND.h>

#include <RME.h>
#include <U3xU4irrep.h>
#include <U3xU4MultiShellTensors.h>

// #define FRONT_CREATE
#define REAR_CREATE

namespace U3xU4
{
  // Adaptation from lsu3shell
  int8_t
  MultiShellTensor::uncoupling_phase(const ManyNucleonBasis::Irrep& bra,
                                     const ManyNucleonBasis::Irrep& ket) const
  {
    if (bra.omega.empty())
      return 1;

    int G1 = Gamma[0].ads + Gamma[0].ats, G2 = Gamma[1].ads + Gamma[1].ats,
        g1_ket = ket.gamma[0].particles(), g2_ket = ket.gamma[1].particles(),
        g1_bra = bra.gamma[0].particles(), g2_bra = bra.gamma[1].particles();

#ifdef FRONT_CREATE
    int phi = g1_ket * G2 + g2_bra * (G1 + g1_ket + g1_bra);
    int8_t phase = (phi % 2 == 0) ? 1 : -1;
#endif
#ifdef REAR_CREATE
    int phi = g2_bra * G1 + g1_ket * (G2 + g2_ket + g2_bra);
    int8_t phase = (phi % 2 == 0) ? 1 : -1;
#endif

    for (size_t i = 2; i < bra.gamma.size(); i++)
    {
      g1_bra += g2_bra;
      G1 += G2;
      g1_ket += g2_ket;

      g2_bra = bra.gamma[i].particles();
      G2 = Gamma[i].ads + Gamma[i].ats;
      g2_ket = ket.gamma[i].particles();

#ifdef FRONT_CREATE
      int phi = g1_ket * G2 + g2_bra * (G1 + g1_ket + g1_bra);
      phase *= (phi % 2 == 0) ? 1 : -1;
#endif
#ifdef REAR_CREATE
      int phi = g2_bra * G1 + g1_ket * (G2 + g2_ket + g2_bra);
      phase *= (phi % 2 == 0) ? 1 : -1;
#endif
    }

    assert((phase == 1) || (phase == -1));

    return phase;
  }

  sam::RME MultiShellTensor::RME(const U3xU4::ManyNucleonBasis::Irrep& bra,
                                 const U3xU4::ManyNucleonBasis::Irrep& ket,
                                 sam::Databases& dbs) const
  {
    // (1) everything must already be unified:

    assert(bra.gamma.size() == ket.gamma.size());
    assert(Gamma.size() == ket.gamma.size());

    assert(bra.omega.size() == ket.omega.size());
    assert(Omega.size() == bra.omega.size());

    assert(ket.gamma.size() == ket.omega.size() + 1);

    // (2) calculate phase:

    auto phase = uncoupling_phase(bra, ket);

    // (3) first single-shell RMEs:

    const U3xU4::SingleShellTensor& T = Gamma[0];
    const auto& f = bra.gamma[0];
    const auto& i = ket.gamma[0];

    assert(T.shell() == f.shell());
    assert(T.shell() == i.shell());

    std::size_t rhomaxA, etamaxA; // multi for Wigner-Eckart theorem
    auto RMEsA = T.RME(f, i, rhomaxA, etamaxA, dbs);

    if (RMEsA.empty())
      return {};

    // (4) multi-shell RMEs:

    for (size_t k = 1; k < ket.gamma.size(); k++)
    {
      const U3xU4::SingleShellTensor& T = Gamma[k];
      const auto& f = bra.gamma[k];
      const auto& i = ket.gamma[k];

      assert(T.shell() == f.shell());
      assert(T.shell() == i.shell());

      std::size_t rhomaxB, etamaxB; // multi for Wigner-Eckart theorem
      auto RMEsB = T.RME(f, i, rhomaxB, etamaxB, dbs);

      if (RMEsB.empty())
        return {};

      const U3xU4::irrep& u3u4_11 =
          (k > 1) ? static_cast<const U3xU4::irrep&>(ket.omega[k - 2])
                  : static_cast<const U3xU4::irrep&>(ket.gamma[0]);
      const U3xU4::irrep& u3u4_12 =
          (k > 1) ? static_cast<const U3xU4::irrep&>(Omega[k - 2])
                  : static_cast<const U3xU4::irrep&>(Gamma[0]);
      const U3xU4::irrep& u3u4_13 =
          (k > 1) ? static_cast<const U3xU4::irrep&>(bra.omega[k - 2])
                  : static_cast<const U3xU4::irrep&>(bra.gamma[0]);

      const U3xU4::irrep& u3u4_21 = i;
      const U3xU4::irrep& u3u4_22 = T;
      const U3xU4::irrep& u3u4_23 = f;

      const U3xU4::irrep& u3u4_31 = ket.omega[k - 1];
      const U3xU4::irrep& u3u4_32 = Omega[k - 1];
      const U3xU4::irrep& u3u4_33 = bra.omega[k - 1];

      size_t rhomax = u3::Rho(u3u4_31.u3(), u3u4_32.u3(), u3u4_33.u3());
      size_t etamax = u4::Eta(u3u4_31.u4(), u3u4_32.u4(), u3u4_33.u4());

      // 9-U(3) recoupling coefficients
      auto U3U9 = dbs.u3u9db.get_U3U9(
          u3u4_11.u3(), u3u4_12.u3(), u3u4_21.u3(), u3u4_22.u3(), u3u4_13.u3(),
          u3u4_23.u3(), u3u4_31.u3(), u3u4_32.u3(), u3u4_33.u3(),
          ket.omega[k - 1].rhomax(), Omega[k - 1].rhomax(),
          bra.omega[k - 1].rhomax(), RMEsA.rhomax(), RMEsB.rhomax(), rhomax);

      if (U3U9.empty())
        return {};

      // 9-U(4) recoupling coefficients
      auto U4U9 = dbs.u4u9db.get_U4U9(
          u3u4_11.u4(), u3u4_12.u4(), u3u4_21.u4(), u3u4_22.u4(), u3u4_13.u4(),
          u3u4_23.u4(), u3u4_31.u4(), u3u4_32.u4(), u3u4_33.u4(),
          ket.omega[k - 1].etamax(), Omega[k - 1].etamax(),
          bra.omega[k - 1].etamax(), RMEsA.etamax(), RMEsB.etamax(), etamax);

      if (U4U9.empty())
        return {};

      // combine RMEsA, RMEsB, U3U9coeff, and U4U9coeff into RMEs:

      // multiplicities for resulting RMEs:
      qcx_utils::NDIndex ndi_res;

      ndi_res.add_dimensions(
          RMEsA.afmax(), RMEsB.afmax(), bra.omega[k - 1].rhomax(),
          bra.omega[k - 1].etamax(), RMEsA.aimax(), RMEsB.aimax(),
          ket.omega[k - 1].rhomax(), ket.omega[k - 1].etamax(), RMEsA.a0max(),
          RMEsB.a0max(), Omega[k - 1].rhomax(), Omega[k - 1].etamax(), rhomax,
          etamax);

      size_t n = ndi_res.cardinality();

      std::vector<double> rmes(n);

      for (size_t k = 0; k < n; k++)
      {
        std::array<size_t, 14> multi_res;
        ndi_res.to_multi(k, multi_res);

        for (size_t rhoA = 0; rhoA < RMEsA.rhomax(); rhoA++)
          for (size_t rhoB = 0; rhoB < RMEsB.rhomax(); rhoB++)
          {
            auto u3u9 = U3U9.get(multi_res[6], multi_res[10], multi_res[2],
                                 rhoA, rhoB, multi_res[12]);

            for (size_t etaA = 0; etaA < RMEsA.etamax(); etaA++)
              for (size_t etaB = 0; etaB < RMEsB.etamax(); etaB++)
              {
                auto u4u9 = U4U9.get(multi_res[7], multi_res[11], multi_res[3],
                                     etaA, etaB, multi_res[12]);

                auto rmeA = RMEsA.get(multi_res[0], multi_res[4], multi_res[8],
                                      rhoA, etaA);
                auto rmeB = RMEsB.get(multi_res[1], multi_res[5], multi_res[9],
                                      rhoB, etaB);

                rmes[k] += u3u9 * u4u9 * rmeA * rmeB;
              }
          }
      }

      size_t af_max = ndi_res.dimension(0) * ndi_res.dimension(1) *
                      ndi_res.dimension(2) * ndi_res.dimension(3);
      size_t ai_max = ndi_res.dimension(4) * ndi_res.dimension(5) *
                      ndi_res.dimension(6) * ndi_res.dimension(7);
      size_t a0_max = ndi_res.dimension(8) * ndi_res.dimension(9) *
                      ndi_res.dimension(10) * ndi_res.dimension(11);

      RMEsA = {std::move(rmes), af_max, ai_max, a0_max, rhomax, etamax};
    }

    RMEsA.phase(phase);

    return RMEsA;
  }

} // namespace U3xU4
