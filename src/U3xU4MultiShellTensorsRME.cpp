#include <numeric>

#include <qcx_convert.h>
#include <qcx_ND.h>

#include <RME.h>
#include <U3xU4irrep.h>
#include <U3xU4MultiShellTensors.h>

#define FRONT_CREATE
//#define REAR_CREATE

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

  // helper function to check RMEs multiplicities:
  template <typename G, typename O>
  static size_t alphamax(const G& gamma, const O& omega)
  {
    assert(gamma.size() == omega.size() + 1);

    size_t res = 1;

    for (const auto& ge : gamma)
      if (ge.state() == U3xU4::GammaElement::State::Basis)
        res *= ge.alphamax();
      else if (ge.state() == U3xU4::GammaElement::State::Tensor)
        res *= ge.rhomax() * ge.etamax();

    for (const auto& oe : omega)
      res *= oe.rhomax() * oe.etamax();

    return res;
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

    // (3) calculate RMEs:

    sam::RME RMEsL; // left part

    for (size_t k = 0; k < ket.gamma.size(); k++)
    {
      // (I) next single-shell RMEs:

      const U3xU4::SingleShellTensor& T = Gamma[k];
      const auto& f = bra.gamma[k];
      const auto& i = ket.gamma[k];

      // check unified shells:
      assert(T.shell() == f.shell());
      assert(T.shell() == i.shell());

      //std::size_t rhomax_temp, etamax_temp; // multi for Wigner-Eckart theorem
      //auto RMEsR = T.RME(f, i, rhomax_temp, etamax_temp, dbs); // right part
      auto RMEsR = T.RME(f, i, dbs); // right part

      if (RMEsR.empty()) return {};

      // nothing more to do in first iteration:
      if (k == 0)
      {
        RMEsL = std::move(RMEsR);
        continue;
      }

      // (II) 9-recoupling coefficients:

      const U3xU4::irrep& A =
          (k > 1) ? ket.omega[k - 2].u3u4() : ket.gamma[0].u3u4();
      const U3xU4::irrep& B = (k > 1) ? Omega[k - 2].u3u4() : Gamma[0].u3u4();
      const U3xU4::irrep& AB =
          (k > 1) ? bra.omega[k - 2].u3u4() : bra.gamma[0].u3u4();

      const U3xU4::irrep& C = i;
      const U3xU4::irrep& D = T;
      const U3xU4::irrep& CD = f;

      const U3xU4::irrep& AC = ket.omega[k - 1];
      const U3xU4::irrep& BD = Omega[k - 1];
      const U3xU4::irrep& ABCD = bra.omega[k - 1];

      // 9-U(3) recoupling coefficients:

      auto rhoACmax = ket.omega[k - 1].rhomax();
      auto rhoBDmax = Omega[k - 1].rhomax();
      auto rhoABCDmax = bra.omega[k - 1].rhomax();
      auto rhoABmax = RMEsL.rhomax();
      auto rhoCDmax = RMEsR.rhomax();
      auto rhoACBDmax = u3::Rho(AC.u3(), BD.u3(), ABCD.u3());

      /*
      if (rhoACmax == 0 || rhoBDmax == 0 || rhoABCDmax == 0 ||
          rhoABmax == 0 || rhoCDmax == 0 || rhoACBDmax == 0)
        return {};
      */

      auto U3U9 =
          dbs.u3u9db.get_U3U9(A.u3(), B.u3(), C.u3(), D.u3(), AB.u3(), CD.u3(),
                              AC.u3(), BD.u3(), ABCD.u3(), rhoACmax, rhoBDmax,
                              rhoABCDmax, rhoABmax, rhoCDmax, rhoACBDmax);

      if (U3U9.empty())
        return {};

      // 9-U(4) recoupling coefficients:

      auto etaACmax = ket.omega[k - 1].etamax();
      auto etaBDmax = Omega[k - 1].etamax();
      auto etaABCDmax = bra.omega[k - 1].etamax();
      auto etaABmax = RMEsL.etamax();
      auto etaCDmax = RMEsR.etamax();
      auto etaACBDmax = u4::Eta(AC.u4(), BD.u4(), ABCD.u4());

      /*
      if (etaACmax == 0 || etaBDmax == 0 || etaABCDmax == 0 ||
          etaABmax == 0 || etaCDmax == 0 || etaACBDmax == 0)
        return {};
      */

      auto U4U9 =
          dbs.u4u9db.get_U4U9(A.u4(), B.u4(), C.u4(), D.u4(), AB.u4(), CD.u4(),
                              AC.u4(), BD.u4(), ABCD.u4(), etaACmax, etaBDmax,
                              etaABCDmax, etaABmax, etaCDmax, etaACBDmax);

      if (U4U9.empty())
        return {};

      // (III) combine RMEsL, RMEsR, U3U9coeff, and U4U9coeff into RMEs:

      // multiplicities for calculated RMEs:
      qcx_utils::NDIndex14 ndi_res;
      //std::cout << "Start adding dimensions" << std::endl;
      ndi_res.add_dimensions(RMEsL.afmax(), RMEsR.afmax(), rhoABCDmax,
                             etaABCDmax, RMEsL.aimax(), RMEsR.aimax(), rhoACmax,
                             etaACmax, RMEsL.a0max(), RMEsR.a0max(), rhoBDmax,
                             etaBDmax, rhoACBDmax, etaACBDmax);
      //std::cout << "Finish adding dimensions" << std::endl;
      size_t n = ndi_res.cardinality();

      std::vector<double> rmes(n);

      for (size_t j = 0; j < n; j++)
      {
        std::array<size_t, 14> multi_res;
        ndi_res.to_multi(j, multi_res);

        auto rhoAC = multi_res[6];
        auto rhoBD = multi_res[10];
        auto rhoABCD = multi_res[2];
        auto rhoACBD = multi_res[12];

        auto etaAC = multi_res[7];
        auto etaBD = multi_res[11];
        auto etaABCD = multi_res[3];
        auto etaACBD = multi_res[13];

        auto aLf = multi_res[0];
        auto aLi = multi_res[4];
        auto aL0 = multi_res[8];

        auto aRf = multi_res[1];
        auto aRi = multi_res[5];
        auto aR0 = multi_res[9];

        for (size_t rhoAB = 0; rhoAB < rhoABmax; rhoAB++)
          for (size_t rhoCD = 0; rhoCD < rhoCDmax; rhoCD++)
          {
            auto u3u9 = U3U9.get(rhoAC, rhoBD, rhoABCD, rhoAB, rhoCD, rhoACBD);

            for (size_t etaAB = 0; etaAB < etaABmax; etaAB++)
              for (size_t etaCD = 0; etaCD < etaCDmax; etaCD++)
              {
                auto u4u9 = U4U9.get(etaAC, etaBD, etaABCD, etaAB, etaCD, etaACBD);
                auto rmeL = RMEsL.get(aLf, aLi, aL0, rhoAB, etaAB);
                auto rmeR = RMEsR.get(aRf, aRi, aR0, rhoCD, etaCD);

                rmes[j] += u3u9 * u4u9 * rmeL * rmeR;
              }
          }
      }

      // resolve multiplicities of resulting RMEs:

      size_t af_max = ndi_res.dimension(0) * ndi_res.dimension(1) *
                      ndi_res.dimension(2) * ndi_res.dimension(3);
      size_t ai_max = ndi_res.dimension(4) * ndi_res.dimension(5) *
                      ndi_res.dimension(6) * ndi_res.dimension(7);
      size_t a0_max = ndi_res.dimension(8) * ndi_res.dimension(9) *
                      ndi_res.dimension(10) * ndi_res.dimension(11);

      RMEsL = {std::move(rmes),
               af_max,
               ai_max,
               a0_max,
               static_cast<size_t>(rhoACBDmax),
               static_cast<size_t>(etaACBDmax)};
    }

    // check multiplicities:
    assert(RMEsL.afmax() == alphamax(bra.gamma, bra.omega));
    assert(RMEsL.aimax() == alphamax(ket.gamma, ket.omega));
    assert(RMEsL.a0max() == alphamax(Gamma, Omega));

    // apply phase factor:
    RMEsL.phase(phase);

    return RMEsL;
  }

} // namespace U3xU4
