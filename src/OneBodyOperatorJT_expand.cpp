#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>

#include <qcx_convert.h>
#include <qcx_readwrite.h>
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>
#include <so3Coeffs/SO3_Fano.h>

#include <U3SO3xU3SO3.h>
#include <U4SU2SU2xU4SU2SU2.h>

#include <GlobalConstants.h>
#include <HOutilities.h>
#include <OneBodyOperatorJT.h>

namespace JT
{
  std::pair<bool, std::vector<double>>
  OneBodyOperator::getTensorStrength_ME(int lam0, int mu0, int P10, int P20,
                                     int P30, int kappa0, int LL0, int SS0,
                                     int na, int nb) const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    std::vector<double> x(kappa0, 0.0);

    // loop over la
    for (int lla = 2 * na; lla >= 2 * (na % 2); lla -= 4)
    {
      // loop over jja
      for (int jja = lla + 1; jja >= std::fabs(lla - 1); jja -= 2)
      {
        // loop over lb
        for (int llb = 2 * nb; llb >= 2 * (nb % 2); llb -= 4)
        {
          // check if lla x llb -> LL0 coupling is allowed
          if (LL0 < std::fabs(lla - llb) || LL0 > lla + llb)
            continue;

          // loop over jjb
          for (int jjb = llb + 1; jjb >= std::fabs(llb - 1); jjb -= 2)
          {
            // check if jja x jjb -> JJ coupling is allowed
            if (JJ() < std::fabs(jja - jjb) || JJ() > jja + jjb)
              continue;

            double rme;

            double U9j;
            if (na <= nb)
              U9j = racah9.get({lla, 1, jja, llb, 1, jjb, LL0, SS0, JJ()});
            else // na > nb
              U9j = racah9.get({llb, 1, jjb, lla, 1, jja, LL0, SS0, JJ()});

            std::vector<double> Coeffs3, Coeffs4;
            if (na <= nb)
            {
              Coeffs3 = wigU3SO3db.get(
                na, 0, lla / 2, 0, nb, llb / 2, lam0, mu0, LL0 / 2);
              Coeffs4 = wigU4SU2SU2db.get(
                1, 0, 0, 1, 1, 0, 0, 1, 1, 1, P10, P20, P30, SS0, TT());
            }
            else // na > nb
            {
              Coeffs3 = wigU3SO3db.get(
                0, nb, llb / 2, na, 0, lla / 2, lam0, mu0, LL0 / 2);
              Coeffs4 = wigU4SU2SU2db.get(
                0, 0, 1, 1, 1, 1, 0, 0, 1, 1, P10, P20, P30, SS0, TT());
            }

            // loop over projections
            for (int mmja = jja; mmja >= -jja; mmja -= 2)
            {
              int mmjb = mmja - MMJ(); // selection rule of sphericalME

              for (int mmta = 1; mmta >= -1; mmta -= 2)
              {
                int mmtb = mmta - MMT(); // selection rule of sphericalME

                double me = sphericalME(na, lla, jja, mmja, mmta, nb, llb,
                                          jjb, mmjb, mmtb);
                if (me == 0.0) continue;

                // phase = (-)^{(nb + lb + 1)/2 - jb + mjb + mtb}
#ifdef ADD_SU3LIB_PHASE_U3SO3
                double phase =
                    ((2 * nb + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0)
                          ? 1.0
                          : -1.0;
#else
                double phase =
                    ((nb + llb / 2 + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0)
                          ? 1.0
                          : -1.0;
#endif
                if (na <= nb)
                {
                  double cgcJ = cgcs.get({jja, jjb, JJ(), mmja, -mmjb, MMJ()}),
                         cgcT = cgcs.get({1, 1, TT(), mmta, -mmtb, MMT()});

                  for (int k = 0; k < kappa0; k++)
                  {
                    double y = phase * cgcJ * cgcT * U9j
                             * Coeffs3[k] * Coeffs4[0] * me;
                    // std::cout << y << std::endl;
                    x[k] += y;
                  }
                }
                else // na > nb
                {
                  double cgcJ = cgcs.get({jjb, jja, JJ(), -mmjb, mmja, MMJ()}),
                         cgcT = cgcs.get({1, 1, TT(), -mmtb, mmta, MMT()});

                  for (int k = 0; k < kappa0; k++)
                  {
                    double y = phase * cgcJ * cgcT * U9j
                             * Coeffs3[k] * Coeffs4[0] * me;
                    // std::cout << y << std::endl;
                    x[k] -= y;
                  }
                }
              }
            }
          }
        }
      }
    }
    // std::cout << "Finish getTensorStrength\n";
    double sum = 0.0;
    for (const auto& X : x) sum += std::fabs(X);
    
    return std::make_pair((std::fabs(sum) < 1e-8), x);
  }

  // Using sphericalRME to avoid loops over projection mj and mt
  std::pair<bool, std::vector<double>>
  OneBodyOperator::getTensorStrength_RME(int lam0, int mu0, int P10, int P20,
                                     int P30, int kappa0, int LL0, int SS0,
                                     int na, int nb) const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    std::vector<double> x(kappa0, 0.0);

    // loop over la
    for (int lla = 2 * na; lla >= 2 * (na % 2); lla -= 4)
    {
      // loop over jja
      for (int jja = lla + 1; jja >= std::fabs(lla - 1); jja -= 2)
      {
        // loop over lb
        for (int llb = 2 * nb; llb >= 2 * (nb % 2); llb -= 4)
        {
          // check if lla x llb -> LL0 coupling is allowed
          if (LL0 < std::fabs(lla - llb) || LL0 > lla + llb)
            continue;

          // loop over jjb
          for (int jjb = llb + 1; jjb >= std::fabs(llb - 1); jjb -= 2)
          {
            // check if jja x jjb -> JJ coupling is allowed
            if (JJ() < std::fabs(jja - jjb) || JJ() > jja + jjb)
              continue;

            double rme = sphericalRME(na, lla, jja, nb, llb, jjb);
            if (rme == 0.0) continue;

            double U9j;
            if (na <= nb)
              U9j = racah9.get({lla, 1, jja, llb, 1, jjb, LL0, SS0, JJ()});
            else // na > nb
              U9j = racah9.get({llb, 1, jjb, lla, 1, jja, LL0, SS0, JJ()});

            std::vector<double> Coeffs3, Coeffs4;
            if (na <= nb)
            {
              Coeffs3 = wigU3SO3db.get(
                na, 0, lla / 2, 0, nb, llb / 2, lam0, mu0, LL0 / 2);
              Coeffs4 = wigU4SU2SU2db.get(
                1, 0, 0, 1, 1, 0, 0, 1, 1, 1, P10, P20, P30, SS0, TT());

#ifdef ADD_SU3LIB_PHASE_U3SO3
              double phase = ((nb + 1) % 2 == 0) ? 1.0 : -1.0;
#else
              double phase = ((nb + llb / 2 + 2) / 2 % 2 == 0) ? 1.0 : -1.0;
#endif
              double factor = std::sqrt(2.0 * (jja + 1.0) / (JJ() + 1.0) /
                                          (TT() + 1.0));
              for (int k = 0; k < kappa0; k++)
              {
                double y = phase * U9j * Coeffs3[k] * Coeffs4[0] * rme *
                           factor;
                // std::cout << y << std::endl;
                x[k] += y;
              }
            }
            else // na > nb
            {
              Coeffs3 = wigU3SO3db.get(
                0, nb, llb / 2, na, 0, lla / 2, lam0, mu0, LL0 / 2);
              Coeffs4 = wigU4SU2SU2db.get(
                0, 0, 1, 1, 1, 1, 0, 0, 1, 1, P10, P20, P30, SS0, TT());

#ifdef ADD_SU3LIB_PHASE_U3SO3
              double phase = ((2* nb + jja + jjb - JJ() - TT()) / 2 % 2 == 0)
                               ? 1.0
                               : -1.0;
#else
              double phase =
                           ((nb + llb / 2 + jja + jjb - JJ() - TT()) / 2 % 2 == 0)
                               ? 1.0
                               : -1.0;
#endif
              double factor = std::sqrt(2.0 * (jja + 1.0) / (JJ() + 1.0) /
                                          (TT() + 1.0));
              for (int k = 0; k < kappa0; k++)
              {
                double y = phase * U9j * Coeffs3[k] * Coeffs4[0] * rme *
                           factor;
                // std::cout << y << std::endl;
                x[k] -= y;
              }
            }
          }
        }
      }
    }
    // std::cout << "Finish getTensorStrength\n";
    double sum = 0.0;
    for (const auto& X : x)
      sum += std::fabs(X);
    
    return std::make_pair((std::fabs(sum) < 1e-8), x);
  }

  U3xU4::OneBodyMultiShellTensors
    OneBodyOperator::expand_old()
  {
    if (on_the_fly == false) OBME->read();

    U3xU4::OneBodyMultiShellTensors tensors;

    // loop for ad
    for (int na = 0; na <= maxShell_; na++)
    {
      u3::irrep u3a(na, 0, 0);
      u4::irrep u4a(1, 0, 0, 0);

      // loop for at
      for (int nb = 0; nb <= maxShell_; nb++)
      {
        u3::irrep u3b(nb, nb, 0);
        u4::irrep u4b(1, 1, 1, 0);

        // coupled irreps
        std::vector<u3::irrep> u3couple;
        std::vector<int> u3multi;

        std::vector<u4::irrep> u4couple;
        std::vector<int> u4multi;

        if (na <= nb)
        {
          std::tie(u3couple, u3multi) = u3::decomposition(u3a, u3b);
          std::tie(u4couple, u4multi) = u4::decomposition(u4a, u4b);
        }
        else // na > nb => switch the order
        {
          std::tie(u3couple, u3multi) = u3::decomposition(u3b, u3a);
          std::tie(u4couple, u4multi) = u4::decomposition(u4b, u4a);
        }

        for (int i = 0; i < u3couple.size(); i++)
        {
          assert(u3multi[i] == 1);

          // decompose into L0
          std::vector<u3::physicalvector> pv3s = u3::physicalBasis(u3couple[i]);

          for (int j = 0; j < u4couple.size(); j++)
          {
            assert(u4multi[j] == 1);

            // decompose into 2S0 and 2T0
            std::vector<u4::physicalvector> pv4s =
                u4::physicalBasis(u4couple[j]);

            for (int pv3 = 0; pv3 < pv3s.size(); pv3++)
            {
              int lam0 = pv3s[pv3].lam(), mu0 = pv3s[pv3].mu(),
                  kappa0 = pv3s[pv3].kappa(), LL0 = 2 * pv3s[pv3].L();

              for (int pv4 = 0; pv4 < pv4s.size(); pv4++)
              {
                int P10 = pv4s[pv4].P1(), P20 = pv4s[pv4].P2(),
                    P30 = pv4s[pv4].P3(), SS0 = pv4s[pv4].SS(),
                    TT0 = pv4s[pv4].TT();

                // if the operator has definite isospin assignment
                if (TT() >= 0 && TT() != TT0) continue;

                // couple LL0 and SS0 to JJ0
                if (JJ() < std::fabs(LL0 - SS0) || JJ() > (LL0 + SS0))
                  continue;

                // the selection rule of CGCs also dictates that
                // MMJ0 == MMJ and MMT0 == MMT

                // compute the expansion coefficients
                auto [ignore, coeff] = getTensorStrength_ME(
                    lam0, mu0, P10, P20, P30, kappa0, LL0, SS0, na,
                    nb // pv3as, pv3bs, pv4as, pv4bs,
                    );

                if (ignore == false)
                {
                  if (na <= nb)
                  {
                    U3xU4::OneBodyMultiShellTensor tensor(
                        U3xU4::irrep(u3a, u4a), U3xU4::irrep(u3b, u4b),
                        U3xU4::irrep(u3couple[i], u4couple[j]), kappa0, LL0,
                        SS0, TT_, MMT_, JJ_, MMJ_, coeff);
                    tensors.push_back(tensor);
                  }
                  else // na > nb
                  {
                    U3xU4::OneBodyMultiShellTensor tensor(
                        U3xU4::irrep(u3b, u4b), U3xU4::irrep(u3a, u4a),
                        U3xU4::irrep(u3couple[i], u4couple[j]), kappa0, LL0,
                        SS0, TT_, MMT_, JJ_, MMJ_, coeff);
                    tensors.push_back(tensor);
                  }
                }
              }
            }
          }
        }
      }
    }

    return tensors;
  }

  // Selection rules
  bool OneBodyOperator::LLselection(const int& LL) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" || name_ == "Rrel" ||
        name_ == "Radius" || name_ == "Radius_p" || name_ == "Radius_n" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic" || 
        name_ == "FermiPlus" || name_ == "FermiMinus" ||
        name_ == "GamowTellerPlus" || name_ == "GamowTellerMinus")
    {
      if (LL != 0) flag = false;
    }
    else if (name_ == "E2")
    {
      if (LL != 4) flag = false;
    }
    else if (name_ == "M1")
    {
      if (LL != 0 && LL != 2) flag = false;
    }
    
    return flag;
  }
  
  bool OneBodyOperator::SSselection(const int& SS) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" || name_ == "E2" ||
        name_ == "Rrel" || name_ == "Radius" || name_ == "Radius_p" ||
        name_ == "Radius_n" || name_ == "CoMNumberOp" || name_ == "Kinetic" || 
        name_ == "FermiPlus" || name_ == "FermiMinus")
    {
      if (SS != 0) flag = false;
    }
    else if (name_ == "M1")
    {
      if (SS != 0 && SS != 2) flag = false;
    }
    else if (name_ == "GamowTellerPlus" || name_ == "GamowTellerMinus")
    {
      if (SS != 2) flag = false;
    }
    
    return flag;
  }

  bool OneBodyOperator::U3selection(const u3::irrep& irr) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" || name_ == "CoMNumberOp" || 
        name_ == "FermiPlus" || name_ == "FermiMinus" ||
        name_ == "GamowTellerPlus" || name_ == "GamowTellerMinus")
    {
      if (irr != u3::irrep(0, 0, 0)) flag = false;
    }
    else if (name_ == "Rrel" || name_ == "Radius" || name_ == "Radius_p" ||
             name_ == "Radius_n" || name_ == "Kinetic")
    {
      if (irr != u3::irrep(0, 0, 0) &&
          irr != u3::irrep(2, 0, 0) &&
          irr != u3::irrep(2, 2, 0)) 
        flag = false;
    }
    else if (name_ == "E2")
    {
      if (irr != u3::irrep(2, 1, 0) &&
          irr != u3::irrep(2, 0, 0) &&
          irr != u3::irrep(2, 2, 0)) 
        flag = false;
    }
    else if (name_ == "M1")
    {
      if (irr != u3::irrep(2, 1, 0) && irr != u3::irrep(0, 0, 0)) 
        flag = false;
    }
    
    return flag;
  }

  bool OneBodyOperator::U4selection(const u4::irrep& irr) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" || name_ == "E2" ||
        name_ == "Rrel" || name_ == "Radius" || name_ == "Radius_p" ||
        name_ == "Radius_n" || name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (irr != u4::irrep(0, 0, 0, 0)) flag = false;
    }
    else if (name_ == "M1")
    {
      if (irr != u4::irrep(2, 1, 1, 0) && irr != u4::irrep(0, 0, 0, 0)) 
        flag = false;
    }
    else if (name_ == "FermiPlus" || name_ == "FermiMinus" ||
             name_ == "GamowTellerPlus" || name_ == "GamowTellerMinus")
    {
      if (irr != u4::irrep(2, 1, 1, 0)) flag = false;
    }
    
    return flag;
  }

  void OneBodyOperator::setTensorStrength(U3xU4::MultiShellTensor_phys& tensor,
                                    int na, int nb) const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah9 = qcx_utils::fanos();
    auto& wigU3SO3db = u3::wigU3SO3db();
    auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

    std::vector<std::tuple<uint8_t,              // kappamax
                           HO::qn_t,             // LL
                           uint8_t,              // zetamax
                           HO::qn_t,             // SS
                           HO::qn_t,             // TT
                           std::vector<double>>> LLSSTTstrengths;

    // decompose into L
    u3::physicalvectors pv3s = u3::physicalBasis(tensor.getLastIrrep().u3());

    // decompose into 2S and 2T
    u4::physicalvectors pv4s = u4::physicalBasis(tensor.getLastIrrep().u4());

    for (const auto& pv3 : pv3s)
    {
      int lam0 = pv3.lam(), mu0 = pv3.mu(),
          kappa0 = pv3.kappa(), LL0 = 2 * pv3.L();

      if (LLselection(LL0) == false) continue;

      for (const auto& pv4 : pv4s)
      {
        assert(pv4.zeta() == 1);

        int P10 = pv4.P1(), P20 = pv4.P2(), P30 = pv4.P3(),
            SS0 = pv4.SS(), TT0 = pv4.TT();

        if (SSselection(SS0) == false) continue;

        // if the operator has definite isospin assignment
        if (TT() >= 0 && TT() != TT0) continue;

        // couple LL0 and SS0 to JJ0
        if (u2::mult(LL0, SS0, JJ()) == 0) continue;

        const auto cgcU4 = (na <= nb)
          ? wigU4SU2SU2db.get(
              1, 0, 0, 1, 1, 0, 0, 1, 1, 1, P10, P20, P30, SS0, TT0)
          : wigU4SU2SU2db.get(
              0, 0, 1, 1, 1, 1, 0, 0, 1, 1, P10, P20, P30, SS0, TT0);

        if (cgcU4.empty()) continue;

        std::vector<double> strengths(kappa0, 0.0);

        // loop over la
        for (int lla = 2 * na; lla >= 2 * (na % 2); lla -= 4)
        {
          // loop over lb
          for (int llb = 2 * nb; llb >= 2 * (nb % 2); llb -= 4)
          {
            const auto cgcU3 = (na <= nb) 
              ? wigU3SO3db.get(
                  na, 0, lla / 2, 0, nb, llb / 2, lam0, mu0, LL0 / 2)
              : wigU3SO3db.get(
                  0, nb, llb / 2, na, 0, lla / 2, lam0, mu0, LL0 / 2);

            if (cgcU3.empty()) continue;

            // loop over jja
            for (int jja = lla + 1; jja >= abs(lla - 1); jja -= 2)
            {
              // loop over jjb
              for (int jjb = llb + 1; jjb >= std::fabs(llb - 1); jjb -= 2)
              {
                // check if jja x jjb -> JJ coupling is allowed
                if (JJ() < abs(jja - jjb) || JJ() > jja + jjb) continue;

                double U9j = (na <= nb)
                  ? racah9.get({lla, 1, jja, llb, 1, jjb, LL0, SS0, JJ()})
                  : racah9.get({llb, 1, jjb, lla, 1, jja, LL0, SS0, JJ()});

                // loop over projections
                for (int mmja = jja; mmja >= -jja; mmja -= 2)
                {
                  int mmjb = mmja - MMJ(); // selection rule of sphericalME
                  if (abs(mmjb) > jjb) continue;

                  double cgcJ = (na <= nb)
                    ? cgcs.get({jja, jjb, JJ(), mmja, -mmjb, MMJ()})
                    : cgcs.get({jjb, jja, JJ(), -mmjb, mmja, MMJ()});

                  for (int mmta = 1; mmta >= -1; mmta -= 2)
                  {
                    int mmtb = mmta - MMT(); // selection rule of sphericalME
                    if (abs(mmtb) > 1) continue;

                    double cgcT = (na <= nb)
                      ? cgcs.get({1, 1, TT0, mmta, -mmtb, MMT()})
                      : cgcs.get({1, 1, TT0, -mmtb, mmta, MMT()});

                    double me = sphericalME(na, lla, jja, mmja, mmta,
                                            nb, llb, jjb, mmjb, mmtb);
                    if (me == 0.0) continue;

#ifdef ADD_SU3LIB_PHASE_U3SO3
                    double phase = 
                      ((2 * nb + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0) 
                      ? 1.0 : -1.0;
#else
                    // phase = (-)^{(nb + lb + 1)/2 - jb + mjb + mtb}
                    double phase = 
                      ((nb + llb / 2 + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0)
                      ? 1.0 : -1.0;
#endif
                    double cgcU2 = phase * U9j * cgcJ * cgcT * cgcU4[0] * me;
                    if (na > nb) cgcU2 *= -1.0;

                    for (int k = 0; k < kappa0; k++)
                    {
                      double y =  cgcU2 * cgcU3[k];
                      // std::cout << y << std::endl;
                      strengths[k] += y;
                    }
                  }
                }
              }
            }
          }
        }

        double sum = 0.0;
        for (const auto& s : strengths) sum += std::fabs(s);

        if (std::fabs(sum) > 1e-8)
          LLSSTTstrengths.emplace_back(kappa0, LL0, 1, SS0, TT0, strengths);
      }
    }

    tensor.setLLSSTTstrengths(LLSSTTstrengths);
  }

  U3xU4::MultiShellTensors_phys 
    OneBodyOperator::expand_new()
  {
    if (on_the_fly == false) OBME->read();

    U3xU4::MultiShellTensors_phys tensors;

    u4::irrep u4a(1, 0, 0, 0), u4b(1, 1, 1, 0);
    auto [u4couple, u4multi] = u4::decomposition(u4a, u4b); 
    // same as u4::decomposition(u4b, u4a)

    // loop for ad
    for (int na = 0; na <= maxShell_; na++)
    {
      if ((na - valenceShell_) > Nmax_) continue;

      u3::irrep u3a(na, 0, 0);
      U3xU4::SingleShellTensor 
        a(na, 1, 0, {}, {}, U3xU4::irrep(u3a, u4a), 1, 1);

      // loop for at
      for (int nb = 0; nb <= maxShell_; nb++)
      {
        if ((nb - valenceShell_) > Nmax_) continue;

        u3::irrep u3b(nb, nb, 0);
        U3xU4::SingleShellTensor 
          b(nb, 0, 1, {}, {}, U3xU4::irrep(u3b, u4b), 1, 1);

        // coupled irreps
        std::vector<u3::irrep> u3couple;
        std::vector<int> u3multi;

        if (na <= nb)
          std::tie(u3couple, u3multi) = u3::decomposition(u3a, u3b);
        else // na > nb => switch the order
          std::tie(u3couple, u3multi) = u3::decomposition(u3b, u3a);
        
        for (int i = 0; i < u3couple.size(); i++)
        {
          assert(u3multi[i] == 1);

          if (U3selection(u3couple[i]) == false) continue;

          for (int j = 0; j < u4couple.size(); j++)
          {
            assert(u4multi[j] == 1); 

            if (U4selection(u4couple[j]) == false) continue;

            U3xU4::OmegaElement t(1, u3couple[i], 1, u4couple[j]);

            if (na == nb)
            {
              U3xU4::SingleShellTensor ab(na, 1, 1, {}, {}, 
                            U3xU4::irrep(u3couple[i], u4couple[j]), 1, 1);

              U3xU4::MultiShellTensor_phys 
                tensor({ab}, {}, MMT_, JJ_, MMJ_, HO_labels_);

              setTensorStrength(tensor, na, nb);

              if (!tensor.LLSSTTstrengths().empty())
                tensors.push_back(std::move(tensor));
            }
            else
            {
              U3xU4::MultiShellTensor_phys tensor(
                (na < nb) ? std::vector({a, b}) : std::vector({b, a}), 
                {t}, MMT_, JJ_, MMJ_, HO_labels_);
              if (tensor.shift_quanta() < 0) continue;

              setTensorStrength(tensor, na, nb);

              if (!tensor.LLSSTTstrengths().empty())
                tensors.push_back(std::move(tensor));
            }
          }
        }
      }
    }

    return tensors;
  }

  U3xU4::MultiShellTensors_phys 
    OneBodyOperator::expand()
  {
    std::string base_dir = dirname() + "/Expansion";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_ + "_hw";
    if (on_the_fly) filename += "20";
    else filename += std::to_string(static_cast<int>(hw_));

    // if this file exists, returns immediately
    if (std::filesystem::exists(filename))
    {
      return readBinary(filename);
    }
    else
    {
#ifdef USE_MPI
      int mpi_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

      if (mpi_rank == 0)
        std::cout << "File doesn't exist, new expansion is called" << std::endl;
#endif

      auto requested_hw = hw_;
      // only change hw to 20 if the matrix elements are computed on the fly
      if (on_the_fly) hw_ = 20;

      auto tensors = expand_new();

#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
      writeBinary(filename, tensors);

      hw_ = requested_hw;
      // return tensors;
      return readBinary(filename);
    }
  }
} // namespace JT
