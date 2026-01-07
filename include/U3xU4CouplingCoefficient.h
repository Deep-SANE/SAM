#pragma once

#include <Databases.h>
#include <U3xU4irrep.h>

namespace U3xU4
{
  /* A class to store coupling coefficient of the form

  / (lam1,mu1)(P1,P1',P1'')   ; (lam2,mu2)(P2,P2',P2'')   | (lam,mu)(P,P',P'') \
  \ k1 L1 z1 S1 T1 MT1 J1 MJ1 ; k2 L2 z2 S2 T2 MT2 J2 MJ2 | k L z S T MT J MJ
  /rho,eta
  */
  class CouplingCoefficient
  {
    std::vector<double> coeffs;

    int k1max, k2max, k3max, z1max, z2max, z3max, rhomax, etamax;

  public:
    CouplingCoefficient(const U3xU4::irrep& a, int LL1, int SS1, int TT1,
                        int MMT1, int JJ1, int MMJ1, const U3xU4::irrep& b,
                        int LL2, int SS2, int TT2, int MMT2, int JJ2, int MMJ2,
                        const U3xU4::irrep& c, int LL3, int SS3, int TT3,
                        int MMT3, int JJ3, int MMJ3,
                        sam::Databases& dbs)
    {
      auto& cgcs = dbs.cgcs;
      auto& racah9 = dbs.racah9;
      auto& wigU3SO3db = dbs.wigU3SO3db;
      auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

      double cgcJ = cgcs.get({JJ1, JJ2, JJ3, MMJ1, MMJ2, MMJ3}),
             cgcT = cgcs.get({TT1, TT2, TT3, MMT1, MMT2, MMT3}),
             U9j = racah9.get({LL1, SS1, JJ1, LL2, SS2, JJ2, LL3, SS3, JJ3}),
             factor = cgcJ * cgcT * U9j;

      auto [allowU3, cgcU3] = wigU3SO3db.get(a.lam(), a.mu(), LL1 / 2, 
                                             b.lam(), b.mu(), LL2 / 2, 
                                             c.lam(), c.mu(), LL3 / 2);

      auto [allowU4, cgcU4] =
          wigU4SU2SU2db.get(a.P1(), a.P2(), a.P3(), SS1, TT1,
                            b.P1(), b.P2(), b.P3(), SS2, TT2,
                            c.P1(), c.P2(), c.P3(), SS3, TT3);

      k1max = u3::Kappa(a.u3(), LL1 / 2);
      k2max = u3::Kappa(b.u3(), LL2 / 2);
      k3max = u3::Kappa(c.u3(), LL3 / 2);
      z1max = u4::Zeta(a.u4(), SS1, TT1);
      z2max = u4::Zeta(b.u4(), SS2, TT2);
      z3max = u4::Zeta(c.u4(), SS3, TT3);
      rhomax = u3::Rho(a.u3(), b.u3(), c.u3());
      etamax = u4::Eta(a.u4(), b.u4(), c.u4());

      for (int k1 = 0; k1 < k1max; k1++)
        for (int k2 = 0; k2 < k2max; k2++)
          for (int k3 = 0; k3 < k3max; k3++)
            for (int z1 = 0; z1 < z1max; z1++)
              for (int z2 = 0; z2 < z2max; z2++)
                for (int z3 = 0; z3 < z3max; z3++)
                  for (int rho = 0; rho < rhomax; rho++)
                    for (int eta = 0; eta < etamax; eta++)
                    {
                      double c = factor *
                                 cgcU3((k1 * k2max + k2) * k3max + k3, rho) *
                                 cgcU4((z1 * z2max + z2) * z3max + z3, eta);
                      coeffs.push_back(c);
                    }
    }

    double get(int k1, int k2, int k3, int z1, int z2, int z3, int rho,
               int eta) const
    {
      int index =
          ((((((k1 * k2max + k2) * k3max + k3) * z1max + z1) * z2max + z2) *
                z3max +
            z3) *
               rhomax +
           rho) *
              etamax +
          eta;
      return coeffs[index];
    }
  };
} // namespace U3xU4
