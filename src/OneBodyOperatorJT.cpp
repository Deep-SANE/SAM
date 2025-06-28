#include <cassert>
#include <cmath>

#include <OneBodyOperatorJT.h>

#include <U3decomposition.h>

#include <U4decomposition.h>

namespace JT
{
  // ME: < na lla jja mmja mmta | O | nb llb jjb mmjb mmtb >
  double OneBodyOperator::sphericalME(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb)
  {
    double x = 0.0;
    switch (type)
    {
    case 1: // N operator
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = nb;
      break;
    case 2: // One-body part of L^2
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = llb * (llb + 2.0) / 4.0;
      break;
    case 3: // One-body-part of L.S
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = (jjb * (jjb + 2.0) - llb * (llb + 2.0) - 3.0) / 8.0;
      break;
    case 4: // One-body part of CoM number operator
      if (na == nb && jja == jjb && mmja == mmjb && mmta == mmtb)
      {
      	double factor = 1.0 / (jjb + 1);
      	int nc = nb - 1, llc_min = (nc % 2 == 0) ? 0 : 2;
      	for (int llc = 2 * nc; llc >= llc_min; llc -= 2)
      	{
      		for (int jjc = llc + 1; jjc >= llc - 1; jjc -= 2)
      		{
      			double phase = ((jjc - jjb) / 2 * 2 == 0) ? 1.0 : -1.0,
      				   rme_bd, rme_b;

      			// < na la ja || bd || nc lc jc >, na = nb = nc + 1
      			double coeff_bd = 1.0;
      			if (lla == llc - 2)
      				rme_bd = - coeff_bd * std::sqrt(llc * (2.0 * nc - llc + 4.0) / 4.0); 
      			else if (lla == llc + 2)
      				rme_bd = coeff_bd * std::sqrt((llc + 2.0) * (2.0 * nc + llc + 6.0) / 4.0); 
      			else
      				continue;

      			// < nc lc jc || b || nb lb jb > 
      			double coeff_b = 1.0;
      			if (llc == llb - 2)
      				rme_b = - coeff_b * std::sqrt(llb * (2.0 * nb + llb + 2.0) / 4.0);
      			else if (llc == llb + 2)
      				rme_b = coeff_b * std::sqrt((llb + 2.0) * (2.0 * nb - llb) / 4.0);
      			else
      				continue;

      			x += phase * rme_bd * rme_b;
      		}
      	}
      }
    default:
      throw std::invalid_argument("ME has not been implemented");
    }
    return x;
  }

  // RME: < na lla jja || O || nb llb jjb >
  // = ME / ( < jb mjb ; J MJ | ja mja > * < 1/2 mtb ; T MT | 1/2 mta > )
  double OneBodyOperator::sphericalRME(int na, int lla, int jja, int nb, int llb, int jjb)
  {
    double x = 0.0;
    switch (type)
    {
    case 1: // N operator
      if (na == nb && lla == llb && jja == jjb)
        x = nb;
      break;
    case 2: // One-body part of L^2
      if (na == nb && lla == llb && jja == jjb)
        x = llb * (llb + 2.0) / 4.0;
      break;
    default:
      throw std::invalid_argument("ME has not been implemented");
    }

    return x;
  }

  // Implement using sphericalRME to avoid loops over projection mj and mt
  std::vector<double> OneBodyOperator::getTensorStrength(
      int kappa0, int LL0, int SS0, int na, int nb,
      const std::vector<u3::physicalvector>& pv3as,
      const std::vector<u3::physicalvector>& pv3bs,
      const std::vector<u4::physicalvector>& pv4as,
      const std::vector<u4::physicalvector>& pv4bs, int pv3,
      const std::vector<std::tuple<int, int, int>>& tArr3,
      const std::vector<Eigen::MatrixXd>& coeffs3, int pv4,
      const std::vector<std::tuple<int, int, int>>& tArr4,
      const std::vector<Eigen::MatrixXd>& coeffs4, u34utils::SO3_CGC& cgcs,
      u34utils::SO3_Racah9& racah9)
  {
    std::vector<double> x(kappa0, 0.0);

    // loop over la
    for (int pv3a = 0; pv3a < pv3as.size(); pv3a++)
    {
      int lla = 2 * pv3as[pv3a].L;
      // loop over jja
      for (int jja = lla + 1; jja >= std::fabs(lla - 1); jja -= 2)
      {
        // loop over lb
        for (int pv3b = 0; pv3b < pv3bs.size(); pv3b++)
        {
          int llb = 2 * pv3bs[pv3b].L;

          // check if lla x llb -> LL0 coupling is allowed
          if (LL0 < std::fabs(lla - llb) || LL0 > lla + llb)
          	continue;

          // loop over jjb
          for (int jjb = llb + 1; jjb >= std::fabs(llb - 1); jjb -= 2)
          {
          	// check if jja x jjb -> JJ coupling is allowed
          	if (JJ < std::fabs(jja - jjb) || JJ > jja + jjb)
          		continue;

            double rme = sphericalRME(na, lla, jja, nb, llb, jjb);
            if (rme == 0.0)
            	continue;

            double U9j;
            if (na <= nb)
              U9j = racah9.compute(lla, 1, jja, llb, 1, jjb, LL0, SS0, JJ);
            else // na > nb
              U9j = racah9.compute(llb, 1, jjb, lla, 1, jja, LL0, SS0, JJ);
            //std::cout << "U9j = " << U9j << std::endl;

            double cgcU4;
            for (int i = 0; i < tArr4.size(); i++)
              if (tArr4[i] == std::make_tuple(0, 0, pv4))
              {
                cgcU4 = coeffs4[i](0, 0);
                break;
              }
            //std::cout << "cgcU4 = " << cgcU4 << std::endl;

            if (na <= nb)
            {
              double phase = ((nb + llb / 2 + 2)/2 % 2 == 0) ? 1.0 : -1.0,
              factor = std::sqrt(2.0 * (jja + 1.0) / (JJ + 1.0) / (TT + 1.0));
              //std::cout << "phase = " << phase << std::endl;
              for (int i = 0; i < tArr3.size(); i++)
                if (tArr3[i] == std::make_tuple(pv3a, pv3b, pv3))
                {
                  for (int k = 0; k < kappa0; k++)
                  {
                    double y = phase * U9j * coeffs3[i](k, 0) * cgcU4 
                    		 * rme * factor ;
                    //std::cout << y << std::endl;
                    x[k] += y;
                  }
                  break;
                }
            }

            else // na > nb
            {
              double phase = ((nb + llb / 2 + jja + jjb - JJ - TT)/2 % 2 == 0) ? 1.0 : -1.0,
              factor = std::sqrt(2.0 * (jja + 1.0) / (JJ + 1.0) / (TT + 1.0));
              //std::cout << "phase = " << phase << std::endl;
              for (int i = 0; i < tArr3.size(); i++)
                if (tArr3[i] == std::make_tuple(pv3b, pv3a, pv3))
                {
                  for (int k = 0; k < kappa0; k++)
                  {
                    double y = phase * U9j * coeffs3[i](k, 0) * cgcU4 
                    		 * rme * factor;
                    //std::cout << y << std::endl;
                    x[k] -= y;
                  }
                  break;
                }
            }
          }
        }
      }
    }
    return x;
  }

  std::pair<std::vector<double>, std::vector<U3xU4::OneBodyMultiShellTensor>>
  OneBodyOperator::expand(int maxShell, u34utils::SO3_CGC& cgcs,
                          u34utils::SO3_Racah9& racah9,
                          u3::WigU3U2_db& wigU3U2db, u4::WigU4U3_db& wigU4U3db)
  {
    std::vector<double> coeffs;
    std::vector<U3xU4::OneBodyMultiShellTensor> tensors;

    // loop for ad
    for (int na = 0; na <= maxShell; na++)
    {
      u3::irrep u3a(na, 0, 0);
      std::vector<u3::physicalvector> pv3as =
          u3::physicalBasis(u3a); // array of la
      u4::irrep u4a(1, 0, 0, 0);
      std::vector<u4::physicalvector> pv4as =
          u4::physicalBasis(u4a); // only one element 2sa = 2ta = 1

      // loop for at
      for (int nb = 0; nb <= maxShell; nb++)
      {
        u3::irrep u3b(nb, nb, 0);
        std::vector<u3::physicalvector> pv3bs =
            u3::physicalBasis(u3b); // array of lb
        u4::irrep u4b(1, 1, 1, 0);
        std::vector<u4::physicalvector> pv4bs =
            u4::physicalBasis(u4b); // only one element 2sb = 2tb = 1

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

          // compute coupling coefficients
          std::vector<std::tuple<int, int, int>> tArr3;
          std::vector<Eigen::MatrixXd> coeffs3;

          if (na <= nb)
            coeffs3 = u3::WigU3SO3(u3multi[i], pv3as, pv3bs, pv3s, tArr3, cgcs,
                                   wigU3U2db);
          else // na > nb
            coeffs3 = u3::WigU3SO3(u3multi[i], pv3bs, pv3as, pv3s, tArr3, cgcs,
                                   wigU3U2db);

          for (int j = 0; j < u4couple.size(); j++)
          {
            assert(u4multi[j] == 1);

            // decompose into 2S0 and 2T0
            std::vector<u4::physicalvector> pv4s =
                u4::physicalBasis(u4couple[j]);

            // compute coupling coefficients
            std::vector<std::tuple<int, int, int>> tArr4;
            std::vector<Eigen::MatrixXd> coeffs4;

            if (na <= nb)
              coeffs4 = u4::WigU4SU2SU2(u4multi[j], pv4as, pv4bs, pv4s, tArr4,
                                        cgcs, wigU3U2db, wigU4U3db);
            else // na > nb
              coeffs4 = u4::WigU4SU2SU2(u4multi[j], pv4bs, pv4as, pv4s, tArr4,
                                        cgcs, wigU3U2db, wigU4U3db);

            for (int pv3 = 0; pv3 < pv3s.size(); pv3++)
            {
              int kappa0 = pv3s[pv3].kappa, LL0 = 2 * pv3s[pv3].L;

              for (int pv4 = 0; pv4 < pv4s.size(); pv4++)
              {
                int SS0 = pv4s[pv4].SS, TT0 = pv4s[pv4].TT;

                // couple LL0 and SS0 to JJ0
                // the sum over CGCs will kill JJ0 != JJ and TT0 != TT
                if (JJ < std::fabs(LL0 - SS0) || JJ > LL0 + SS0 || TT0 != TT)
                	continue;
                
                // the selection rule of CGCs also dictates that
                // MMJ0 == MMJ and MMT0 == MMT

                // compute the expansion coefficient
                std::vector<double> coeff = getTensorStrength(
                    kappa0, LL0, SS0, na, nb, pv3as, pv3bs, pv4as, pv4bs, pv3,
                    tArr3, coeffs3, pv4, tArr4, coeffs4, cgcs, racah9);

                for (int k = 0; k < kappa0; k++)
                {
                  if (std::fabs(coeff[k]) > 1e-8)
                  {
                    coeffs.push_back(coeff[k]);
                    if (na <= nb)
                    {
                      U3xU4::OneBodyMultiShellTensor tensor(
                          na, -nb, u3couple[i], u4couple[j], k + 1, LL0, SS0,
                          TT, MMT, JJ, MMJ);
                      tensors.push_back(tensor);
                    }
                    else // na > nb
                    {
                      U3xU4::OneBodyMultiShellTensor tensor(
                          -nb, na, u3couple[i], u4couple[j], k + 1, LL0, SS0,
                          TT, MMT, JJ, MMJ);
                      tensors.push_back(tensor);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    return std::make_pair(coeffs, tensors);
  }
} // namespace JT

/* Implementation using sphericalME
  std::vector<double> OneBodyOperator::getTensorStrength(
      int kappa0, int LL0, int SS0, int na, int nb,
      const std::vector<u3::physicalvector>& pv3as,
      const std::vector<u3::physicalvector>& pv3bs,
      const std::vector<u4::physicalvector>& pv4as,
      const std::vector<u4::physicalvector>& pv4bs, int pv3,
      const std::vector<std::tuple<int, int, int>>& tArr3,
      const std::vector<Eigen::MatrixXd>& coeffs3, int pv4,
      const std::vector<std::tuple<int, int, int>>& tArr4,
      const std::vector<Eigen::MatrixXd>& coeffs4, u34utils::SO3_CGC& cgcs,
      u34utils::SO3_Racah9& racah9)
  {
    std::vector<double> x(kappa0, 0.0);

    // loop over la
    for (int pv3a = 0; pv3a < pv3as.size(); pv3a++)
    {
      int lla = 2 * pv3as[pv3a].L;
      // loop over jja
      for (int jja = lla + 1; jja >= std::fabs(lla - 1); jja -= 2)
      {
        // loop over lb
        for (int pv3b = 0; pv3b < pv3bs.size(); pv3b++)
        {
          int llb = 2 * pv3bs[pv3b].L;

          if (LL0 < std::fabs(lla - llb) || LL0 > lla + llb)
          	continue;

          // loop over jjb
          for (int jjb = llb + 1; jjb >= std::fabs(llb - 1); jjb -= 2)
          {
          	if (JJ < std::fabs(jja - jjb) || JJ > jja + jjb)
          		continue;

            double U9j;
            if (na <= nb)
              U9j = racah9.compute(lla, 1, jja, llb, 1, jjb, LL0, SS0, JJ);
            else // na > nb
              U9j = racah9.compute(llb, 1, jjb, lla, 1, jja, LL0, SS0, JJ);
            if (std::fabs(U9j) < 1e-14)
              continue;
            //std::cout << "U9j = " << U9j << std::endl;

            double cgcU4;
            for (int i = 0; i < tArr4.size(); i++)
              if (tArr4[i] == std::make_tuple(0, 0, pv4))
              {
                cgcU4 = coeffs4[i](0, 0);
                break;
              }
            //std::cout << "cgcU4 = " << cgcU4 << std::endl;

            // loop over projections
            for (int mmja = jja; mmja >= -jja; mmja -= 2)
            {
              int mmjb = mmja - MMJ; // selection rule of sphericalME

              for (int mmta = 1; mmta >= -1; mmta -= 2)
              {
                int mmtb = mmta - MMT; // selection rule of sphericalME

                double me = sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb,
                                        mmjb, mmtb);
                if (me == 0.0)
                  continue;

                // phase = (-)^{(nb + lb + 1)/2 - jb + mjb + mtb}
                double phase =
                    ((nb + pv3bs[pv3b].L + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0)
                        ? 1.0
                        : -1.0;

                if (na <= nb)
                {
                  double cgcJ = cgcs.compute(jja, jjb, JJ, mmja, -mmjb, MMJ),
                         cgcT = cgcs.compute(1, 1, TT, mmta, -mmtb, MMT);

                  for (int i = 0; i < tArr3.size(); i++)
                    if (tArr3[i] == std::make_tuple(pv3a, pv3b, pv3))
                    {
                      for (int k = 0; k < kappa0; k++)
                      {
                        double y = phase * cgcJ * cgcT * U9j *
                                   coeffs3[i](k, 0) * cgcU4 * me;
                         std::cout << y << std::endl;
                        x[k] += y;
                      }
                      break;
                    }
                }
                else // na > nb
                {
                  double cgcJ = cgcs.compute(jjb, jja, JJ, -mmjb, mmja, MMJ),
                         cgcT = cgcs.compute(1, 1, TT, -mmtb, mmta, MMT);

                  for (int i = 0; i < tArr3.size(); i++)
                    if (tArr3[i] == std::make_tuple(pv3b, pv3a, pv3))
                    {
                      for (int k = 0; k < kappa0; k++)
                      {
                        double y = phase * cgcJ * cgcT * U9j *
                                   coeffs3[i](k, 0) * cgcU4 * me;
                         std::cout << y << std::endl;
                        x[k] -= y;
                      }
                      break;
                    }
                }
              }
            }
          }
        }
      }
    }
    return x;
  }
*/