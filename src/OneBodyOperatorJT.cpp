#include <cassert>
#include <cmath>

#include <HOutilities.h>
#include <OneBodyOperatorJT.h>

#include <U3decomposition.h>

#include <U4decomposition.h>

namespace JT
{
  // Check the type of this operator
  void OneBodyOperator::is() const
  {
    switch (type)
    {
    case 1:
      std::cout << "Number operator N" << std::endl;
      break;
    case 2:
      std::cout << "One-body part of L.L" << std::endl;
      break;
    case 3:
      std::cout << "One-body part of L.S" << std::endl;
      break;      
    case 4:
      std::cout << "One-body part of the CoM number operator without 1/A" << std::endl;
      break;
    case 5:
      std::cout << "One-body part of the relative kinetic energy without hw(A-1)/A" << std::endl;
      break;
    case 6:
      std::cout << "One-body part of the intrinsic mass radius squared without (A-1)/A" << std::endl;
      break;
    case 7:
      std::cout << "E2 transition operator without (HO strength)^2" << std::endl;
      break;
    case 8:
      std::cout << "M1 transition operator without sqrt(3/4pi)" << std::endl;
      break;
    case 9:
      std::cout << "Fermi for beta+" << std::endl;
      break;  
    case 10:
      std::cout << "Fermi for beta-" << std::endl;
      break; 
    case 11:
      std::cout << "Gamow-Teller for beta+" << std::endl;
      break;  
    case 12:
      std::cout << "Gamow-Teller for beta-" << std::endl;
      break;  
    default:
      throw std::invalid_argument("This operator has not been implemented");
    }
  }

  // ME: < na lla jja mmja mmta | O | nb llb jjb mmjb mmtb >
  double OneBodyOperator::sphericalME(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb,
                                      qcx_utils::SO3_CGC& cgcs, 
                                      qcx_utils::SO3_Racah& racah,
                                      qcx_utils::SO3_Racah9& racah9) const
  {
    double x = 0.0;
    switch (type)
    {
    case 1: // N operator
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = nb;
      break;
    case 2: // One-body part of LL
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = llb * (llb + 2.0) / 4.0;
      break;
    case 3: // One-body-part of LS
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = (jjb * (jjb + 2.0) - llb * (llb + 2.0) - 3.0) / 8.0;
      break;
    case 4: // One-body part of CoM number operator
      if (na == nb && jja == jjb && mmja == mmjb && mmta == mmtb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));

            x += phase * factor 
              * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
              * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
      }
      break;
    case 5: // One-body part of the relative kinetic energy without hw(A-1)/A
      if (jja == jjb && mmja == mmjb && mmta == mmtb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor 
                * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
            else if (na == nb - 2)
              x -= phase * factor 
                * HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor 
                * HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonCreatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
            else if (na == nb + 2)
              x -= phase * factor 
                * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonCreatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
      }
      x /= 4.0;
      break;
    case 6: // One-body part of the intrinsic mass radius squared without (A-1)/A
      if (jja == jjb && mmja == mmjb && mmta == mmtb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor 
                * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
            else if (na == nb - 2)
              x += phase * factor 
                * HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor 
                * HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonCreatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
            else if (na == nb + 2)
              x += phase * factor 
                * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonCreatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
      }
      break;
    case 7: // E2 transition operator without (HO strength)^2
      if (mmta == mmtb && mmtb == -1)
      {
        double phase = ((lla - llb + jjb - jja)/2 % 2 == 0) ? 1.0 : -1.0,
               factor = cgcs.compute(jjb,4,jja,mmjb,0,mmja)
                      * racah.compute(llb,1,4,jja,jjb,lla);
        int nc = nb - 1, llc_min = 2 * (nc % 2 );
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          double factor = racah.compute(llb,2,lla,2,llc,4);
          if (na == nb)
            x += factor 
              * HO::bosonCreatorRME(na,lla,nc,llc)
              * HO::bosonAnnihilatorRME(nc,llc,nb,llb);
          else if (na == nb - 2)
            x += factor 
              * HO::bosonAnnihilatorRME(na,lla,nc,llc)
              * HO::bosonAnnihilatorRME(nc,llc,nb,llb);
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          double factor = racah.compute(llb,2,lla,2,llc,4);
          if (na == nb)
            x += factor 
              * HO::bosonAnnihilatorRME(na,lla,nc,llc)
              * HO::bosonCreatorRME(nc,llc,nb,llb);
          else if (na == nb + 2)
            x += factor 
              * HO::bosonCreatorRME(na,lla,nc,llc)
              * HO::bosonCreatorRME(nc,llc,nb,llb);
        }
      }
      x *= std::sqrt(1.5);
      break;
    case 8: // M1 transition operator without sqrt(3/4pi)
      if (na == nb && lla == llb && mmta == mmtb)
      {
        double gl, gs;
        if (mmta == 1) // neutron
        {
          gl = 0;
          gs = -3.8263;
        }
        else if (mmta == -1) // proton
        {
          gl = 1;
          gs = 5.5857;
        }
        double phase = ((jjb - jja)/2 % 2 == 0) ? 1.0 : -1.0;
        x = cgcs.compute(jjb,2,jja,mmjb,0,mmja) 
          * (gl * phase * racah.compute(llb,1,2,jja,jjb,llb) 
          * std::sqrt(llb * (llb + 2.0) / 4.0)
          + gs * racah.compute(1,llb,2,jja,jjb,1) * std::sqrt(0.75));
      }
      break;
    case 9: // Fermi for beta+
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb + 2)
        x = std::sqrt(0.75 - mmtb * (mmtb - 2.0) / 4.0);
      break;
    case 10: // Fermi for beta-
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb - 2)
        x = std::sqrt(0.75 - mmtb * (mmtb + 2.0) / 4.0);
      break;
    default:
      throw std::invalid_argument("ME has not been implemented");
    }
    return x;
  }

  // RME: < na lla jja || O || nb llb jjb >
  // = ME / ( < jb mjb ; J MJ | ja mja > * < 1/2 mtb ; T MT | 1/2 mta > )
  double OneBodyOperator::sphericalRME(int na, int lla, int jja, int nb,
                                       int llb, int jjb,
                                       qcx_utils::SO3_CGC& cgcs,
                                       qcx_utils::SO3_Racah& racah, 
                                       qcx_utils::SO3_Racah9& racah9) const
  {
    double x = 0.0;
    switch (type)
    {
    case 1: // N operator
      if (na == nb && lla == llb && jja == jjb)
        x = nb;
      break;
    case 2: // One-body part of LL
      if (na == nb && lla == llb && jja == jjb)
        x = llb * (llb + 2.0) / 4.0;
      break;
    case 3: // One-body-part of LS
      if (na == nb && lla == llb && jja == jjb)
        x = (jjb * (jjb + 2.0) - llb * (llb + 2.0) - 3.0) / 8.0;
      break;
    case 4: // One-body part of CoM number operator
      if (na == nb && jja == jjb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));

            x += phase * factor 
              * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
              * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
      }
      break;
    case 5: // One-body part of the relative kinetic energy without hw(A-1)/A
      if (jja == jjb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor 
                * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
            else if (na == nb - 2)
              x -= phase * factor 
                * HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor 
                * HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonCreatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
            else if (na == nb + 2)
              x -= phase * factor 
                * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonCreatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
      }
      x /= 4.0;
      break;
    case 6: // One-body part of the intrinsic mass radius squared without (A-1)/A
      if (jja == jjb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor 
                * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
            else if (na == nb - 2)
              x += phase * factor 
                * HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonAnnihilatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb)/2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor 
                * HO::bosonAnnihilatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonCreatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
            else if (na == nb + 2)
              x += phase * factor 
                * HO::bosonCreatorRME(na,lla,jja,nc,llc,jjc,racah9)
                * HO::bosonCreatorRME(nc,llc,jjc,nb,llb,jjb,racah9);
          }
        }
      }
      break;
    default:
      throw std::invalid_argument("RME has not been implemented");
    }
    return x;
  }

  // Using sphericalRME to avoid loops over projection mj and mt
  std::pair<bool, std::vector<double>> OneBodyOperator::getTensorStrength(
      int lam0, int mu0, int P10, int P20, int P30, int kappa0, int LL0,
      int SS0, int na, int nb, 
      qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racah,
      qcx_utils::SO3_Racah9& racah9, u3::WigU3SO3_db& wigU3SO3db,
      u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
  {
    std::vector<double> x(kappa0, 0.0);
    double sum = 0.0;
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
            if (JJ < std::fabs(jja - jjb) || JJ > jja + jjb)
              continue;

            double rme;
            if (USE_ONEBODY_SPHERICAL_RME)
            { 
              rme = sphericalRME(na,lla,jja,nb,llb,jjb,cgcs,racah,racah9);
              if (rme == 0.0)
                continue;
            }

            double U9j;
            if (na <= nb)
              U9j = racah9.compute(lla, 1, jja, llb, 1, jjb, LL0, SS0, JJ);
            else // na > nb
              U9j = racah9.compute(llb, 1, jjb, lla, 1, jja, LL0, SS0, JJ);

            Eigen::MatrixXd Coeffs3, Coeffs4;
            if (na <= nb)
            {
              Coeffs3 =
                  wigU3SO3db
                      .get(na, 0, lla / 2, 0, nb, llb / 2, lam0, mu0, LL0 / 2)
                      .second;
              Coeffs4 =
                  wigU4SU2SU2db
                      .get(1, 0, 0, 1, 1, 0, 0, 1, 1, 1, P10, P20, P30, SS0, TT)
                      .second;
            }
            else // na > nb
            {
              Coeffs3 =
                  wigU3SO3db
                      .get(0, nb, llb / 2, na, 0, lla / 2, lam0, mu0, LL0 / 2)
                      .second;
              Coeffs4 =
                  wigU4SU2SU2db
                      .get(0, 0, 1, 1, 1, 1, 0, 0, 1, 1, P10, P20, P30, SS0, TT)
                      .second;
            }

            if (USE_ONEBODY_SPHERICAL_RME)
            {
              if (na <= nb)
              {
                double phase = ((nb + llb / 2 + 2) / 2 % 2 == 0) ? 1.0 : -1.0,
                       factor =
                         std::sqrt(2.0 * (jja + 1.0) / (JJ + 1.0) / (TT + 1.0));
                for (int k = 0; k < kappa0; k++)
                {
                  double y =
                    phase * U9j * Coeffs3(k, 0) * Coeffs4(0, 0) * rme * factor;
                  // std::cout << y << std::endl;
                  sum += y;
                  x[k] += y;
                }
              }
              else // na > nb
              {
                double phase = ((nb + llb / 2 + jja + jjb - JJ - TT) / 2 % 2 == 0)
                                   ? 1.0
                                   : -1.0,
                       factor =
                         std::sqrt(2.0 * (jja + 1.0) / (JJ + 1.0) / (TT + 1.0));
                for (int k = 0; k < kappa0; k++)
                {
                  double y =
                    phase * U9j * Coeffs3(k, 0) * Coeffs4(0, 0) * rme * factor;
                  // std::cout << y << std::endl;
                  sum += y;
                  x[k] += y;
                }
              }
            }
            else
            {
              // loop over projections
              for (int mmja = jja; mmja >= -jja; mmja -= 2)
              {
                int mmjb = mmja - MMJ; // selection rule of sphericalME

                for (int mmta = 1; mmta >= -1; mmta -= 2)
                {
                  int mmtb = mmta - MMT; // selection rule of sphericalME

                  double me = sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb,
                                          mmjb, mmtb, cgcs, racah, racah9);
                  if (me == 0.0)
                    continue;

                  // phase = (-)^{(nb + lb + 1)/2 - jb + mjb + mtb}
                  double phase =
                    ((nb + llb / 2 + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0)
                        ? 1.0
                        : -1.0;

                  if (na <= nb)
                  {
                    double cgcJ = cgcs.compute(jja, jjb, JJ, mmja, -mmjb, MMJ),
                           cgcT = cgcs.compute(1, 1, TT, mmta, -mmtb, MMT);

                    for (int k = 0; k < kappa0; k++)
                    {
                      double y = phase * cgcJ * cgcT * U9j * Coeffs3(k, 0) *
                                 Coeffs4(0, 0) * me;
                      // std::cout << y << std::endl;
                      sum += y;
                      x[k] += y;
                    }
                  }
                  else // na > nb
                  {
                    double cgcJ = cgcs.compute(jjb, jja, JJ, -mmjb, mmja, MMJ),
                           cgcT = cgcs.compute(1, 1, TT, -mmtb, mmta, MMT);

                    for (int k = 0; k < kappa0; k++)
                    {
                      double y = phase * cgcJ * cgcT * U9j * Coeffs3(k, 0) *
                               Coeffs4(0, 0) * me;
                      // std::cout << y << std::endl;
                      sum -= y;
                      x[k] -= y;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } 
    //std::cout << "Finish getTensorStrength\n";
    return std::make_pair((std::fabs(sum) < 1e-8), x);
  }

  std::vector<U3xU4::OneBodyMultiShellTensor> OneBodyOperator::expand(
      int maxShell, qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racah,
      qcx_utils::SO3_Racah9& racah9, u3::WigU3SO3_db& wigU3SO3db, 
      u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
  {
    std::vector<U3xU4::OneBodyMultiShellTensor> tensors;
    
    // loop for ad
    for (int na = 0; na <= maxShell; na++)
    {
      u3::irrep u3a(na, 0, 0);
      u4::irrep u4a(1, 0, 0, 0);

      // loop for at
      for (int nb = 0; nb <= maxShell; nb++)
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
              int lam0 = pv3s[pv3].toprow.lam(), mu0 = pv3s[pv3].toprow.mu(),
                  kappa0 = pv3s[pv3].kappa, LL0 = 2 * pv3s[pv3].L;

              for (int pv4 = 0; pv4 < pv4s.size(); pv4++)
              {
                int P10 = pv4s[pv4].toprow.P1(), P20 = pv4s[pv4].toprow.P2(),
                    P30 = pv4s[pv4].toprow.P3(), SS0 = pv4s[pv4].SS,
                    TT0 = pv4s[pv4].TT;

                // couple LL0 and SS0 to JJ0
                // the sum over CGCs will kill JJ0 != JJ and TT0 != TT
                if (JJ < std::fabs(LL0 - SS0) || JJ > (LL0 + SS0) || TT0 != TT)
                  continue;

                // the selection rule of CGCs also dictates that
                // MMJ0 == MMJ and MMT0 == MMT

                // compute the expansion coefficients
                auto [ignore, coeff] =
                    getTensorStrength(lam0, mu0, P10, P20, P30, kappa0, LL0,
                                      SS0, na, nb, //pv3as, pv3bs, pv4as, pv4bs,
                                      cgcs, racah, racah9, wigU3SO3db, wigU4SU2SU2db);
                
                if (ignore == false)
                {
                  if (na <= nb)
                  {
                    U3xU4::OneBodyMultiShellTensor tensor(
                        U3xU4::irrep(u3a,u4a),
                        U3xU4::irrep(u3b,u4b),
                        U3xU4::irrep(u3couple[i], u4couple[j]), kappa0, LL0,
                        SS0, TT, MMT, JJ, MMJ, coeff);
                    tensors.push_back(tensor);
                  }
                  else // na > nb
                  {
                    U3xU4::OneBodyMultiShellTensor tensor(
                        U3xU4::irrep(u3b,u4b),
                        U3xU4::irrep(u3a,u4a),
                        U3xU4::irrep(u3couple[i], u4couple[j]), kappa0, LL0,
                        SS0, TT, MMT, JJ, MMJ, coeff);
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
} // namespace JT
