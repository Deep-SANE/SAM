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
  // ME: < na lla jja mmja mmta | O | nb llb jjb mmjb mmtb >
  double OneBodyOperator::sphericalME(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb) const
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& racah = qcx_utils::racahs();
    auto& racah9 = qcx_utils::fanos();

    double x = 0.0;

    if (name_ == "NumberOp") // N operator
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = nb;
    }
    else if (name_ == "L.L") // One-body part of LL
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = llb * (llb + 2.0) / 4.0;
    }
    else if (name_ == "L.S") // One-body-part of LS
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = (jjb * (jjb + 2.0) - llb * (llb + 2.0) - 3.0) / 8.0;
    }
    else if (name_ == "CoMNumberOp") // One-body part of CoM number operator
    {
      if (na == nb && jja == jjb && mmja == mmjb && mmta == mmtb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        if (nc >= 0)
          for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
          {
            for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
            {
              double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                     factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));

              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
            }
          }
      }
    }
    else if (name_ == "Kinetic") // One-body part of the relative kinetic energy
                                // without (A-1)/A
    {
      if (jja == jjb && mmja == mmjb && mmta == mmtb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        if (nc >= 0)
          for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
          {
            for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
            {
              double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                     factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
              if (na == nb)
                x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
              else if (na == nb - 2)
                x -= phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
            }
          }

        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
            else if (na == nb + 2)
              x -= phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
          }
        }
        x = (static_cast<int>(hw_) * x / 4.0);
      }      
    }
    else if (name_ == "Rrel" || name_ == "Radius") // One-body part of the 
                            // intrinsic mass radius squared without (A-1)/A
    {
      if (jja == jjb && mmja == mmjb && mmta == mmtb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        if (nc >= 0)
          for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
          {
            for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
            {
              double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                     factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
              if (na == nb)
                x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
              else if (na == nb - 2)
                x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
            }
          }

        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
            else if (na == nb + 2)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
          }
        }
        x *= HO::oscilatorLength2(hw_, -1) / 2.0; // use proton mass
      }
    }
    else if (name_ == "Radius_p") // Point-proton radius
    {
      if (jja == jjb && mmja == mmjb && mmta == mmtb && mmta == -1)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        if (nc >= 0)
          for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
          {
            for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
            {
              double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                     factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
              if (na == nb)
                x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
              else if (na == nb - 2)
                x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
            }
          }

        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
            else if (na == nb + 2)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
          }
        }
        x *= HO::oscilatorLength2(hw_, -1) / 2.0; // use proton mass
      }
    }
    else if (name_ == "Radius_n") // Point-proton radius
    {
      if (jja == jjb && mmja == mmjb && mmta == mmtb && mmta == 1)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        if (nc >= 0)
          for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
          {
            for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
            {
              double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                     factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
              if (na == nb)
                x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
              else if (na == nb - 2)
                x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
            }
          }

        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
            else if (na == nb + 2)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
          }
        }
        x *= HO::oscilatorLength2(hw_, 1) / 2.0; // use neutron mass
      }
    }
    else if (name_ == "E2") // E2 transition operator
    {
      if (mmta == mmtb && mmtb == -1)
      {
        double factor1 = cgcs.get({jjb, 4, jja, mmjb, 0, mmja}) *
                        racah9.get({llb,4,lla, 1,0,1, jjb,4,jja});
        for (int nc = nb + 1; nc >= 0; nc -= 2)
        {
          int llc_min = 2 * (nc % 2);
          for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
          {
            double phase = ((llc + 2 - llb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor2 = std::sqrt(5.0 * (llc + 1.0) / 3.0 / (llb + 1.0)) *
                            racah.get({llc, 2, lla, 4, llb, 2});
            x += phase * factor1 * factor2 
               * HO::rRME(hw_, na, lla, nc, llc, mmta) 
               * HO::rRME(hw_, nc, llc, nb, llb, mmtb);
          }
        } 
        x *= std::sqrt(6.0);               
      }
      
      /* Old implementation
      if (mmta == mmtb && mmtb == -1)
      {
        double phase = ((lla - llb + jjb - jja) / 2 % 2 == 0) ? 1.0 : -1.0,
               factor = cgcs.get({jjb, 4, jja, mmjb, 0, mmja}) *
                        racah.get({llb, 1, 4, jja, jjb, lla});
        
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        if (nc >= 0)
          for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
          {
            double factor = racah.get({llb, 2, lla, 2, llc, 4});
            if (na == nb)
              x += factor * HO::bosonCreatorRME(na, lla, nc, llc) *
                         HO::bosonAnnihilatorRME(nc, llc, nb, llb);
            else if (na == nb - 2)
              x += factor * HO::bosonAnnihilatorRME(na, lla, nc, llc) *
                         HO::bosonAnnihilatorRME(nc, llc, nb, llb);
          }
        
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          double factor = racah.get({llb, 2, lla, 2, llc, 4});
          if (na == nb)
            x += factor * HO::bosonAnnihilatorRME(na, lla, nc, llc) *
                 HO::bosonCreatorRME(nc, llc, nb, llb);
          else if (na == nb + 2)
            x += factor * HO::bosonCreatorRME(na, lla, nc, llc) *
                 HO::bosonCreatorRME(nc, llc, nb, llb);
        }
        x *= std::sqrt(1.5) * HO::oscilatorLength2(hw_, -1);
      }
      */
    }
    else if (name_ == "M1") // M1 transition operator
    {
      if (na == nb && lla == llb && mmta == mmtb)
      {
        double gl = (mmta == 1) ? gl_n : gl_p; 
        double gs = (mmta == 1) ? gs_n : gs_p;

        double phase = ((jjb - jja) / 2 % 2 == 0) ? 1.0 : -1.0;
        x = cgcs.get({jjb, 2, jja, mmjb, 0, mmja}) *
            (gl * phase * racah.get({llb, 1, 2, jja, jjb, llb}) *
                 std::sqrt(llb * (llb + 2.0) / 4.0) +
             gs * racah.get({1, llb, 2, jja, jjb, 1}) * std::sqrt(0.75));
        x *= std::sqrt(3.0 / 4.0 / PI);
      }
    }
    else if (name_ == "FermiPlus") // Fermi for beta+
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb &&
          mmta == mmtb + 2)
        x = std::sqrt(0.75 - mmtb * (mmtb + 2.0) / 4.0);
    }
    else if (name_ == "FermiMinus") // Fermi for beta-
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb &&
          mmta == mmtb - 2)
        x = std::sqrt(0.75 - mmtb * (mmtb - 2.0) / 4.0);
    }
    else if (name_ == "GamowTellerPlus") // GT for beta+
    {
      if (na == nb && lla == llb && mmja == mmjb && mmta == mmtb + 2)
      {
        x = std::sqrt(0.75 - mmtb * (mmtb + 2.0) / 4.0) * SQRT3 *
            cgcs.get({jjb, 2, jja, mmjb, 0, mmja}) *
            racah.get({1, llb, 2, jja, jjb, 1});
      }
    }
    else if (name_ == "GamowTellerMinus") // GT for beta-
    {
      if (na == nb && lla == llb && mmja == mmjb && mmta == mmtb - 2)
      {
        x = std::sqrt(0.75 - mmtb * (mmtb - 2.0) / 4.0) * SQRT3 *
            cgcs.get({jjb, 2, jja, mmjb, 0, mmja}) *
            racah.get({1, llb, 2, jja, jjb, 1});
      }
    }
    else if (on_the_fly == false)
      x = OBME->getME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb);
    else
      throw std::invalid_argument("ME has not been implemented");

    return x;
  }

  // RME: < na lla jja || O || nb llb jjb >
  // = ME / ( < jb mjb ; J MJ | ja mja > * < 1/2 mtb ; T MT | 1/2 mta > )
  double OneBodyOperator::sphericalRME(int na, int lla, int jja, int nb,
                                       int llb, int jjb) const
  { 
    double x = 0.0;

    if (name_ == "NumberOp") // N operator
    {
      if (na == nb && lla == llb && jja == jjb)
        x = nb;
    }
    else if (name_ == "L.L") // One-body part of LL
    {
      if (na == nb && lla == llb && jja == jjb)
        x = llb * (llb + 2.0) / 4.0;
    }
    else if (name_ == "L.S") // One-body-part of LS
    {
      if (na == nb && lla == llb && jja == jjb)
        x = (jjb * (jjb + 2.0) - llb * (llb + 2.0) - 3.0) / 8.0;
    }
    else if (name_ == "CoMNumberOp") // One-body part of CoM number operator
    {
      if (na == nb && jja == jjb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));

            x += phase * factor *
                 HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                 HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
          }
        }
      }
    }
    else if (name_ == "Kinetic") // One-body part of the relative kinetic energy
                                // without (A-1)/A
    {
      if (jja == jjb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
            else if (na == nb - 2)
              x -= phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
            else if (na == nb + 2)
              x -= phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
          }
        }
      }
      x = (static_cast<int>(hw_) * x / 4.0);
    }
    else if (name_ == "Rrel" || name_ == "Radius") // One-body part of the
                            // intrinsic mass radius squared without (A-1)/A
    {
      if (jja == jjb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
            else if (na == nb - 2)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
            else if (na == nb + 2)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb);
          }
        }
      }
    }
    else
      throw std::invalid_argument("RME has not been implemented");

    return x;
  }

} // namespace JT
