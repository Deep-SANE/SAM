#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <omp.h>

#include <convert.h>
#include <print.h>
using utils::print;
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>
#include <so3Coeffs/SO3_Fano.h>

#include <sam.h>

#include <TwoBodyOperatorJT.h>

namespace JT
{
  // ME: / na lla jja mmja mmta | O | nc llc jjc mmjc mmtc \
  // 	   \ nb llb jjb mmjb mmtb |	  | nd lld jjb mmjd mmtd /
  double TwoBodyOperator::sphericalME(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb, int nc, int llc,
                                      int jjc, int mmjc, int mmtc, int nd,
                                      int lld, int jjd, int mmjd, int mmtd) const
  {
    auto& cgcs = utils::cgcs();
    auto& racah = utils::racahs();
    auto& racah9 = utils::fanos();

    double x = 0.0;

    if (name_ == "L.L") // Two-body part of L.L
    {
      if (na == nc && nb == nd && lla == llc && llb == lld && mmta == mmtc &&
          mmtb == mmtd)
      {
        double phase = ((jja + jjb + jjc + jjd) / 2 % 2 == 0) ? 1.0 : -1.0,
               factor = std::sqrt(llc * (llc + 2.0) * lld * (lld + 2.0)) / 4.0,
               U1 = racah.get({llc, 1, 2, jja, jjc, llc}),
               U2 = racah.get({lld, 1, 2, jjb, jjd, lld}), cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = 2 * phase * factor * U1 * U2 * cgc;
      }
    }
    else if (name_ == "L.S") // Two-body part of L.S
    {
      if (na == nc && nb == nd && lla == llc && llb == lld && mmta == mmtc &&
          mmtb == mmtd)
      {
        double U1 = racah.get({llc, 1, 2, jja, jjc, llc}),
               U2 = racah.get({1, lld, 2, jjb, jjd, 1}),
               U3 = racah.get({lld, 1, 2, jjb, jjd, lld}),
               U4 = racah.get({1, llc, 2, jja, jjc, 1}),
               phase12 = ((jja - jjc) / 2 % 2 == 0) ? 1.0 : -1.0,
               phase34 = ((jjb - jjd) / 2 % 2 == 0) ? 1.0 : -1.0,
               factor =
                   (phase12 * std::sqrt(llc * (llc + 2.0) / 4.0) * U1 * U2 +
                    phase34 * std::sqrt(lld * (lld + 2.0) / 4.0) * U3 * U4),
               cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = std::sqrt(0.75) * cgc * factor;
      }
    }
    else if (name_ ==
             "CoMNumberOp") // Two-body part of CoM number operator without 1/A
    {
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = cgc *
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                 HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                 HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd));
      }
    }
    else if (name_ == "Kinetic") // Two-body part of the relative kinetic energy
                                // without 1/A
    {
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = static_cast<double>(hw_) * cgc *
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                 (HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd) -
                  HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd)) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                 (HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd) -
                  HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd))) /
            2.0;
      }
    }
    else if (name_ == "Rrel") // Two-body part of the intrinsic mass radius
                            // squared without -1/A
    {
      if (mmta == mmtc && mmtb == mmtd)
      {
        double cgc = 0.0;
        for (int mu = -1; mu <= 1; mu++)
          cgc += (mu == 0 ? 1.0 : -1.0) *
                 cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
                 cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
        x = HO::oscilatorLength2(hw_, -1) * cgc *
            (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
                 (HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd) +
                  HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd)) +
             HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
                 (HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd) +
                  HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd)));
      }
    }
    else
      throw std::invalid_argument("This operator has not been implemented");

    return x;
  }

  // Antisymmetrized ME
  double TwoBodyOperator::sphericalME_AS(int na, int lla, int jja, int mmja,
                                         int mmta, int nb, int llb, int jjb,
                                         int mmjb, int mmtb, int nc, int llc,
                                         int jjc, int mmjc, int mmtc, int nd,
                                         int lld, int jjd, int mmjd, int mmtd) const
  {
    assert(on_the_fly == true);
    return (sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                        nc, llc, jjc, mmjc, mmtc, nd, lld, jjd, mmjd, mmtd) -
            sphericalME(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                        nd, lld, jjd, mmjd, mmtd, nc, llc, jjc, mmjc, mmtc)) / 4.0;
  }

} // namespace JT
