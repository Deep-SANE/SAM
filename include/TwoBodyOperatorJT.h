#pragma once

#include <array>
#include <stdexcept>
#include <utility>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

#include <U3basis.h>
#include <U3SO3xU3SO3.h>
#include <U3U2xU3U2.h>

#include <U4basis.h>
#include <U4SU2SU2xU4SU2SU2.h>
#include <U4U3xU4U3.h>

#include <U3xU4MultiShellTensors.h>

namespace JT
{
  class TwoBodyOperator
  {
  public:
    int JJ, MMJ, // twice angular momentum and projection
        TT, MMT, // twice isospin and projection
        type;    // the type of this operator, see method is()

    // Constructor
    TwoBodyOperator(int JJ_, int MMJ_, int TT_, int MMT_, int type_)
        : JJ(JJ_), MMJ(MMJ_), TT(TT_), MMT(MMT_), type(type_)
    { }

    // Check the type of this operator
    void is() const
    {
      switch (type)
      {
      case 1:
        std::cout << "Two-body part of L^2" << std::endl;
        break;
      case 2:
        std::cout << "Two-body part of L.S" << std::endl;
        break;
      case 3:
        std::cout << "Two-body part of CoM number operator" << std::endl;
        break;
      default:
        throw std::invalid_argument("This operator has not been implemented");
      }
    }

    // ME: / na lla jja mmja mmta | O | nc llc jjc mmjc mmtc \
    // 	   \ nb llb jjb mmjb mmtb |	  | nd lld jjb mmjd mmtd /
    double sphericalME(int na, int lla, int jja, int mmja, int mmta, int nb,
                       int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                       int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                       int mmjd, int mmtd, u34utils::SO3_CGC& cgcs,
                       u34utils::SO3_Racah& racah) const;
    // Antisymmetrized matrix element
    double sphericalME_AS(int na, int lla, int jja, int mmja, int mmta, int nb,
                       int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                       int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                       int mmjd, int mmtd, u34utils::SO3_CGC& cgcs,
                       u34utils::SO3_Racah& racah) const;

    // Generate a list of al {na, nb, nc, nd} combinations such that
    // 0 <= na, nb, nc, nd <= maxShell
    std::vector<std::array<int, 4>>
    GenerateAllCombinationsHOShells(int maxShell) const;
    // Restrict:
    // 0 <= (na - valenceShell + (nb - valenceShell) <= Nmax
    // 0 <= (nc - valenceShell + (nd - valenceShell) <= Nmax
    std::vector<std::array<int, 4>>
    GenerateAllCombinationsHOShells(int Nmax, int valenceShell,
                                    int maxShell) const;

    // Set coefficient for two-body multi-shell tensor.
    // It returns a flag, which indicate whether this tensor can be ignored.
    bool setCoefficient(U3xU4::TwoBodyMultiShellTensor& tensor, double phase,
                        const std::vector<int>& inverse,
                        u34utils::SO3_CGC& cgcs, u34utils::SO3_Racah& racah,
                        u34utils::SO3_Racah9& racah9,
                        u3::WigU3SO3_db& wigU3SO3db,
                        u4::WigU4SU2SU2_db& wigU4SU2SU2db) const;

    // expand into a sum of multi-shell two-body tensors
    std::vector<U3xU4::TwoBodyMultiShellTensor>
    expand(int Nmax, int valenceShell, int maxShell, u34utils::SO3_CGC& cgcs,
           u34utils::SO3_Racah& racah, u34utils::SO3_Racah9& racah9,
           u3::WigU3SO3_db& wigU3SO3db,
           u4::WigU4SU2SU2_db& wigU4SU2SU2db) const;
  };

} // namespace JT