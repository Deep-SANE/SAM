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
                       int mmjd, int mmtd);

    // Generate a list of al {na, nb, nc, nd} combinations such that
    // 0 <= na, nb, nc, nd <= maxShell
    std::vector<std::array<int, 4>> GenerateAllCombinationsHOShells(int maxShell);
    // Restrict: 
    // 0 <= (na - valenceShell + (nb - valenceShell) <= Nmax
    // 0 <= (nc - valenceShell + (nd - valenceShell) <= Nmax
    std::vector<std::array<int, 4>> GenerateAllCombinationsHOShells(int Nmax, 
    											int valenceShell, int maxShell);

    // expand into a sum of multi-shell two-body tensors
    std::vector<U3xU4::TwoBodyMultiShellTensor>
    expand(int Nmax, int valenceShell, int maxShell, 
    	   u34utils::SO3_CGC& cgcs, 
    	   u34utils::SO3_Racah9& racah9, 
    	   u3::WigU3U2_db& wigU3U2db, 
    	   u4::WigU4U3_db& wigU4U3db);
  };

} // namespace JT