#pragma once

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
  /* Class for holding one-body operator in JT-coupled scheme
   JJ TT					       (lam0 mu0) (P0 P0'P0'') 
  O  		= sum of [Ta x Tb] 
   MMJ MMT               kappa0 2L0 2S0 2T0 2MT0 2J0 2MJ0 
   where MMJ = MMJ0 and MMT = MMT0
  */
  class OneBodyOperator
  {
  public:
    int JJ, MMJ, // twice angular momentum and projection
        TT, MMT, // twice isospin and projection
        type;    // the type of this operator, see method is()

    // Constructor
    OneBodyOperator(int JJ_, int MMJ_, int TT_, int MMT_, int type_)
        : JJ(JJ_), MMJ(MMJ_), TT(TT_), MMT(MMT_), type(type_)
    { }

    // Check the type of this operator
    void is() const
    {
      switch (type)
      {
      case 1:
        std::cout << "Number operator N" << std::endl;
        break;
      case 2:
        std::cout << "One-body part of L^2" << std::endl;
        break;
      case 3:
        std::cout << "One-body part of L.S" << std::endl;
        break;
      case 4:
        std::cout << "One-body part of CoM number operator" << std::endl;
        break;
      default:
        throw std::invalid_argument("This operator has not been implemented");
      }
    }

    // ME: < na lla jja mmja mmta | O | nb llb jjb mmjb mmtb >
    double sphericalME(int na, int lla, int jja, int mmja, int mmta, int nb,
                       int llb, int jjb, int mmjb, int mmtb);

    // RME:             < na lla jja || O || nb llb jjb >
    // = ME / ( < jb mjb ; J MJ | ja mja > * < 1/2 mtb ; T MT | 1/2 mta > )
    double sphericalRME(int na, int lla, int jja, int nb, int llb, int jjb);

    // compute the coefficients in the sum
    std::vector<double>
    getTensorStrength(int kappa0, int LL0, int SS0, int na, int nb,
                      const std::vector<u3::physicalvector>& pv3as,
                      const std::vector<u3::physicalvector>& pv3bs,
                      const std::vector<u4::physicalvector>& pv4as,
                      const std::vector<u4::physicalvector>& pv4bs, int pv3,
                      const std::vector<std::tuple<int, int, int>>& tArr3,
                      const std::vector<Eigen::MatrixXd>& coeffs3, int pv4,
                      const std::vector<std::tuple<int, int, int>>& tArr4,
                      const std::vector<Eigen::MatrixXd>& coeffs4,
                      u34utils::SO3_CGC& cgcs, u34utils::SO3_Racah9& racah9);

    // expand into a sum of multi-shell one-body tensors
    std::pair<std::vector<double>, std::vector<U3xU4::OneBodyMultiShellTensor>>
    expand(int maxShell, u34utils::SO3_CGC& cgcs, u34utils::SO3_Racah9& racah9,
           u3::WigU3U2_db& wigU3U2db, u4::WigU4U3_db& wigU4U3db);
  };

} // namespace JT