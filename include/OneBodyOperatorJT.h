#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

#include <U3SO3xU3SO3.h>
#include <U3U2xU3U2.h>

#include <U4SU2SU2xU4SU2SU2.h>
#include <U4U3xU4U3.h>

#include <U3xU4Irrep.h>
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
  private:
    int8_t JJ, MMJ, // twice angular momentum and projection
        TT, MMT;    // twice isospin and projection

    bool parity; // parity, true = +1, false = -1

    bool USE_ONEBODY_SPHERICAL_RME = true;

    std::string name;
    int8_t hw, maxShell;

  public:
    // Constructor
    OneBodyOperator(const std::string& name_, int8_t hw_, int8_t maxShell_);

    void is() const;
    void GetJJ() const
    {
      std::cout << "2J=" << JJ << ", 2MJ=" << MMJ << std::endl;
    }
    void GetTT() const
    {
      std::cout << "2T=" << TT << ", 2MT=" << MMT << std::endl;
    }
    void GetParity() const
    {
      std::cout << "P=" << ((parity == true) ? 1 : -1) << std::endl;
    }

    // ME: < na lla jja mmja mmta | O | nb llb jjb mmjb mmtb >
    double sphericalME(int na, int lla, int jja, int mmja, int mmta, int nb,
                       int llb, int jjb, int mmjb, int mmtb,
                       qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racah,
                       qcx_utils::SO3_Racah9& racah9) const;

    // RME:             < na lla jja || O || nb llb jjb >
    // = ME / ( < jb mjb ; J MJ | ja mja > * < 1/2 mtb ; T MT | 1/2 mta > )
    double sphericalRME(int na, int lla, int jja, int nb, int llb, int jjb,
                        qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racah,
                        qcx_utils::SO3_Racah9& racah9) const;

    // compute the coefficients in the sum
    std::pair<bool, std::vector<double>> getTensorStrength(
        int lam0, int mu0, int P10, int P20, int P30, int kappa0, int LL0,
        int SS0, int na, int nb, qcx_utils::SO3_CGC& cgcs,
        qcx_utils::SO3_Racah& racah, qcx_utils::SO3_Racah9& racah9,
        u3::WigU3SO3_db& wigU3SO3db, u4::WigU4SU2SU2_db& wigU4SU2SU2db) const;

    // expand into a sum of multi-shell one-body tensors
    std::vector<U3xU4::OneBodyMultiShellTensor>
    expand(qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racah,
           qcx_utils::SO3_Racah9& racah9, u3::WigU3SO3_db& wigU3SO3db,
           u4::WigU4SU2SU2_db& wigU4SU2SU2db) const;
  };

} // namespace JT
