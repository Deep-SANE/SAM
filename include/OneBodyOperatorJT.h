#pragma once

#include <cstdint>
#include <string>
#include <utility>

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
                       sam::Databases& dbs) const;

    // RME:             < na lla jja || O || nb llb jjb >
    // = ME / ( < jb mjb ; J MJ | ja mja > * < 1/2 mtb ; T MT | 1/2 mta > )
    double sphericalRME(int na, int lla, int jja, int nb, int llb, int jjb,
                        sam::Databases& dbs) const;

    // compute the coefficients in the sum
    std::pair<bool, std::vector<double>> getTensorStrength(
        int lam0, int mu0, int P10, int P20, int P30, int kappa0, int LL0,
        int SS0, int na, int nb,
        sam::Databases& dbs) const;

    // expand into a sum of multi-shell one-body tensors
    std::vector<U3xU4::OneBodyMultiShellTensor>
    expand(sam::Databases& dbs) const;
  };

} // namespace JT
