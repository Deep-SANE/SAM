#include <cmath>
#include <cstdint>

#include <HObasics.h>

#include <convert.h>
#include <print.h>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Fano.h>

#define POSITIVE_INFINITY
//#define POSITIVE_ORIGIN

namespace HO
{
  /////////////// DETERMINING VALENCE SHELL ///////////////
  int valenceShell(int A)
  {
    int num_sps = 0, shell = 0;
    while (num_sps < A)
    {
      num_sps += 2 * (shell + 1) * (shell + 2);
      shell++;
    }
    return shell - 1;
  }

  /////////////// DETERMINING OSCILLATOR STRENGTH ///////////////
  double oscilatorLength2(int hw, int mmt)
  {
    double mass = (mmt == 1) ? mass_n : mass_p;
    return hbar * hbar / mass / static_cast<double>(hw);
  }

  double oscilatorLength(int hw, int mmt)
  {
    return std::sqrt(oscilatorLength2(hw, mmt));
  }

  /////////////// RME OF BOSON OPERATORS ///////////////

  // < na lla || b+ || nb llb >
  double bosonCreatorRME(int na, int lla, int nb, int llb)
  {
    if (na == nb + 1)
    {
      double factor = 1.0 / std::sqrt(lla + 1.0);
      if (lla == llb + 2)
        return factor * std::sqrt((llb / 2.0 + 1.0) * (nb + llb / 2.0 + 3.0));
      else if (lla == llb - 2)
      {
#ifdef POSITIVE_INFINITY
        return -1.0 * factor * std::sqrt((llb / 2.0) * (nb - llb / 2.0 + 2.0));
#endif
#ifdef POSITIVE_ORIGIN
        return factor * std::sqrt((llb / 2.0) * (nb - llb / 2.0 + 2.0));
#endif
      }
      else
        return 0.0;
    }
    else
      return 0.0;
  }

  // < na lla jja || b+ || nb llb jjb >
  double bosonCreatorRME(int na, int lla, int jja, 
    int nb, int llb, int jjb)
  {
    if (na == nb + 1)
    {
      double factor = utils::fanos().get({llb, 2, lla, 1, 0, 1, jjb, 2, jja});
      return factor * bosonCreatorRME(na, lla, nb, llb);
    }
    else
      return 0;
  }

  // < na lla || b- || nb llb >
  double bosonAnnihilatorRME(int na, int lla, int nb, int llb)
  {
    if (na == nb - 1)
    {
      double factor = 1.0 / std::sqrt(lla + 1.0);
      if (lla == llb + 2)
      {
#ifdef POSITIVE_INFINITY
        return factor * std::sqrt((llb / 2.0 + 1.0) * (nb - llb / 2.0));
#endif
#ifdef POSITIVE_ORIGIN
        return -1.0 * factor * std::sqrt((llb / 2.0 + 1.0) * (nb - llb / 2.0));
#endif
      }
      else if (lla == llb - 2)
        return -1.0 * factor * std::sqrt((llb / 2.0) * (nb + llb / 2.0 + 1.0));
      else
        return 0.0;
    }
    else
      return 0.0;
  }

  // < na lla jja || b- || nb llb jjb >
  double bosonAnnihilatorRME(int na, int lla, int jja, 
    int nb, int llb, int jjb)
  {
    if (na == nb - 1)
    {
      double factor = utils::fanos().get({llb, 2, lla, 1, 0, 1, jjb, 2, jja});
      return factor * bosonAnnihilatorRME(na, lla, nb, llb);
    }
    else
      return 0;
  }

  /////////////// RME OF POSITION & MOMENTUM OPERATORS ///////////////

  // < na lla || r || nb llb >
  double rRME(int hw, int na, int lla, int nb, int llb, int mmt)
  {
    return (bosonCreatorRME(na, lla, nb, llb) + 
            bosonAnnihilatorRME(na, lla, nb, llb)) *
            oscilatorLength(hw, mmt) / SQRT2;
  }

  // < na lla jja || r || nb llb jjb >
  double rRME(int hw, int na, int lla, int jja, 
    int nb, int llb, int jjb, int mmt)
  {
    return (bosonCreatorRME(na, lla, jja, nb, llb, jjb) + 
            bosonAnnihilatorRME(na, lla, jja, nb, llb, jjb)) *
            oscilatorLength(hw, mmt) / SQRT2;
  }

  // < na lla || p || nb llb > defined without i
  double pRME(int hw, int na, int lla, int nb, int llb, int mmt)
  {
    return (bosonCreatorRME(na, lla, nb, llb) - 
            bosonAnnihilatorRME(na, lla, nb, llb)) *
            hbar / (oscilatorLength(hw, mmt) * SQRT2);
  }

  // < na lla jja || p || nb llb jjb > defined without i
  double pRME(int hw, int na, int lla, int jja, 
    int nb, int llb, int jjb, int mmt)
  {
    return (bosonCreatorRME(na, lla, jja, nb, llb, jjb) - 
            bosonAnnihilatorRME(na, lla, jja, nb, llb, jjb)) *
            hbar / (oscilatorLength(hw, mmt) * SQRT2);
  }
} // namespace HO
