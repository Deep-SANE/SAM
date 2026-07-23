#pragma once

#include <cstdint>

#include <GlobalConstants.h>

namespace HO
{
  /////////////// DETERMINING VALENCE SHELL ///////////////
  int valenceShell(int A);

  /////////////// DETERMINING OSCILLATOR STRENGTH ///////////////
  
  // mmt = 1 for neutron, -1 for proton
  double oscilatorLength2(int hw, int mmt); // squared
  double oscilatorLength(int hw, int mmt);

  /////////////// RME OF BOSON OPERATORS ///////////////

  // < na lla || b+ || nb llb >
  double bosonCreatorRME(int na, int lla, int nb, int llb);

  // < na lla jja || b+ || nb llb jjb >
  double bosonCreatorRME(int na, int lla, int jja, 
    int nb, int llb, int jjb);

  // < na lla || b- || nb llb >
  double bosonAnnihilatorRME(int na, int lla, int nb, int llb);

  // < na lla jja || b- || nb llb jjb >
  double bosonAnnihilatorRME(int na, int lla, int jja, 
    int nb, int llb, int jjb);

  /////////////// RME OF POSITION & MOMENTUM OPERATORS ///////////////

  // mmt = 1 for neutron, -1 for proton

  // < na lla || r || nb llb >
  double rRME(int hw, int na, int lla, int nb, int llb, int mmt);

  // < na lla jja || r || nb llb jjb >
  double rRME(int hw, int na, int lla, int jja, 
    int nb, int llb, int jjb, int mmt);

  // < na lla || p || nb llb > defined without i
  double pRME(int hw, int na, int lla, int nb, int llb, int mmt);

  // < na lla jja || p || nb llb jjb > defined without i
  double pRME(int hw, int na, int lla, int jja, 
    int nb, int llb, int jjb, int mmt);
} // namespace HO
