#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <HOutilities.h>

namespace HO
{
  //////////// Class Table_nljmjmt /////////////

  // Constructor
  Table_nljmjmt::Table_nljmjmt(int maxShell)
  {
    int index = 0;
    for (int n = 0; n <= maxShell; n++)
      for (int l = n; l >= 0; l -= 2)
        for (int jj = 2 * l + 1; jj >= std::abs(2 * l - 1); jj -= 2)
          for (int mmj = jj; mmj >= -jj; mmj -= 2)
            for (int mmt = 1; mmt >= -1; mmt -= 2)
            {
              key_type key = {n, 2 * l, jj, mmj, mmt};
              state_to_index[key] = index;
              index_to_state.push_back(key);
              index++;
            }
    dim = index_to_state.size();
  }

  // Get index from quantum numbers
  int Table_nljmjmt::getIndex(int n, int ll, int jj, int mmj, int mmt) const
  {
    key_type key = {n, ll, jj, mmj, mmt};
    auto it = state_to_index.find(key);
    if (it == state_to_index.end())
      throw std::out_of_range("Labels not found");
    return it->second;
  }

  // Get quantum numbers from index
  std::tuple<int16_t, int16_t, int16_t, int16_t, int16_t>
  Table_nljmjmt::getLabels(int index) const
  {
    if (index < 0 || index >= static_cast<int>(dim))
      throw std::out_of_range("Index out of bounds");
    return index_to_state[index];
  }

  /////////////// Reduced matrix elements of boson operators ///////////////

  // < na lla || b+ || nb llb >
  double bosonCreatorRME(int na, int lla, int nb, int llb)
  {
    if (na == nb + 1)
    {
      double factor = 1.0 / std::sqrt(lla + 1.0);
      if (lla == llb + 2)
        return factor * std::sqrt((llb / 2.0 + 1.0) * (nb + llb / 2.0 + 3.0));
      else if (lla == llb - 2)
        return -1.0 * factor * std::sqrt((llb / 2.0) * (nb - llb / 2.0 + 2.0));
      else
        return 0.0;
    }
    else
      return 0.0;
  }

  // < na lla jja || b+ || nb llb jjb >
  double bosonCreatorRME(int na, int lla, int jja, int nb, int llb, int jjb,
                         qcx_utils::SO3_Racah9& racah9)
  {
    if (na == nb + 1)
    {
      // double phase = ((lla - llb + jjb - jja) / 2 % 2 == 0) ? 1.0 : -1.0;
      // double factor = phase * racah.compute(llb,1,2,jja,jjb,lla) /
      // std::sqrt(lla + 1.0);
      double factor = racah9.compute(llb, 2, lla, 1, 0, 1, jjb, 2, jja);
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
        return factor * std::sqrt((llb / 2.0 + 1.0) * (nb - llb / 2.0));
      else if (lla == llb - 2)
        return -1.0 * factor * std::sqrt((llb / 2.0) * (nb + llb / 2.0 + 1.0));
      else
        return 0.0;
    }
    else
      return 0.0;
  }

  // < na lla jja || b- || nb llb jjb >
  double bosonAnnihilatorRME(int na, int lla, int jja, int nb, int llb, int jjb,
                             qcx_utils::SO3_Racah9& racah9)
  {
    if (na == nb - 1)
    {
      // double phase = ((lla - llb + jjb - jja) / 2 % 2 == 0) ? 1.0 : -1.0;
      // double factor = phase * racah.compute(llb,1,2,jja,jjb,lla) /
      // std::sqrt(lla + 1.0);
      double factor = racah9.compute(llb, 2, lla, 1, 0, 1, jjb, 2, jja);
      return factor * bosonAnnihilatorRME(na, lla, nb, llb);
    }
    else
      return 0;
  }

  ////////////// Class Orbit ///////////////

  bool Orbit::operator==(const Orbit& other) const
  {
    return other.n == n && other.l == l && other.jj == jj && other.mmt == mmt;
  }

  bool Orbit::operator<(const Orbit& other) const
  {
    return std::tie(n, l, jj, mmt) <
           std::tie(other.n, other.l, other.jj, other.mmt);
  }

  //////////// Class OrbitsMap //////////////

  // Constructor
  OrbitsMap::OrbitsMap(uint16_t emax_)
  {
    // It does not take in lmax, it is set to default lmax = emax
    emax = emax_;

    int idx = 1;
    for (int e = 0; e <= emax; ++e)
    {
      for (int l = e % 2; l <= e; l += 2)
      {
        int n = (e - l) / 2;
        for (int jj = abs(2 * l - 1); jj <= 2 * l + 1; jj += 2)
        {
          // add proton
          orbits.push_back({n, l, jj, -1});
          index_map[{n, l, jj, -1}] = idx;
          idx++;

          // add neutron
          orbits.push_back({n, l, jj, 1});
          index_map[{n, l, jj, 1}] = idx;
          idx++;
        }
      }
    }
    num_orbits = orbits.size();
  }

  // Get index of the orbit
  // Caution: n = (e - l)/2 is the nodal quantum number
  // whereas e is the shell number
  int OrbitsMap::getIndex(int n, int l, int jj, int mmt) const
  {
    auto it = index_map.find({n, l, jj, mmt});
    if (it != index_map.end())
      return it->second;
    else
      throw std::runtime_error("Orbit not found");
  }

  //////////// Class TBME_scalar /////////////

  TBME_scalar::TBME_scalar(const std::string& name)
  {
    if (name == "N3LO-bare")
    {
      orbit_map = OrbitsMap(14);
#ifdef READ_FROM_BINARY
      read_from_binary(
          "data/Ab-initio_Interactions/"
          "TwBME-HO_NN-only_N3LO_EM500_bare_hw16_emax14_e2max28.myg.bin");
#else
      read_from_file(
          "data/Ab-initio_Interactions/"
          "TwBME-HO_NN-only_N3LO_EM500_bare_hw16_emax14_e2max28.myg");
#endif
    }
  }

  /* Reading method is based on 
  	 subroutine read_scalar_operator_binary_myg(this, f)
  	 in NuHamil-public/src/TwoBody/TwoBodyLabOps.F90
  */
// #define CHECKING_2B_CHANNELS
  void TBME_scalar::read_from_binary(const std::string& filename)
  {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile)
      throw std::runtime_error("Failed to open binary file: " + filename);

    // Read number of matrix elements
    infile.read(reinterpret_cast<char*>(&num_me), sizeof(int));
    if (!infile)
      throw std::runtime_error("Failed to read count from file: " + filename);

    // Allocate temporary buffers
    std::vector<uint> ia(num_me), ib(num_me), ic(num_me), id(num_me), J(num_me);
    std::vector<double> v(num_me);

    // auto t0 = std::chrono::high_resolution_clock::now();

    // Read data arrays
    infile.read(reinterpret_cast<char*>(ia.data()), num_me * sizeof(uint));
    infile.read(reinterpret_cast<char*>(ib.data()), num_me * sizeof(uint));
    infile.read(reinterpret_cast<char*>(ic.data()), num_me * sizeof(uint));
    infile.read(reinterpret_cast<char*>(id.data()), num_me * sizeof(uint));
    infile.read(reinterpret_cast<char*>(J.data()), num_me * sizeof(uint));
    infile.read(reinterpret_cast<char*>(v.data()), num_me * sizeof(double));

    // auto t1 = std::chrono::high_resolution_clock::now();

    if (!infile)
      throw std::runtime_error("Failed to read matrix elements from file: " +
                               filename);

    // Populate the map

#ifdef CHECKING_2B_CHANNELS
  	int violate_isospin = 0, violate_parity = 0, np_cnt = 0, 
  		violate_Pauli = 0, double_cnt = 0;
#endif

    me_map.reserve(num_me);
    for (int i = 0; i < num_me; ++i)
    {
#ifdef CHECKING_2B_CHANNELS
      const auto& oa = orbit_map.getOrbit(ia[i]),
      	    	  ob = orbit_map.getOrbit(ib[i]),
      	    	  oc = orbit_map.getOrbit(ic[i]),
      	    	  od = orbit_map.getOrbit(id[i]);

      if (oa.mmt + ob.mmt != oc.mmt + od.mmt) violate_isospin++;
      if ((oa.l + ob.l + oc.l + od.l) % 2 != 0) violate_parity++;
      if ((oa.mmt == 1 && ob.mmt == -1) || 
      	  (oc.mmt == 1 && od.mmt == -1)) np_cnt++;
      if ((ia[i] == ib[i] && J[i] % 2 == 1) || 
      	  (ic[i] == id[i] && J[i] % 2 == 1)) violate_Pauli++;
      if ((abs(oa.mmt + ob.mmt) == 2 && ia[i] < ib[i]) ||
      	  (abs(oc.mmt + od.mmt) == 2 && ic[i] < id[i])) double_cnt++;
#endif

      me_map[{static_cast<uint16_t>(ia[i]), static_cast<uint16_t>(ib[i]),
              static_cast<uint16_t>(ic[i]), static_cast<uint16_t>(id[i]),
              static_cast<uint16_t>(J[i])}] = v[i];

  	  /* Comment on the phase: in the subroutine, one sees these lines
  	  	 ph = ms%jpz(ch)%GetPhase(a,b) * ms%jpz(ch)%GetPhase(c,d)
  	  	 this%MatCh(ch,ch)%m(bra,ket) = v * ph
  	  	 Do we need to include this? NO! Why?
		 Checking the initialization of TwoBodyLabPNChan, one realizes:

		 i) For pp or nn channels, the first index (i1) is always >= the second
		 index (i2), and there is no interchange when printing out.
		 Hence, there is only phase(i1 >= i2) = 1 when reading back.

		 ii) There are no np channels, only pn ones. We can think that i1 is ip
		 and i2 is in, thus phase(ip,in) = 1 while phase(in,ip) = -(-)^(jp+jn-J). 
		 When reading back, there is no np channels, so we always have phase 1. 
  	  */
    }

    num_me = me_map.size();

    // auto t2 = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    // std::chrono::duration<double> duration1 = t1 - t0;
    // std::chrono::duration<double> duration2 = t2 - t1;

    // Output the time in seconds
    // std::cout << "Reading time: " << duration1.count()/60.0 << " minutes" 
    // 			 << std::endl;
    // std::cout << "Mapping time: " << duration2.count()/60.0 << " minutes" 
    // 			 << std::endl;

#ifdef CHECKING_2B_CHANNELS
    std::cout << "There are " << violate_isospin << " isospin violations" 
    		  << std::endl;
    std::cout << "There are " << violate_parity << " parity violations" 
    		  << std::endl;
    std::cout << "There are " << np_cnt << " np channels" << std::endl;
    std::cout << "There are " << violate_Pauli 
    		  << " channels violating Pauli" << std::endl;
    std::cout << "There are " << double_cnt 
    		  << " pp or nn channels counted twice" << std::endl;
#endif

   	return;
  }

  void TBME_scalar::read_from_file(const std::string& filename)
  {
    std::ifstream infile(filename);
    if (!infile)
      throw std::runtime_error("Failed to open file: " + filename);

    std::string line;

    // Skip first header line
    std::getline(infile, line);

    // Get number of matrix elements
    infile >> num_me;
    me_map.reserve(num_me);
    // Skip the rest of that line
    std::getline(infile, line);

    // Skip label line
    std::getline(infile, line);

    // Read all matrix elements
    uint a, b, c, d, J;
    double v;

#ifdef CHECKING_2B_CHANNELS
  	int violate_isospin = 0, violate_parity = 0, np_cnt = 0, 
  		violate_Pauli = 0, double_cnt = 0;
#endif

    while (infile >> a >> b >> c >> d >> J >> v)
    {
#ifdef CHECKING_2B_CHANNELS
      const auto& oa = orbit_map.getOrbit(a),
      	    	  ob = orbit_map.getOrbit(b),
      	    	  oc = orbit_map.getOrbit(c),
      	    	  od = orbit_map.getOrbit(d);

      if (oa.mmt + ob.mmt != oc.mmt + od.mmt) violate_isospin++;
      if ((oa.l + ob.l + oc.l + od.l) % 2 != 0) violate_parity++;
      if ((oa.mmt == 1 && ob.mmt == -1) || 
      	  (oc.mmt == 1 && od.mmt == -1)) np_cnt++;
      if ((a == b && J % 2 == 1) || 
      	  (c == d && J % 2 == 1)) violate_Pauli++;
      if ((abs(oa.mmt + ob.mmt) == 2 && a < b) ||
      	  (abs(oc.mmt + od.mmt) == 2 && c < d)) double_cnt++;
#endif

      me_map[{static_cast<uint16_t>(a), static_cast<uint16_t>(b),
              static_cast<uint16_t>(c), static_cast<uint16_t>(d),
              static_cast<uint16_t>(J)}] = v;
    }

#ifdef CHECKING_2B_CHANNELS
    std::cout << "There are " << violate_isospin << " isospin violations" 
    		  << std::endl;
    std::cout << "There are " << violate_parity << " parity violations" 
    		  << std::endl;
    std::cout << "There are " << np_cnt << " np channels" << std::endl;
    std::cout << "There are " << violate_Pauli 
    		  << " channels violating Pauli" << std::endl;
    std::cout << "There are " << double_cnt 
    		  << " pp or nn channels counted twice" << std::endl;
#endif

    return;
  }

  // Get ME from channels
  double TBME_scalar::getME(uint16_t a, uint16_t b, uint16_t c, uint16_t d,
                            uint16_t J) const
  {
    auto it1 = me_map.find({a, b, c, d, J}),
    	 it2 = me_map.find({b, a, c, d, J}),
    	 it3 = me_map.find({a, b, d, c, J}),
    	 it4 = me_map.find({b, a, d, c, J});

    if (it1 != me_map.end())
      return it1->second;
    else if (it2 != me_map.end())
    {
      const auto& oa = orbit_map.getOrbit(a),
      	    	  ob = orbit_map.getOrbit(b);
      double phase = (((oa.jj + ob.jj)/2 - J) % 2 == 0) ? -1.0 : 1.0;
      return phase * (it2->second);
    }
    else if (it3 != me_map.end())
    {
      const auto& oc = orbit_map.getOrbit(c),
      	    	  od = orbit_map.getOrbit(d);
      double phase = (((oc.jj + od.jj)/2 - J) % 2 == 0) ? -1.0 : 1.0;
      return phase * (it3->second);
    }
    else if (it4 != me_map.end())
    {
      const auto& oa = orbit_map.getOrbit(a),
      	    	  ob = orbit_map.getOrbit(b),
      	    	  oc = orbit_map.getOrbit(c),
      	    	  od = orbit_map.getOrbit(d);

      double phase = (((oa.jj + ob.jj + oc.jj + od.jj)/2) % 2 == 0) ? -1.0 : 1.0;
      return phase * (it4->second);
    }
    else
	{
 	  try 
 	  {
 	    return getME(c, d, a, b, J);
 	  }
   	  catch (const std::out_of_range&) 
   	  {
    	throw std::out_of_range("TBME not found in any permutation");
      }
	}
  }
} // namespace HO
