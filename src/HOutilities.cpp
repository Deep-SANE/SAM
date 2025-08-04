#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <GlobalConstants.h>
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

  TBME_scalar::TBME_scalar()
      : me_map(),
        orbit_map(),
        num_me(0),
        name(""),
        hw(0),
        Nmax(0),
        valenceShell(0),
        maxShell(0)
  { }

  TBME_scalar::TBME_scalar(const std::string& name_, int8_t hw_, int8_t Nmax_,
                           int8_t valenceShell_, int8_t maxShell_)
  {
    if (maxShell_ > 14)
      throw std::out_of_range(
          "Sorry, we only have maxShell up to 14 at the moment");

    hw = (hw_ >= 0) ? hw_ : 20;
    Nmax = (Nmax_ >= 0) ? Nmax_ : 14;
    valenceShell = (valenceShell_ > 0) ? valenceShell_ : 0;
    maxShell = (maxShell_ >= 0) ? maxShell_ : 14;

    orbit_map = OrbitsMap(maxShell_);

    const std::unordered_set<std::string> known_potentials = {
        "N2LO_EMN500_bare", "N3LO_EMN500_bare", "N4LO_EMN500_bare",
        "N3LO_EM500_bare",  "N2LO_opt_bare",    "N2LO_sat_bare"};

    if (known_potentials.count(name_))
    {
      name = name_;

      const char* nuhamil_root = std::getenv("NUHAMIL_ROOT");    
      if (!nuhamil_root)
        throw std::runtime_error(
          "Environment variable NUHAMIL_ROOT is not set");

      std::string filename = std::string(nuhamil_root) + 
          "/exe/TwBME-HO_NN-only_" + name + "_hw" +
          std::to_string(static_cast<int>(hw)) + "_emax14_e2max28.myg";

#ifdef READ_FROM_BINARY
      filename += ".bin";
      read_from_binary(filename);
#else
      read_from_file(filename);
#endif
    }
    else
    {
      throw std::runtime_error(
          "Name is incorrect or 2-body Hamiltonian is not implemented");
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
    int violate_isospin = 0, violate_parity = 0, np_cnt = 0, violate_Pauli = 0,
        double_cnt = 0;
#endif
    me_map.reserve(num_me);
    for (int i = 0; i < num_me; ++i)
    {
      const auto &oa = orbit_map.getOrbit(ia[i]),
                 ob = orbit_map.getOrbit(ib[i]), oc = orbit_map.getOrbit(ic[i]),
                 od = orbit_map.getOrbit(id[i]);

      int ea = 2 * oa.n + oa.l, eb = 2 * ob.n + ob.l, ec = 2 * oc.n + oc.l,
          ed = 2 * od.n + od.l;
      if (ea > maxShell || eb > maxShell || ec > maxShell || ed > maxShell)
        continue;

      int ab_Nhw = 0;
      if (ea > valenceShell)
        ab_Nhw += (ea - valenceShell);
      if (eb > valenceShell)
        ab_Nhw += (eb - valenceShell);
      if (ab_Nhw > Nmax)
        continue;

      int cd_Nhw = 0;
      if (ec > valenceShell)
        cd_Nhw += (ec - valenceShell);
      if (ed > valenceShell)
        cd_Nhw += (ed - valenceShell);
      if (cd_Nhw > Nmax)
        continue;

#ifdef CHECKING_2B_CHANNELS
      if (oa.mmt + ob.mmt != oc.mmt + od.mmt)
        violate_isospin++;
      if ((oa.l + ob.l + oc.l + od.l) % 2 != 0)
        violate_parity++;
      if ((oa.mmt == 1 && ob.mmt == -1) || (oc.mmt == 1 && od.mmt == -1))
        np_cnt++;
      if ((ia[i] == ib[i] && J[i] % 2 == 1) ||
          (ic[i] == id[i] && J[i] % 2 == 1))
        violate_Pauli++;
      if ((abs(oa.mmt + ob.mmt) == 2 && ia[i] < ib[i]) ||
          (abs(oc.mmt + od.mmt) == 2 && ic[i] < id[i]))
        double_cnt++;
#endif

      me_map[{static_cast<uint16_t>(ia[i]), static_cast<uint16_t>(ib[i]),
              static_cast<uint16_t>(ic[i]), static_cast<uint16_t>(id[i]),
              static_cast<uint16_t>(J[i])}] = v[i];

      /* Comment on the phase: in the subroutine, one sees these lines
             ph = ms%jpz(ch)%GetPhase(a,b) * ms%jpz(ch)%GetPhase(c,d)
             this%MatCh(ch,ch)%m(bra,ket) = v * ph
             Do we need to include this? NO! Why?
             Checking the initialization of TwoBodyLabPNChan, one realizes:

             i) For pp or nn channels, the first index (i1) is always >= the
         second index (i2), and there is no interchange when printing out.
             Hence, there is only phase(i1 >= i2) = 1 when reading back.

             ii) There are no np channels, only pn ones. We can think that i1 is
         ip and i2 is in, thus phase(ip,in) = 1 while phase(in,ip) =
         -(-)^(jp+jn-J). When reading back, there is no np channels, so we
         always have phase 1.
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
    std::cout << "There are " << violate_Pauli << " channels violating Pauli"
              << std::endl;
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
    int violate_isospin = 0, violate_parity = 0, np_cnt = 0, violate_Pauli = 0,
        double_cnt = 0;
#endif

    while (infile >> a >> b >> c >> d >> J >> v)
    {
      const auto &oa = orbit_map.getOrbit(a), ob = orbit_map.getOrbit(b),
                 oc = orbit_map.getOrbit(c), od = orbit_map.getOrbit(d);
      int ea = 2 * oa.n + oa.l, eb = 2 * ob.n + ob.l;
      int ab_Nhw = 0;
      if (ea > valenceShell)
        ab_Nhw += (ea - valenceShell);
      if (eb > valenceShell)
        ab_Nhw += (eb - valenceShell);
      if (ab_Nhw > Nmax)
        continue;

      int ec = 2 * oc.n + oc.l, ed = 2 * od.n + od.l;
      int cd_Nhw = 0;
      if (ec > valenceShell)
        cd_Nhw += (ec - valenceShell);
      if (ed > valenceShell)
        cd_Nhw += (ed - valenceShell);
      if (cd_Nhw > Nmax)
        continue;

#ifdef CHECKING_2B_CHANNELS
      if (oa.mmt + ob.mmt != oc.mmt + od.mmt)
        violate_isospin++;
      if ((oa.l + ob.l + oc.l + od.l) % 2 != 0)
        violate_parity++;
      if ((oa.mmt == 1 && ob.mmt == -1) || (oc.mmt == 1 && od.mmt == -1))
        np_cnt++;
      if ((a == b && J % 2 == 1) || (c == d && J % 2 == 1))
        violate_Pauli++;
      if ((abs(oa.mmt + ob.mmt) == 2 && a < b) ||
          (abs(oc.mmt + od.mmt) == 2 && c < d))
        double_cnt++;
#endif

      me_map[{static_cast<uint16_t>(a), static_cast<uint16_t>(b),
              static_cast<uint16_t>(c), static_cast<uint16_t>(d),
              static_cast<uint16_t>(J)}] = v;
    }

    num_me = me_map.size();

#ifdef CHECKING_2B_CHANNELS
    std::cout << "There are " << violate_isospin << " isospin violations"
              << std::endl;
    std::cout << "There are " << violate_parity << " parity violations"
              << std::endl;
    std::cout << "There are " << np_cnt << " np channels" << std::endl;
    std::cout << "There are " << violate_Pauli << " channels violating Pauli"
              << std::endl;
    std::cout << "There are " << double_cnt
              << " pp or nn channels counted twice" << std::endl;
#endif

    return;
  }

  // Get ME from channels
  double TBME_scalar::getME(uint16_t a, uint16_t b, uint16_t c, uint16_t d,
                            uint16_t J) const
  {
    if (a == b && J % 2 == 1)
      return 0.0;
    if (c == d && J % 2 == 1)
      return 0.0;

    const auto &oa = orbit_map.getOrbit(a), ob = orbit_map.getOrbit(b),
               oc = orbit_map.getOrbit(c), od = orbit_map.getOrbit(d);

    int ea = 2 * oa.n + oa.l, eb = 2 * ob.n + ob.l, ec = 2 * oc.n + oc.l,
        ed = 2 * od.n + od.l;
    if (ea > maxShell || eb > maxShell || ec > maxShell || ed > maxShell)
      throw std::out_of_range("ME beyond maxShell");

    int ab_Nhw = 0;
    if (ea > valenceShell)
      ab_Nhw += (ea - valenceShell);
    if (eb > valenceShell)
      ab_Nhw += (eb - valenceShell);
    if (ab_Nhw > Nmax)
      throw std::out_of_range("bra channel beyond Nmax");

    int cd_Nhw = 0;
    if (ec > valenceShell)
      cd_Nhw += (ec - valenceShell);
    if (ed > valenceShell)
      cd_Nhw += (ed - valenceShell);
    if (cd_Nhw > Nmax)
      throw std::out_of_range("ket channel beyond Nmax");

    auto it = me_map.find({a, b, c, d, J});
    if (it != me_map.end())
      return it->second;
    it = me_map.find({c, d, a, b, J});
    if (it != me_map.end())
      return it->second; // if not Hermitian, *= -1

    double phase_ab = (((oa.jj + ob.jj) / 2 - J) % 2 == 0) ? -1.0 : 1.0;
    it = me_map.find({b, a, c, d, J});
    if (it != me_map.end())
      return phase_ab * (it->second);
    it = me_map.find({c, d, b, a, J});
    if (it != me_map.end())
      return phase_ab * (it->second);

    double phase_cd = (((oc.jj + od.jj) / 2 - J) % 2 == 0) ? -1.0 : 1.0;
    it = me_map.find({a, b, d, c, J});
    if (it != me_map.end())
      return phase_cd * (it->second);
    it = me_map.find({d, c, a, b, J});
    if (it != me_map.end())
      return phase_cd * (it->second);

    it = me_map.find({b, a, d, c, J});
    if (it != me_map.end())
      return phase_ab * phase_cd * (it->second);
    it = me_map.find({d, c, b, a, J});
    if (it != me_map.end())
      return phase_ab * phase_cd * (it->second);

    return 0.0;
  }

  // Get ME from quantum numbers
  // Note that these e's are shell numbers, not nodal quantum numbers
  // The reason we have e's here is that this function would be compatible
  // with double TwoBodyOperator::sphericalME_AS(...)
  double TBME_scalar::getME(int ea, int lla, int jja, int mmja, int mmta,
                            int eb, int llb, int jjb, int mmjb, int mmtb,
                            int ec, int llc, int jjc, int mmjc, int mmtc,
                            int ed, int lld, int jjd, int mmjd, int mmtd,
                            qcx_utils::SO3_CGC& cgcs) const
  {
    double x = 0.0;
    int MMJab = mmja + mmjb, MMJcd = mmjc + mmjd, MMTab = mmta + mmtb,
        MMTcd = mmtc + mmtd;
    if ((MMJab != MMJcd) || (MMTab != MMTcd))
      return x;

    int JJ_min = std::max(abs(jja - jjb), abs(jjc - jjd)),
        JJ_max = std::min(jja + jjb, jjc + jjd);

    if (JJ_min > JJ_max)
      return x;

    int la = lla / 2, lb = llb / 2, lc = llc / 2, ld = lld / 2,
        na = (ea - la) / 2, nb = (eb - lb) / 2, nc = (ec - lc) / 2,
        nd = (ed - ld) / 2;

    auto a = orbit_map.getIndex(na, la, jja, mmta),
         b = orbit_map.getIndex(nb, lb, jjb, mmtb),
         c = orbit_map.getIndex(nc, lc, jjc, mmtc),
         d = orbit_map.getIndex(nd, ld, jjd, mmtd);

    double norm_ab = (a == b) ? SQRT2 : 1.0, norm_cd = (c == d) ? SQRT2 : 1.0;

    for (int JJ = JJ_min; JJ <= JJ_max; JJ += 2)
    {
      x += cgcs.compute(jja, jjb, JJ, mmja, mmjb, MMJab) *
           cgcs.compute(jjc, jjd, JJ, mmjc, mmjd, MMJab) *
           getME(a, b, c, d, JJ / 2);
    }

    return norm_ab * norm_cd * x;
  }
} // namespace HO
