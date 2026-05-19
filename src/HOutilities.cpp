#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <HOutilities.h>

#include <qcx_convert.h>
#include <qcx_print.h>
#include <qcx_readwrite.h>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

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
      double factor = qcx_utils::racahs9().get({llb, 2, lla, 1, 0, 1, jjb, 2, jja});
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
      double factor = qcx_utils::racahs9().get({llb, 2, lla, 1, 0, 1, jjb, 2, jja});
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

  //////////// CLASS Table_nljmjmt /////////////

  // Constructor
  Table_nljmjmt::Table_nljmjmt(int maxShell)
  {
    assert(maxShell <= 44);
    uint16_t index = 0;
    for (qn_t n = 0; n <= maxShell; n++)
      for (qn_t l = n; l >= 0; l -= 2)
        for (qn_t jj = 2 * l + 1; jj >= std::abs(2 * l - 1); jj -= 2)
          for (qn_t mmj = jj; mmj >= -jj; mmj -= 2)
            for (qn_t mmt = 1; mmt >= -1; mmt -= 2)
            {
              key_type key = {n, 2 * l, jj, mmj, mmt};
              state_to_index[key] = index;
              index_to_state.push_back(key);
              index++;
            }
  }

  // Get index from quantum numbers
  uint16_t Table_nljmjmt::getIndex(qn_t n, qn_t ll, 
                        qn_t jj, qn_t mmj, qn_t mmt) const
  {
    key_type key = {n, ll, jj, mmj, mmt};
    auto it = state_to_index.find(key);
    if (it == state_to_index.end())
      throw std::out_of_range("Labels not found");
    return it->second;
  }

  // Get quantum numbers from index
  const std::tuple<qn_t, qn_t, qn_t, qn_t, qn_t>&
  Table_nljmjmt::getLabels(uint16_t index) const
  {
    return index_to_state[index];
  }

  ////////////// CLASS Orbit ///////////////

  bool Orbit::operator==(const Orbit& other) const
  {
    return other.n_ == n_ && other.l_ == l_ && 
           other.jj_ == jj_ && other.mmt_ == mmt_;
  }

  bool Orbit::operator<(const Orbit& other) const
  {
    return std::tie(n_, l_, jj_, mmt_) <
           std::tie(other.n_, other.l_, other.jj_, other.mmt_);
  }

  //////////// CLASS OrbitsMap //////////////

  // Constructor
  OrbitsMap::OrbitsMap(int emax, std::string style)
  {
    assert(emax <= 44);
    // It does not take in lmax, it is set to default lmax = emax
    emax_ = qcx_utils::ito8(emax);

    if (style == "NuHamil")
    {
      uint16_t idx = 1;
      for (qn_t e = 0; e <= emax; ++e)
      {
        for (qn_t l = e % 2; l <= e; l += 2)
        {
          qn_t n = (e - l) / 2;
          for (qn_t jj = abs(2 * l - 1); jj <= 2 * l + 1; jj += 2)
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
    }
    else if (style == "IMSRG")
    {
      uint16_t idx = 1;
      for (qn_t e = 0; e <= emax; ++e)
      {
        for (qn_t l = e; l >= 0; l -= 2)
        {
          qn_t n = (e - l) / 2;
          for (qn_t jj = 2 * l + 1; jj >= abs(2 * l - 1); jj -= 2)
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
    }
    else
      throw std::runtime_error("Style of orbit map not known");
  }

  // Get index of the orbit
  // Caution: n = (e - l)/2 is the nodal quantum number
  // whereas e is the shell number
  uint16_t OrbitsMap::getIndex(qn_t n, qn_t l, qn_t jj, qn_t mmt) const
  {
    auto it = index_map.find({n, l, jj, mmt});
    if (it != index_map.end())
      return it->second;
    else
      throw std::runtime_error("Orbit not found");
  }


  //////////// Class OBME_scalar /////////////
  OBME_scalar::OBME_scalar()
      : me_map(),
        nlj_map(),
        name_(""),
        hw_(0),
        Nmax_(0),
        valenceShell_(0),
        maxShell_(0)
  { }

  OBME_scalar::OBME_scalar(const std::string& name, 
    int hw, int Nmax, int valenceShell, int maxShell)
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
    std::cout << "WARNING: This operator isn't from SAM" << std::endl;

    if (maxShell > 14)
      throw std::out_of_range("Sorry, we only have maxShell up to 14");

    name_ = name;
    hw_ = (hw >= 0) ? qcx_utils::ito8(hw) : 20;
    Nmax_ = (Nmax >= 0) ? qcx_utils::ito8(Nmax) : 14;
    valenceShell_ = (valenceShell > 0) ? qcx_utils::ito8(valenceShell) : 0;
    maxShell_ = (maxShell >= 0) ? qcx_utils::ito8(maxShell) : 14;
    
    custom_name = true;

    int idx = 1;
    for (qn_t e = 0; e <= maxShell_; ++e)
    {
      for (qn_t l = e % 2; l <= e; l += 2)
      {
        qn_t n = (e - l) / 2;
        for (qn_t jj = abs(2 * l - 1); jj <= 2 * l + 1; jj += 2)
        {
          nlj_map[{n, l, jj}] = idx;
          idx++;
        }
      }
    }
  }

  void OBME_scalar::is() const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
    std::cout << "One-body potential with " << name_
              << " hw=" << static_cast<int>(hw_)
              << " valenceShell=" << static_cast<int>(valenceShell_)
              << " Nmax=" << static_cast<int>(Nmax_)
              << " maxShell=" << static_cast<int>(maxShell_) << std::endl;
  }

  std::ostream& operator<<(std::ostream& s, const OBME_scalar& x)
  {
    s << "One-body potential with " << x.name_
      << " hw=" << static_cast<int>(x.hw_)
      << " valenceShell=" << static_cast<int>(x.valenceShell_)
      << " Nmax=" << static_cast<int>(x.Nmax_)
      << " maxShell=" << static_cast<int>(x.maxShell_);

    return s;
  }

  void OBME_scalar::read()
  {
    std::string filename;

    if (custom_name)
    {
#ifdef SAM_DATA_DIR
      const char* sam_data = std::getenv("SAM_DATA_DIR");    
      if (!sam_data)
        throw std::runtime_error(
          "Environment variable SAM_DATA_DIR is not set");

      filename = std::string(sam_data) +
        "/Ab-initio_Interactions/OneBodyOperatorJT/" + name_ + ".myg";
#else
      filename = "data/Ab-initio_Interactions/OneBodyOperatorJT/" + name_ + ".myg";
#endif

#ifdef USE_MPI
      int mpi_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

      if (mpi_rank == 0)
#endif
      std::cout << "Start reading matrix elements from file" << std::endl;

      read_from_file_nuhamil(filename);

#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
      std::cout << "Finish reading matrix elements from file" << std::endl;
    }
    else
    {
/*
#ifdef HAVE_NUHAMIL
      const char* nuhamil_root = std::getenv("NUHAMIL_ROOT");    
      if (!nuhamil_root)
        throw std::runtime_error(
          "Environment variable NUHAMIL_ROOT is not set");

      filename = std::string(nuhamil_root) + 
          "/exe/TwBME-HO_NN-only_" + name_ + "_hw" +
          std::to_string(static_cast<int>(hw_)) + "_emax14_e2max28.myg";
#else

#ifdef SAM_DATA_DIR
      const char* sam_data = std::getenv("SAM_DATA_DIR");    
      if (!sam_data)
        throw std::runtime_error(
          "Environment variable SAM_DATA_DIR is not set");

      filename = std::string(sam_data) +
        "/Ab-initio_Interactions/TwBME-HO_NN-only_" + name_ + "_hw" +
        std::to_string(static_cast<int>(hw_)) + "_emax14_e2max28.myg";
#else
      filename =
          "data/Ab-initio_Interactions/TwBME-HO_NN-only_" + name_ + "_hw" +
          std::to_string(static_cast<int>(hw_)) + "_emax14_e2max28.myg";
#endif

#endif

      std::cout << "Start reading matrix elements from file" << std::endl;
#ifdef READ_TWBME_FROM_BINARY
      filename += ".bin";
      read_from_binary_nuhamil(filename);
#else
      read_from_file_nuhamil(filename);
#endif
      std::cout << "Finish reading matrix elements from file" << std::endl;
*/    
      throw std::runtime_error("OBME_scalar::read() not implemented");
    }
  }

  void OBME_scalar::read_from_file_nuhamil(const std::string& filename)
  {
    std::ifstream infile(filename);
    if (!infile)
      throw std::runtime_error("Failed to open file: " + filename);


    // Read all matrix elements
    int a, b, mmta, mmtb;
    double v;

    while (infile >> mmta >> a >> mmtb >> b >> v)
    {
      assert(mmta == mmtb);
      me_map[{mmta, a, mmtb, b}] = v;
    }

    return;
  }

  double OBME_scalar::getME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta, 
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb) const
  {
    if (mmja != mmjb) return 0.0;

    qn_t la = lla / 2, lb = llb / 2, na = (ea - la) / 2, nb = (eb - lb) / 2;
    int a, b;

    auto it_a = nlj_map.find({na, la, jja});
    if (it_a != nlj_map.end()) a = it_a->second;
    else throw std::runtime_error("Orbit not found");

    auto it_b = nlj_map.find({nb, lb, jjb});
    if (it_b != nlj_map.end()) b = it_b->second;
    else throw std::runtime_error("Orbit not found");

    auto it_me = me_map.find({static_cast<int>(mmta), a,
                              static_cast<int>(mmtb), b});

    if (it_me != me_map.end()) 
    {
#ifdef POSITIVE_INFINITY
    double phase = ((na + nb) % 2 == 0) ? 1.0 : -1.0;
    return phase * (it_me->second);
#else
    return it_me->second;
#endif
    }
    else return 0.0;
  }

  //////////// Class TwBME_scalar /////////////
  TwBME_scalar::TwBME_scalar()
      : me_map(),
        orbit_map(),
        name_(""),
        hw_(0),
        Nmax_(0),
        valenceShell_(0),
        maxShell_(0)
  { }

  TwBME_scalar::TwBME_scalar(const std::string& name, 
    int hw, int Nmax, int valenceShell, int maxShell)
  { 
    if (maxShell > 14)
      throw std::out_of_range("Sorry, we only have maxShell up to 14");

    name_ = name;
    hw_ = (hw >= 0) ? qcx_utils::ito8(hw) : 20;
    Nmax_ = (Nmax >= 0) ? qcx_utils::ito8(Nmax) : 14;
    valenceShell_ = (valenceShell > 0) ? qcx_utils::ito8(valenceShell) : 0;
    maxShell_ = (maxShell >= 0) ? qcx_utils::ito8(maxShell) : 14;

    static const std::unordered_set<std::string> known_potentials = {
      "N2LO_EMN500_bare", "N3LO_EMN500_bare", "N4LO_EMN500_bare",
      "N3LO_EM500_bare",  "N2LO_opt_bare",    "N2LO_sat_bare", 
      "N2LO_EMN500_srg1.0", "N3LO_EMN500_srg1.0", "N4LO_EMN500_srg1.0",
      "N3LO_EM500_srg1.0",  "N2LO_opt_srg1.0",    "N2LO_sat_srg1.0",
      "N2LO_EMN500_srg2.0", "N3LO_EMN500_srg2.0", "N4LO_EMN500_srg2.0",
      "N3LO_EM500_srg2.0",  "N2LO_opt_srg2.0",    "N2LO_sat_srg2.0",
      "N2LO_EMN500_srg4.0", "N3LO_EMN500_srg4.0", "N4LO_EMN500_srg4.0",
      "N3LO_EM500_srg4.0",  "N2LO_opt_srg4.0",    "N2LO_sat_srg4.0"};

    if (known_potentials.count(name))
    {
      orbit_map = OrbitsMap(14, "NuHamil");
      custom_name = false;
    }
    else
    {
#ifdef USE_MPI
      int mpi_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

      if (mpi_rank == 0)
#endif
      std::cout << "WARNING: This potential isn't from NuHamil" << std::endl;
      
      // orbit_map = OrbitsMap(14, "IMSRG");
      orbit_map = OrbitsMap(14, "NuHamil");
      custom_name = true;
    }
  }

  void TwBME_scalar::is() const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
    std::cout << "Two-body potential with " << name_
              << " hw=" << static_cast<int>(hw_)
              << " valenceShell=" << static_cast<int>(valenceShell_)
              << " Nmax=" << static_cast<int>(Nmax_)
              << " maxShell=" << static_cast<int>(maxShell_) << std::endl;
  }

  std::ostream& operator<<(std::ostream& s, const TwBME_scalar& x)
  {
    s << "Two-body potential with " << x.name_
      << " hw=" << static_cast<int>(x.hw_)
      << " valenceShell=" << static_cast<int>(x.valenceShell_)
      << " Nmax=" << static_cast<int>(x.Nmax_)
      << " maxShell=" << static_cast<int>(x.maxShell_);

    return s;
  }

  /* Reading method is based on
         subroutine read_scalar_operator_binary_myg(this, f)
         in NuHamil-public/src/TwoBody/TwoBodyLabOps.F90
  */
// #define CHECKING_2B_CHANNELS
  void TwBME_scalar::read()
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

    std::string filename;

    if (custom_name)
    {
#ifdef SAM_DATA_DIR
      const char* sam_data = std::getenv("SAM_DATA_DIR");    
      if (!sam_data)
        throw std::runtime_error(
          "Environment variable SAM_DATA_DIR is not set");

      filename = std::string(sam_data) +
        "/Ab-initio_Interactions/TwoBodyOperatorJT/" + name_ + ".myg.bin";
#else
      filename = "data/Ab-initio_Interactions/TwoBodyOperatorJT/" + name_ + ".myg.bin";
#endif

#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
      std::cout << "Start reading matrix elements from file" << std::endl;
      
      // read_from_binary_imsrg(filename);
      read_from_binary_nuhamil(filename);

#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
      std::cout << "Finish reading matrix elements from file" << std::endl;
    }
    else
    {
#ifdef HAVE_NUHAMIL
      const char* nuhamil_root = std::getenv("NUHAMIL_ROOT");    

      if (!nuhamil_root)
        throw std::runtime_error("Environment variable NUHAMIL_ROOT isn't set");

      filename = std::string(nuhamil_root) + 
          "/exe/TwBME-HO_NN-only_" + name_ + "_hw" +
          std::to_string(static_cast<int>(hw_)) + "_emax14_e2max28.myg";
#else

#ifdef SAM_DATA_DIR
      const char* sam_data = std::getenv("SAM_DATA_DIR"); 

      if (!sam_data)
        throw std::runtime_error("Environment variable SAM_DATA_DIR isn't set");

      filename = std::string(sam_data) +
        "/Ab-initio_Interactions/TwBME-HO_NN-only_" + name_ + "_hw" +
        std::to_string(static_cast<int>(hw_)) + "_emax14_e2max28.myg";
#else
      filename =
          "data/Ab-initio_Interactions/TwBME-HO_NN-only_" + name_ + "_hw" +
          std::to_string(static_cast<int>(hw_)) + "_emax14_e2max28.myg";
#endif

#endif

#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
      std::cout << "Start reading matrix elements from file" << std::endl;

#ifdef READ_TWBME_FROM_BINARY
      filename += ".bin";
      read_from_binary_nuhamil(filename);
#else
      read_from_file_nuhamil(filename);
#endif

#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
      std::cout << "Finish reading matrix elements from file" << std::endl;
    }
  }

  void TwBME_scalar::read_from_binary_nuhamil(const std::string& filename)
  {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile)
      throw std::runtime_error("Failed to open: " + filename);

    // Read number of matrix elements
    int num_me;
    qcx_utils::readBinary(infile, num_me);
    if (!infile)
        throw std::runtime_error("Failed to read count from: " + filename);

#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    
    if (mpi_rank == 0)
#endif
    qcx_utils::print("Number of mes: {}\n", num_me);

    // Allocate temporary buffers
    /* From subroutine write_scalar_operator_binary_myg(this, f)
       in NuHamil-public/src/TwoBody/TwoBodyLabOps.F90:
    integer :: a, b, c, d, bra, ket
    integer :: j, ipar, itz, ich, nn
    integer, allocatable :: ia(:), ib(:), ic(:), id(:), jj(:)
    */
    std::vector<int> ia(num_me), ib(num_me), ic(num_me), id(num_me), J(num_me);
    std::vector<double> v(num_me);

    // Read data arrays
    qcx_utils::readBinary(infile, ia);
    qcx_utils::readBinary(infile, ib);
    qcx_utils::readBinary(infile, ic);
    qcx_utils::readBinary(infile, id);
    qcx_utils::readBinary(infile, J);
    qcx_utils::readBinary(infile, v);

    if (!infile)
      throw std::runtime_error("Failed to read matrix elements from: " + filename);

    // Populate the map
#ifdef CHECKING_2B_CHANNELS
    int violate_isospin = 0, violate_parity = 0, np_cnt = 0, violate_Pauli = 0,
        double_cnt = 0;
#endif
    me_map.reserve(num_me);
    for (int i = 0; i < num_me; ++i)
    {
      const auto &oa = orbit_map.getOrbit(ia[i]);
      const auto &ob = orbit_map.getOrbit(ib[i]); 
      const auto &oc = orbit_map.getOrbit(ic[i]);
      const auto &od = orbit_map.getOrbit(id[i]);
/*
      std::cout << "ia: " << ia[i] << "\t" << oa << std::endl;
      std::cout << "ib: " << ib[i] << "\t" << ob << std::endl;
      std::cout << "ic: " << ic[i] << "\t" << oc << std::endl;
      std::cout << "id: " << id[i] << "\t" << od << std::endl;
*/
      qn_t ea = oa.shell(), eb = ob.shell(), ec = oc.shell(), ed = od.shell();

      assert((oa.mmt_ + ob.mmt_) == (oc.mmt_ + od.mmt_));
      assert((oa.l_ + ob.l_ + oc.l_ + od.l_) % 2 == 0);

      if (ea > maxShell_ || eb > maxShell_ || ec > maxShell_ || ed > maxShell_)
        continue;

      qn_t ab_Nhw = 0;
      if (ea > valenceShell_) ab_Nhw += (ea - valenceShell_);
      if (eb > valenceShell_) ab_Nhw += (eb - valenceShell_);
      if (ab_Nhw > Nmax_) continue;

      qn_t cd_Nhw = 0;
      if (ec > valenceShell_) cd_Nhw += (ec - valenceShell_);
      if (ed > valenceShell_) cd_Nhw += (ed - valenceShell_);
      if (cd_Nhw > Nmax_) continue;

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
              static_cast<qn_t>(J[i])}] = v[i];

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

         To see further confirmation, check 
         subroutine write_scalar_operator_binary_myg(this, f)
         in NuHamil-public/src/TwoBody/TwoBodyLabOps.F90.
         Inside the loops for bra and ket:
         a = ms%jpz(ich)%n2label1(bra)
         b = ms%jpz(ich)%n2label2(bra)
         c = ms%jpz(ich)%n2label1(ket)
         d = ms%jpz(ich)%n2label2(ket)

         Comparing these with subroutine InitTwoBodyLabPNChan in
         NuHamil-public/src/TwoBody/TwoBodyLabChanPN.F90
         ii1 = i1
         ii2 = i2
         this%n2label1(n_states) = ii1
         this%n2label2(n_states) = ii2
         this%iphase(ii1,ii2) = 1
         this%iphase(ii2,ii1) = -(-1) ** ((j1 + j2) / 2 - j)
         we can see that the phase for (a,b) or (c,d) that are printed
         out is always 1. The second phase will be taken when performing
         permutation.
      */
    }

#ifdef CHECKING_2B_CHANNELS
#ifdef USE_MPI
    if (mpi_rank == 0)
#endif
    {
      std::cout << "There are " << violate_isospin << " isospin violations"
                << std::endl;
      std::cout << "There are " << violate_parity << " parity violations"
                << std::endl;
      std::cout << "There are " << np_cnt << " np channels" << std::endl;
      std::cout << "There are " << violate_Pauli << " channels violating Pauli"
                << std::endl;
      std::cout << "There are " << double_cnt
                << " pp or nn channels counted twice" << std::endl;
    }
#endif

    return;
  }

  void TwBME_scalar::read_from_binary_imsrg(const std::string& filename)
  {
    // This function is based on the following function from imsrg code:
    // void ReadWrite::ReadTwoBody_Binary_myg(std::string filename, Operator& op)

    using fortran_int = int32_t;
    using fortran_real8 = double;

    std::ifstream infile(filename, std::ios::binary);
    if (!infile)
      throw std::runtime_error("Failed to open: " + filename);

    std::string header;
    std::getline(infile, header);
    if (!infile.good())
      throw std::runtime_error("Failed to read header from " + filename);

    // Read number of matrix elements
    fortran_int num_me;
    qcx_utils::readBinary(infile, num_me);
    if (!infile)
      throw std::runtime_error("Failed to read count from: " + filename);

    // qcx_utils::print("Number of mes: {}\n", num_me);

    // Allocate temporary buffers
    /* From subroutine write_scalar_operator_binary_myg(this, f)
       in NuHamil-public/src/TwoBody/TwoBodyLabOps.F90:
    integer :: a, b, c, d, bra, ket
    integer :: j, ipar, itz, ich, nn
    integer, allocatable :: ia(:), ib(:), ic(:), id(:), jj(:)
    */
    std::vector<fortran_int> ia(num_me), ib(num_me), ic(num_me), id(num_me), J(num_me);
    std::vector<fortran_real8> v(num_me);

    // Read data arrays
    qcx_utils::readBinary(infile, ia);
    qcx_utils::readBinary(infile, ib);
    qcx_utils::readBinary(infile, ic);
    qcx_utils::readBinary(infile, id);
    qcx_utils::readBinary(infile, J);
    qcx_utils::readBinary(infile, v);

    if (!infile)
      throw std::runtime_error("Failed to read matrix elements from: " + filename);

    // Populate the map
#ifdef CHECKING_2B_CHANNELS
    int violate_isospin = 0, violate_parity = 0, np_cnt = 0, violate_Pauli = 0,
        double_cnt = 0;
#endif
    me_map.reserve(num_me);
    for (fortran_int i = 0; i < num_me; ++i)
    {
      const auto &oa = orbit_map.getOrbit(ia[i] + 1);
      const auto &ob = orbit_map.getOrbit(ib[i] + 1); 
      const auto &oc = orbit_map.getOrbit(ic[i] + 1);
      const auto &od = orbit_map.getOrbit(id[i] + 1);
/*
      std::cout << "ia: " << ia[i] << "\t" << oa << std::endl;
      std::cout << "ib: " << ib[i] << "\t" << ob << std::endl;
      std::cout << "ic: " << ic[i] << "\t" << oc << std::endl;
      std::cout << "id: " << id[i] << "\t" << od << std::endl;
*/
      qn_t ea = oa.shell(), eb = ob.shell(), ec = oc.shell(), ed = od.shell();

      assert((oa.mmt_ + ob.mmt_) == (oc.mmt_ + od.mmt_));
      assert((oa.l_ + ob.l_ + oc.l_ + od.l_) % 2 == 0);

      if (ea > maxShell_ || eb > maxShell_ || ec > maxShell_ || ed > maxShell_)
        continue;

      qn_t ab_Nhw = 0;
      if (ea > valenceShell_) ab_Nhw += (ea - valenceShell_);
      if (eb > valenceShell_) ab_Nhw += (eb - valenceShell_);
      if (ab_Nhw > Nmax_) continue;

      qn_t cd_Nhw = 0;
      if (ec > valenceShell_) cd_Nhw += (ec - valenceShell_);
      if (ed > valenceShell_) cd_Nhw += (ed - valenceShell_);
      if (cd_Nhw > Nmax_) continue;

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

      me_map[{static_cast<uint16_t>(ia[i]) + 1,
              static_cast<uint16_t>(ib[i]) + 1,
              static_cast<uint16_t>(ic[i]) + 1,
              static_cast<uint16_t>(id[i]) + 1,
              static_cast<qn_t>(J[i])}] = v[i];

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

         To see further confirmation, check 
         subroutine write_scalar_operator_binary_myg(this, f)
         in NuHamil-public/src/TwoBody/TwoBodyLabOps.F90.
         Inside the loops for bra and ket:
         a = ms%jpz(ich)%n2label1(bra)
         b = ms%jpz(ich)%n2label2(bra)
         c = ms%jpz(ich)%n2label1(ket)
         d = ms%jpz(ich)%n2label2(ket)

         Comparing these with subroutine InitTwoBodyLabPNChan in
         NuHamil-public/src/TwoBody/TwoBodyLabChanPN.F90
         ii1 = i1
         ii2 = i2
         this%n2label1(n_states) = ii1
         this%n2label2(n_states) = ii2
         this%iphase(ii1,ii2) = 1
         this%iphase(ii2,ii1) = -(-1) ** ((j1 + j2) / 2 - j)
         we can see that the phase for (a,b) or (c,d) that are printed
         out is always 1. The second phase will be taken when performing
         permutation.
      */
    }

#ifdef CHECKING_2B_CHANNELS
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
    {
      std::cout << "There are " << violate_isospin << " isospin violations"
                << std::endl;
      std::cout << "There are " << violate_parity << " parity violations"
                << std::endl;
      std::cout << "There are " << np_cnt << " np channels" << std::endl;
      std::cout << "There are " << violate_Pauli << " channels violating Pauli"
                << std::endl;
      std::cout << "There are " << double_cnt
                << " pp or nn channels counted twice" << std::endl;
    }
#endif

    return;
  }

  void TwBME_scalar::read_from_file_nuhamil(const std::string& filename)
  {
    std::ifstream infile(filename);
    if (!infile)
      throw std::runtime_error("Failed to open file: " + filename);

    std::string line;

    // Skip first header line
    std::getline(infile, line);

    // Get number of matrix elements
    int num_me;
    infile >> num_me;
    me_map.reserve(num_me);
    // Skip the rest of that line
    std::getline(infile, line);

    // Skip label line
    std::getline(infile, line);

    // Read all matrix elements
    int a, b, c, d, J;
    double v;

#ifdef CHECKING_2B_CHANNELS
    int violate_isospin = 0, violate_parity = 0, np_cnt = 0, violate_Pauli = 0,
        double_cnt = 0;
#endif

    while (infile >> a >> b >> c >> d >> J >> v)
    {
      const auto &oa = orbit_map.getOrbit(a);
      const auto &ob = orbit_map.getOrbit(b);
      const auto &oc = orbit_map.getOrbit(c); 
      const auto &od = orbit_map.getOrbit(d);

      assert((oa.mmt_ + ob.mmt_) == (oc.mmt_ + od.mmt_));
      assert((oa.l_ + ob.l_ + oc.l_ + od.l_) % 2 == 0);

      qn_t ea = oa.shell(), eb = ob.shell(), ec = oc.shell(), ed = od.shell();

      if (ea > maxShell_ || eb > maxShell_ || ec > maxShell_ || ed > maxShell_)
        continue;

      qn_t ab_Nhw = 0;
      if (ea > valenceShell_) ab_Nhw += (ea - valenceShell_);
      if (eb > valenceShell_) ab_Nhw += (eb - valenceShell_);
      if (ab_Nhw > Nmax_) continue;

      qn_t cd_Nhw = 0;
      if (ec > valenceShell_) cd_Nhw += (ec - valenceShell_);
      if (ed > valenceShell_) cd_Nhw += (ed - valenceShell_);
      if (cd_Nhw > Nmax_) continue;

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
              static_cast<qn_t>(J)}] = v;
    }

#ifdef CHECKING_2B_CHANNELS
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
    {
      std::cout << "There are " << violate_isospin << " isospin violations"
                << std::endl;
      std::cout << "There are " << violate_parity << " parity violations"
                << std::endl;
      std::cout << "There are " << np_cnt << " np channels" << std::endl;
      std::cout << "There are " << violate_Pauli << " channels violating Pauli"
                << std::endl;
      std::cout << "There are " << double_cnt
                << " pp or nn channels counted twice" << std::endl;
    }
#endif

    return;
  }

  void TwBME_scalar::clear()
  {
    std::unordered_map<key_type, double, boost::hash<key_type>>().swap(me_map);
  }

  // Get ME from channels
  double TwBME_scalar::getME(uint16_t a, uint16_t b, uint16_t c, uint16_t d,
                             qn_t J) const
  {
    if (a == b && J % 2 == 1) return 0.0;
    if (c == d && J % 2 == 1) return 0.0;

    const auto& oa = orbit_map.getOrbit(static_cast<int>(a)); 
    const auto& ob = orbit_map.getOrbit(static_cast<int>(b));
    const auto& oc = orbit_map.getOrbit(static_cast<int>(c)); 
    const auto& od = orbit_map.getOrbit(static_cast<int>(d));

    qn_t ea = oa.shell(), eb = ob.shell(), ec = oc.shell(), ed = od.shell();
/*
    std::cout << "a=" << a << ": " << oa << std::endl;
    std::cout << "b=" << b << ": " << ob << std::endl;
    std::cout << "c=" << c << ": " << oc << std::endl;
    std::cout << "d=" << d << ": " << od << std::endl;
    std::cout << "J: " << static_cast<int>(J) << std::endl;
*/
    //// HERE IS FOR CHECKING IF READING WAS CORRECT ////
    /*
    if (ea > maxShell_ || eb > maxShell_ || ec > maxShell_ || ed > maxShell_)
      throw std::out_of_range("ME beyond maxShell");

    qn_t ab_Nhw = 0;
    if (ea > valenceShell_) ab_Nhw += (ea - valenceShell_);
    if (eb > valenceShell_) ab_Nhw += (eb - valenceShell_);
    if (ab_Nhw > Nmax_)
      throw std::out_of_range("bra channel beyond Nmax");

    qn_t cd_Nhw = 0;
    if (ec > valenceShell_) cd_Nhw += (ec - valenceShell_);
    if (ed > valenceShell_) cd_Nhw += (ed - valenceShell_);
    if (cd_Nhw > Nmax_)
      throw std::out_of_range("ket channel beyond Nmax");
    */
    //////////////////////////////////////////////////////

    auto it = me_map.find({a, b, c, d, J});
    if (it != me_map.end()) return it->second;
    it = me_map.find({c, d, a, b, J});
    if (it != me_map.end()) return it->second; // if antiHermitian, *= -1

    double phase_ab = (((oa.jj_ + ob.jj_) / 2 - J) % 2 == 0) ? -1.0 : 1.0;
    it = me_map.find({b, a, c, d, J});
    if (it != me_map.end()) return phase_ab * (it->second);
    it = me_map.find({c, d, b, a, J});
    if (it != me_map.end()) return phase_ab * (it->second);

    double phase_cd = (((oc.jj_ + od.jj_) / 2 - J) % 2 == 0) ? -1.0 : 1.0;
    it = me_map.find({a, b, d, c, J});
    if (it != me_map.end()) return phase_cd * (it->second);
    it = me_map.find({d, c, a, b, J});
    if (it != me_map.end()) return phase_cd * (it->second);

    it = me_map.find({b, a, d, c, J});
    if (it != me_map.end()) return phase_ab * phase_cd * (it->second);
    it = me_map.find({d, c, b, a, J});
    if (it != me_map.end()) return phase_ab * phase_cd * (it->second);

    // return 0.0;
    throw std::out_of_range("Cannot find in me_map");
  }

  // Get ME from quantum numbers
  // Note that these e's are shell numbers, not nodal quantum numbers
  // The reason we have e's here is that this function would be compatible
  // with double TwoBodyOperator::sphericalME_AS(...)
  double TwBME_scalar::getME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta, 
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb, 
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc, 
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const
  {
    qn_t MMJab = mmja + mmjb, MMJcd = mmjc + mmjd, 
         MMTab = mmta + mmtb, MMTcd = mmtc + mmtd;
    if ((MMJab != MMJcd) || (MMTab != MMTcd)) // spin or isospin violation
      return 0.0;

    qn_t JJ_min = std::max(abs(jja - jjb), abs(jjc - jjd)),
         JJ_max = std::min(jja + jjb, jjc + jjd);

    if (JJ_min > JJ_max) // no coupled J possible
      return 0.0;

    qn_t la = lla / 2, lb = llb / 2, lc = llc / 2, ld = lld / 2,
         na = (ea - la) / 2, nb = (eb - lb) / 2, nc = (ec - lc) / 2,
         nd = (ed - ld) / 2;

    if ((la + lb + lc + ld) % 2 != 0) // parity violation
      return 0.0;

    auto a = orbit_map.getIndex(na, la, jja, mmta),
         b = orbit_map.getIndex(nb, lb, jjb, mmtb),
         c = orbit_map.getIndex(nc, lc, jjc, mmtc),
         d = orbit_map.getIndex(nd, ld, jjd, mmtd);

    double norm_ab = (a == b) ? SQRT2 : 1.0, norm_cd = (c == d) ? SQRT2 : 1.0;

    double x = 0.0;
    for (qn_t JJ = JJ_min; JJ <= JJ_max; JJ += 2)
    {
      if ((a == b && JJ % 4 != 0) || (c == d && JJ % 4 != 0))
        continue;

      x += qcx_utils::cgcs().get({jja, jjb, JJ, mmja, mmjb, MMJab}) *
           qcx_utils::cgcs().get({jjc, jjd, JJ, mmjc, mmjd, MMJab}) *
           getME(a, b, c, d, JJ / 2);
    }

#ifdef POSITIVE_INFINITY
    double phase = ((na + nb + nc + nd) % 2 == 0) ? 1.0 : -1.0;
    return phase * x * norm_ab * norm_cd;
#else
    return x * norm_ab * norm_cd;
#endif
  }
} // namespace HO
