#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <charconv>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <zlib.h>

#include <HOmes.h>

#include <convert.h>
#include <print.h>
#include <readwrite.h>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Fano.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#define POSITIVE_INFINITY
//#define POSITIVE_ORIGIN

namespace HO
{
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
    hw_ = (hw >= 0) ? utils::ito8(hw) : 20;
    Nmax_ = (Nmax >= 0) ? utils::ito8(Nmax) : 14;
    valenceShell_ = (valenceShell > 0) ? utils::ito8(valenceShell) : 0;
    maxShell_ = (maxShell >= 0) ? utils::ito8(maxShell) : 14;
    
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
    hw_ = (hw >= 0) ? utils::ito8(hw) : 20;
    Nmax_ = (Nmax >= 0) ? utils::ito8(Nmax) : 14;
    valenceShell_ = (valenceShell > 0) ? utils::ito8(valenceShell) : 0;
    maxShell_ = (maxShell >= 0) ? utils::ito8(maxShell) : 14;

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
    utils::readBinary(infile, num_me);
    if (!infile)
        throw std::runtime_error("Failed to read count from: " + filename);

#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    
    if (mpi_rank == 0)
#endif
    utils::print("Number of mes: {}\n", num_me);

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
    utils::readBinary(infile, ia);
    utils::readBinary(infile, ib);
    utils::readBinary(infile, ic);
    utils::readBinary(infile, id);
    utils::readBinary(infile, J);
    utils::readBinary(infile, v);

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

      me_map[{pack_abcd(static_cast<uint16_t>(ia[i]),
                        static_cast<uint16_t>(ib[i]),
                        static_cast<uint16_t>(ic[i]),
                        static_cast<uint16_t>(id[i])),
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
    utils::readBinary(infile, num_me);
    if (!infile)
      throw std::runtime_error("Failed to read count from: " + filename);

    // utils::print("Number of mes: {}\n", num_me);

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
    utils::readBinary(infile, ia);
    utils::readBinary(infile, ib);
    utils::readBinary(infile, ic);
    utils::readBinary(infile, id);
    utils::readBinary(infile, J);
    utils::readBinary(infile, v);

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

      me_map[{pack_abcd(static_cast<uint16_t>(ia[i] + 1),
                        static_cast<uint16_t>(ib[i] + 1),
                        static_cast<uint16_t>(ic[i] + 1),
                        static_cast<uint16_t>(id[i] + 1)),
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

      me_map[{pack_abcd(static_cast<uint16_t>(a), static_cast<uint16_t>(b),
                        static_cast<uint16_t>(c), static_cast<uint16_t>(d)),
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

#ifdef UNCOUPLE_ME_CACHE
  // One private hash map per thread -- a "singleton on each thread" via
  // thread_local. Each thread touches only its own map, so getME's lookup is
  // lock-free AND resolving the cache costs just a thread_local access: no
  // omp_get_thread_num(), no shared-vector index. The map is created lazily on a
  // thread's first getME and freed when that thread exits.
  TwBME_mscheme_cache_t& twbme_mscheme_cache()
  {
    thread_local TwBME_mscheme_cache_t cache;
    return cache;
  }

  // Per-thread caches are now self-managed (lazy create, freed at thread exit),
  // so these are no-ops -- kept so sam::initialize / sam::finalize need not
  // change. NOTE: memory is released at thread exit (program end for the OpenMP
  // worker pool) rather than at finalize; ask if you need an explicit mid-run
  // release for very large model spaces.
  void initialize_twbme_cache() { }
  void finalize_twbme_cache() { }

  // Release this thread's m-scheme cache when it outgrows `max_entries`. swap()
  // with a fresh empty map frees the buckets outright (clear() would keep the
  // capacity reserved); the next getME lazily rebuilds. See declaration for the
  // memory/reuse tradeoff. `max_entries == 0` => never release.
  bool maybe_release_twbme_cache(std::size_t max_entries)
  {
    if (max_entries == 0) return false;
    auto& c = twbme_mscheme_cache();
    if (c.size() <= max_entries) return false;
    TwBME_mscheme_cache_t().swap(c);
    return true;
  }

  // ---- m-scheme cache hit/miss counters (diagnostic) --------------------
#ifdef COUNT_MSCHEME_CACHE
  namespace { unsigned long long g_ms_hits[256] = {}, g_ms_misses[256] = {}; }
#endif
  void report_mscheme_cache_stats()
  {
#ifdef COUNT_MSCHEME_CACHE
    unsigned long long h = 0, m = 0;
    for (int i = 0; i < 256; i++) { h += g_ms_hits[i]; m += g_ms_misses[i]; }
    unsigned long long tot = h + m;
    std::cout << "[mscheme-cache] hits=" << h << "  misses=" << m
              << "  total=" << tot << "  ("
              << (tot ? 100.0 * double(h) / double(tot) : 0.0) << "% hit)\n";
    for (int i = 0; i < 256; i++) { g_ms_hits[i] = 0; g_ms_misses[i] = 0; }
#endif
  }

  // ---- intra-tensor m-scheme reuse (diagnostic) -------------------------
#ifdef MEASURE_MSCHEME_REUSE
  namespace
  {
    // this thread's DISTINCT quads for the tensor currently being expanded
    thread_local std::unordered_set<TwBME_mscheme_key,
                                    boost::hash<TwBME_mscheme_key>> g_probed;
    thread_local unsigned long long g_probes_this = 0;
    unsigned long long g_ru_unique[256] = {}, g_ru_probes[256] = {},
                       g_ru_ntensors[256] = {}, g_ru_maxuniq[256] = {};
  }
  void mscheme_reuse_begin() { g_probed.clear(); g_probes_this = 0; }
  void mscheme_reuse_end()
  {
    int t = omp_get_thread_num();
    unsigned long long u = g_probed.size();
    g_ru_unique[t] += u;
    g_ru_probes[t] += g_probes_this;
    g_ru_ntensors[t]++;
    if (u > g_ru_maxuniq[t]) g_ru_maxuniq[t] = u;
  }
#endif
  void report_mscheme_reuse()
  {
#ifdef MEASURE_MSCHEME_REUSE
    unsigned long long u = 0, p = 0, n = 0, mx = 0;
    for (int i = 0; i < 256; i++)
    {
      u += g_ru_unique[i]; p += g_ru_probes[i]; n += g_ru_ntensors[i];
      if (g_ru_maxuniq[i] > mx) mx = g_ru_maxuniq[i];
    }
    std::cout << "[mscheme-reuse] tensors=" << n
              << "  total_probes=" << p
              << "  sum_unique_per_tensor=" << u
              << "  reuse=" << (u ? double(p) / double(u) : 0.0) << "x"
              << "  avg_unique/tensor=" << (n ? u / n : 0)
              << "  max_unique/tensor=" << mx << "\n";
    for (int i = 0; i < 256; i++)
    { g_ru_unique[i] = 0; g_ru_probes[i] = 0; g_ru_ntensors[i] = 0;
      g_ru_maxuniq[i] = 0; }
#endif
  }
#endif

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

    // qn_t ea = oa.shell(), eb = ob.shell(), ec = oc.shell(), ed = od.shell();
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

    auto it = me_map.find({pack_abcd(a, b, c, d), J});
    if (it != me_map.end()) return it->second;
    it = me_map.find({pack_abcd(c, d, a, b), J});
    if (it != me_map.end()) return it->second; // if antiHermitian, *= -1

    double phase_ab = (((oa.jj_ + ob.jj_) / 2 - J) % 2 == 0) ? -1.0 : 1.0;
    it = me_map.find({pack_abcd(b, a, c, d), J});
    if (it != me_map.end()) return phase_ab * (it->second);
    it = me_map.find({pack_abcd(c, d, b, a), J});
    if (it != me_map.end()) return phase_ab * (it->second);

    double phase_cd = (((oc.jj_ + od.jj_) / 2 - J) % 2 == 0) ? -1.0 : 1.0;
    it = me_map.find({pack_abcd(a, b, d, c), J});
    if (it != me_map.end()) return phase_cd * (it->second);
    it = me_map.find({pack_abcd(d, c, a, b), J});
    if (it != me_map.end()) return phase_cd * (it->second);

    it = me_map.find({pack_abcd(b, a, d, c), J});
    if (it != me_map.end()) return phase_ab * phase_cd * (it->second);
    it = me_map.find({pack_abcd(d, c, b, a), J});
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
#ifdef UNCOUPLE_ME_CACHE
    // Resolve this thread's cache once, then reuse the cache-passing overload.
    return getME(twbme_mscheme_cache(),
                 ea, lla, jja, mmja, mmta, eb, llb, jjb, mmjb, mmtb,
                 ec, llc, jjc, mmjc, mmtc, ed, lld, jjd, mmjd, mmtd);
#else
    return computeME(ea, lla, jja, mmja, mmta,
                     eb, llb, jjb, mmjb, mmtb,
                     ec, llc, jjc, mmjc, mmtc,
                     ed, lld, jjd, mmjd, mmtd);
#endif
  }

#ifdef UNCOUPLE_ME_CACHE
  // Memoize the m-scheme ME in the caller-supplied cache. getStrengths resolves
  // the thread_local cache once and passes it in, so the billions of calls in
  // the hot loop skip the per-call thread_local (emulated-TLS) resolution.
  double TwBME_scalar::getME(TwBME_mscheme_cache_t& cache,
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const
  {
    TwBME_mscheme_key key = pack_mscheme_2Bstate(
      ea, lla, jja, mmja, mmta, eb, llb, jjb, mmjb, mmtb,
      ec, llc, jjc, mmjc, mmtc, ed, lld, jjd, mmjd, mmtd);

#ifdef MEASURE_MSCHEME_REUSE
    g_probed.insert(key);
    ++g_probes_this;
#endif

    auto it = cache.find(key);
    if (it != cache.end())
    {
#ifdef COUNT_MSCHEME_CACHE
      ++g_ms_hits[omp_get_thread_num()];
#endif
      return it->second;
    }
#ifdef COUNT_MSCHEME_CACHE
    ++g_ms_misses[omp_get_thread_num()];
#endif
    double v = computeME(ea, lla, jja, mmja, mmta,
                         eb, llb, jjb, mmjb, mmtb,
                         ec, llc, jjc, mmjc, mmtc,
                         ed, lld, jjd, mmjd, mmtd);
    cache.emplace(key, v);
    return v;
  }

  // Two-level lookup: a small per-tensor `local` cache in front of the big
  // `global` one. The ~9x intra-tensor reuse hits `local` (L2/L3), so only the
  // first touch of each quad falls through to `global` (the 2.9GB DRAM table).
  double TwBME_scalar::getME(TwBME_mscheme_cache_t& local,
      TwBME_mscheme_cache_t& global,
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const
  {
    TwBME_mscheme_key key = pack_mscheme_2Bstate(
      ea, lla, jja, mmja, mmta, eb, llb, jjb, mmjb, mmtb,
      ec, llc, jjc, mmjc, mmtc, ed, lld, jjd, mmjd, mmtd);

    auto it = local.find(key);
    if (it != local.end()) return it->second;

    double v = getME(global,
                     ea, lla, jja, mmja, mmta, eb, llb, jjb, mmjb, mmtb,
                     ec, llc, jjc, mmjc, mmtc, ed, lld, jjd, mmjd, mmtd);
    local.emplace(key, v);
    return v;
  }
#endif

  double TwBME_scalar::computeME(
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

      x += utils::cgcs().get({jja, jjb, JJ, mmja, mmjb, MMJab}) *
           utils::cgcs().get({jjc, jjd, JJ, mmjc, mmjd, MMJab}) *
           getME(a, b, c, d, JJ / 2);
    }

#ifdef POSITIVE_INFINITY
    double phase = ((na + nb + nc + nd) % 2 == 0) ? 1.0 : -1.0;
    return phase * x * norm_ab * norm_cd;
#else
    return x * norm_ab * norm_cd;
#endif
  }

  //////////// Class TwBME_general /////////////

  TwBME_general::TwBME_general()
      : me_map(),
        obme_map(),
        NN_only_(true),
        orbit_map(),
        name_(""),
        custom_name(false),
        hw_(0),
        Nmax_(0),
        valenceShell_(0),
        maxShell_(0),
        JJ_(0),
        parity_(0),
        TT_(0)
  { }

  TwBME_general::TwBME_general(const std::string& name,
    int hw, int Nmax, int valenceShell, int maxShell)
  {
    if (maxShell > 14)
      throw std::out_of_range("Sorry, we only have maxShell up to 14");

    name_ = name;
    hw_ = (hw >= 0) ? utils::ito8(hw) : 20;
    Nmax_ = (Nmax >= 0) ? utils::ito8(Nmax) : 14;
    valenceShell_ = (valenceShell > 0) ? utils::ito8(valenceShell) : 0;
    maxShell_ = (maxShell >= 0) ? utils::ito8(maxShell) : 14;

    // overwritten from the file header in read_from_gzip_nuhamil()
    JJ_ = 0; parity_ = 0; TT_ = 0;

    if (known_ops.count(name))
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
      std::cout << "WARNING: This operator isn't known" << std::endl;
      
      // orbit_map = OrbitsMap(14, "IMSRG");
      orbit_map = OrbitsMap(14, "NuHamil");
      custom_name = true;
    }
  }

  void TwBME_general::is() const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
    std::cout << "General operator " << name_
              << " (JJ=" << static_cast<int>(JJ_)
              << " parity=" << static_cast<int>(parity_)
              << " TT=" << static_cast<int>(TT_) << ")"
              << " hw=" << static_cast<int>(hw_)
              << " valenceShell=" << static_cast<int>(valenceShell_)
              << " Nmax=" << static_cast<int>(Nmax_)
              << " maxShell=" << static_cast<int>(maxShell_) << std::endl;
  }

  std::ostream& operator<<(std::ostream& s, const TwBME_general& x)
  {
    s << "General operator " << x.name_
      << " (JJ=" << static_cast<int>(x.JJ_)
      << " parity=" << static_cast<int>(x.parity_)
      << " TT=" << static_cast<int>(x.TT_) << ")"
      << " hw=" << static_cast<int>(x.hw_)
      << " valenceShell=" << static_cast<int>(x.valenceShell_)
      << " Nmax=" << static_cast<int>(x.Nmax_)
      << " maxShell=" << static_cast<int>(x.maxShell_);

    return s;
  }

  void TwBME_general::read()
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

    // General operators live under data/General_operators/
    std::string filename;

#ifdef SAM_DATA_DIR
    const char* sam_data = std::getenv("SAM_DATA_DIR");
    if (!sam_data)
      throw std::runtime_error("Environment variable SAM_DATA_DIR is not set");
      
    filename = std::string(sam_data) + "/General_operators/" + name_;
#else
    filename = "data/General_operators/" + name_;
#endif

    if (!custom_name)
      filename += "_hw" + std::to_string(static_cast<int>(hw_)) + "_emax14_e2max28";
    
    filename += ".me2j.gz";

#ifdef USE_MPI
    if (mpi_rank == 0)
#endif
    std::cout << "Start reading matrix elements from file" << std::endl;

    read_from_gzip_nuhamil(filename);

#ifdef USE_MPI
    if (mpi_rank == 0)
#endif
    std::cout << "Finish reading matrix elements from file" << std::endl;
  }

  /* Reading method mirrors the loop order of
         subroutine write_general_operator_gzip(this, f)
         in NuHamil-public/src/TwoBody/TwoBodyLabOps.F90
     cross-checked against imsrg ReadWrite::ReadOperator2b_Miyagi. The file is a
     gzip-compressed me2j-like text whose entries carry no indices, so we
     replicate the writer's nested loops exactly to stay in sync with the
     stream.
  */
  void TwBME_general::read_from_gzip_nuhamil(const std::string& filename)
  {
    constexpr double tol = 1e-12;

    // Decompress the whole gzip file into memory with zlib (a C library, so
    // this is independent of the C++ standard library -- it avoids the
    // libstdc++/libc++ ABI mismatch that linking a system-built
    // boost::iostreams would introduce under g++/mpicxx).
    gzFile gz = gzopen(filename.c_str(), "rb");
    if (!gz)
      throw std::runtime_error("Failed to open: " + filename);

    std::string content;
    {
      char buf[1 << 16];
      int n;
      while ((n = gzread(gz, buf, sizeof(buf))) > 0)
        content.append(buf, static_cast<size_t>(n));
    }
    gzclose(gz);

    // Parse with std::from_chars over the in-memory buffer: far faster than
    // istream >> for the (millions of) values -- no locale, no sentry, no extra
    // buffer copy. Data after the 2-line header is a flat whitespace-separated
    // token stream (newlines are just whitespace), so we walk a pointer.
    const char* p = content.data();
    const char* end = p + content.size();
    auto skip_ws = [&] {
      while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t'))
        ++p;
    };
    auto next_d = [&](double& v) {
      skip_ws();
      auto res = std::from_chars(p, end, v);
      if (res.ec != std::errc()) throw std::runtime_error("Parse error: " + filename);
      p = res.ptr;
    };
    auto next_i = [&](int& v) {
      skip_ws();
      auto res = std::from_chars(p, end, v);
      if (res.ec != std::errc()) throw std::runtime_error("Parse error: " + filename);
      p = res.ptr;
    };
    auto skip_line = [&] {
      while (p < end && *p != '\n') ++p;
      if (p < end) ++p;
    };

    // (1) header comment
    skip_line();

    // (2) OpJ OpP OpZ Emax E2max [Lmax]  (Lmax ignored)
    int file_emax = 0, file_e2max = 0;
    {
      int J = 0, P = 1, Z = 0;
      next_i(J); next_i(P); next_i(Z); next_i(file_emax); next_i(file_e2max);
      JJ_ = 2 * utils::ito8(J);
      // operator parity in (0=even,1=odd) form, matching imsrg's (1-P)/2
      parity_ = (1 - utils::ito8(P)) / 2;
      TT_ = 2 * utils::ito8(Z);
      skip_line(); // discard the rest of the header line (Lmax)
    }

#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0)
#endif
    std::cout << "JJ=" << static_cast<int>(JJ_)
              << " parity=" << static_cast<int>(parity_)
              << " TT=" << static_cast<int>(TT_)
              << " emax=" << file_emax << " e2max=" << file_e2max << std::endl;

    // Build the orbit map to the file's emax. OrbitsMap("NuHamil") enumerates
    // orbits in the same (e; l; jj) order as the writer and stores, for each
    // isospin orbit i (0-based), the proton at index 2i+1 and the neutron at
    // 2i+2 -- i.e. it already is the nlj list the reader needs.
    orbit_map = OrbitsMap(file_emax, "NuHamil");
    const int num_nlj = static_cast<int>(orbit_map.num_orbits()) / 2;

    // (3) zero-body (always 0.0 in this format); read & discard
    double zerobody = 0.0;
    next_d(zerobody);

    // (4) one-body block: a=1..norbs, b=1..norbs (consume; store iff !NN_only_)

    auto triangle2j = [](int jj1, int jj2, int jj3) {
      return jj3 >= std::abs(jj1 - jj2) && jj3 <= jj1 + jj2 &&
             ((jj1 + jj2 + jj3) % 2 == 0);
    };

    double obme_pp, obme_nn, obme_np, obme_pn;
    for (int i1 = 0; i1 < num_nlj; ++i1)
    {
      const auto& o1 = orbit_map.getOrbit(2 * i1 + 1);
      for (int i2 = 0; i2 < num_nlj; ++i2)
      {
        const auto& o2 = orbit_map.getOrbit(2 * i2 + 1);

        if (((o1.l_ + o2.l_ + parity_) % 2) != 0) continue;  // parity
        if (!triangle2j(o1.jj_, o2.jj_, JJ_)) continue;  // triangle

        next_d(obme_pp); next_d(obme_nn); next_d(obme_np); next_d(obme_pn);

        if (NN_only_) continue; // consume only
        if (o1.shell() > maxShell_ || o2.shell() > maxShell_) continue;

        const uint16_t ap = 2 * i1 + 1, an = 2 * i1 + 2;
        const uint16_t bp = 2 * i2 + 1, bn = 2 * i2 + 2;

        if (std::fabs(obme_pp) > tol) obme_map[pack_ab(ap, bp)] = obme_pp;
        if (std::fabs(obme_nn) > tol) obme_map[pack_ab(an, bn)] = obme_nn;
        if (std::fabs(obme_np) > tol) obme_map[pack_ab(an, bp)] = obme_np; // (an,bp)
        if (std::fabs(obme_pn) > tol) obme_map[pack_ab(ap, bn)] = obme_pn; // (ap,bn)
      }
    }

    // (5) two-body block: a; b<=a; c; d<=c; (e2max cuts); parity; Jab,Jcd
    double me[10];
    for (int i1 = 0; i1 < num_nlj; ++i1)
    {
      const auto& o1 = orbit_map.getOrbit(2 * i1 + 1);
      for (int i2 = 0; i2 <= i1; ++i2)
      {
        const auto& o2 = orbit_map.getOrbit(2 * i2 + 1);

        // check shell restriction
        if (o1.shell() + o2.shell() > file_e2max) continue;

        for (int i3 = 0; i3 < num_nlj; ++i3)
        {
          const auto& o3 = orbit_map.getOrbit(2 * i3 + 1);
          for (int i4 = 0; i4 <= i3; ++i4)
          {
            const auto& o4 = orbit_map.getOrbit(2 * i4 + 1);

            // check shell restriction
            if (o3.shell() + o4.shell() > file_e2max) continue;

            // check parity
            if (((o1.l_ + o2.l_ + o3.l_ + o4.l_ + parity_) % 2) != 0) continue;

            for (int Jab = std::abs(o1.jj_ - o2.jj_) / 2;
                     Jab <= (o1.jj_ + o2.jj_) / 2; ++Jab)
            {
              for (int Jcd = std::abs(o3.jj_ - o4.jj_) / 2;
                       Jcd <= (o3.jj_ + o4.jj_) / 2; ++Jcd)
              {
                if (!triangle2j(2 * Jab, 2 * Jcd, JJ_)) continue;

                for (int k = 0; k < 10; ++k) next_d(me[k]);

                // SAM model-space storage cuts (mirror read_from_binary_nuhamil)
                const int ea = o1.shell(), eb = o2.shell(),
                          ec = o3.shell(), ed = o4.shell();
                if (ea > maxShell_ || eb > maxShell_ || ec > maxShell_ || ed > maxShell_)
                  continue;

                int ab_Nhw = 0;
                if (ea > valenceShell_) ab_Nhw += (ea - valenceShell_);
                if (eb > valenceShell_) ab_Nhw += (eb - valenceShell_);
                if (ab_Nhw > Nmax_) continue;

                int cd_Nhw = 0;
                if (ec > valenceShell_) cd_Nhw += (ec - valenceShell_);
                if (ed > valenceShell_) cd_Nhw += (ed - valenceShell_);
                if (cd_Nhw > Nmax_) continue;

                const uint16_t ap = 2 * i1 + 1, an = 2 * i1 + 2;
                const uint16_t bp = 2 * i2 + 1, bn = 2 * i2 + 2;
                const uint16_t cp = 2 * i3 + 1, cn = 2 * i3 + 2;
                const uint16_t dp = 2 * i4 + 1, dn = 2 * i4 + 2;

                const qn_t JJab = static_cast<qn_t>(Jab);
                const qn_t JJcd = static_cast<qn_t>(Jcd);

                auto put = [&](double v, uint16_t a, uint16_t b,
                                         uint16_t c, uint16_t d)
                {
                  if (std::fabs(v) < tol) return;
/*
                  // Isospin change of THIS combo (mmt = +1 n, -1 p):
                  // 2*dTz = (mmt_a + mmt_b) - (mmt_c + mmt_d). Should match the
                  // operator's OpZ for every nonzero entry.
                  const int MMT = orbit_map.getOrbit(a).mmt_
                                + orbit_map.getOrbit(b).mmt_
                                - orbit_map.getOrbit(c).mmt_
                                - orbit_map.getOrbit(d).mmt_;
                  std::cout << "MMT = " << MMT << "\tME = " << v << "\n";
*/
                  me_map[{pack_abcd(a, b, c, d), pack_JJ(JJab, JJcd)}] = v;
                };

                put(me[0], ap, bp, cp, dp); // pppp
                put(me[1], ap, bp, cp, dn); // pppn
                put(me[2], ap, bp, cn, dp); // ppnp
                put(me[3], ap, bp, cn, dn); // ppnn
                put(me[4], ap, bn, cp, dn); // pnpn
                put(me[5], ap, bn, cn, dp); // pnnp
                put(me[6], ap, bn, cn, dn); // pnnn
                put(me[7], an, bp, cn, dp); // npnp
                put(me[8], an, bp, cn, dn); // npnn
                put(me[9], an, bn, cn, dn); // nnnn
              }
            }
          }
        }
      }
    }

    return;
  }

  void TwBME_general::clear()
  {
    decltype(me_map)().swap(me_map);
    decltype(obme_map)().swap(obme_map);
  }

  // Coupled ME: bra J = Jab, ket J = Jcd. NuHamil writes both bra<->ket
  // orderings, so only the in-pair (a<->b, c<->d) exchanges are reconstructed.
  double TwBME_general::getME(uint16_t a, uint16_t b, uint16_t c, uint16_t d,
                              qn_t Jab, qn_t Jcd) const
  {
    if (a == b && Jab % 2 == 1) return 0.0;
    if (c == d && Jcd % 2 == 1) return 0.0;
    if (u2::mult(2 * Jab, 2 * Jcd, JJ_) == 0) return 0.0;

    const auto& oa = orbit_map.getOrbit(static_cast<int>(a));
    const auto& ob = orbit_map.getOrbit(static_cast<int>(b));
    const auto& oc = orbit_map.getOrbit(static_cast<int>(c));
    const auto& od = orbit_map.getOrbit(static_cast<int>(d));

    // NuHamil stores both bra<->ket orderings, so only the in-pair (a<->b,
    // c<->d) exchanges are needed to reach the stored entry.
    const uint16_t jjkey = pack_JJ(Jab, Jcd);
    auto it = me_map.find({pack_abcd(a, b, c, d), jjkey});
    if (it != me_map.end()) return it->second;

    double phase_ab = (((oa.jj_ + ob.jj_) / 2 - Jab) % 2 == 0) ? -1.0 : 1.0;
    it = me_map.find({pack_abcd(b, a, c, d), jjkey});
    if (it != me_map.end()) return phase_ab * it->second;

    double phase_cd = (((oc.jj_ + od.jj_) / 2 - Jcd) % 2 == 0) ? -1.0 : 1.0;
    it = me_map.find({pack_abcd(a, b, d, c), jjkey});
    if (it != me_map.end()) return phase_cd * it->second;

    it = me_map.find({pack_abcd(b, a, d, c), jjkey});
    if (it != me_map.end()) return phase_ab * phase_cd * it->second;

    return 0.0; // sparse tensor operator: unstored channels are zero
  }

  // m-scheme ME via rank-OpJ Wigner-Eckart:
  //   <ab(m)|O|cd(m)> = sum_{Jab,Jcd} CG(ja jb Jab; mja mjb Mab)
  //                                 * CG(jc jd Jcd; mjc mjd Mcd)
  //                                 * CG(Jcd OpJ Jab; Mcd M0 Mab)
  //                                 * getME(a,b,c,d,Jab,Jcd)
  // Reduces to TwBME_scalar::getME(qn...) when OpJ=0,OpP=+1,OpZ=0 because
  // CG(J 0 J; M 0 M) = 1.
  double TwBME_general::getME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const
  {
#ifdef UNCOUPLE_ME_CACHE
    // Resolve this thread's cache once, then reuse the cache-passing overload.
    return getME(twbme_mscheme_cache(),
                 ea, lla, jja, mmja, mmta, eb, llb, jjb, mmjb, mmtb,
                 ec, llc, jjc, mmjc, mmtc, ed, lld, jjd, mmjd, mmtd);
#else
    return computeME(ea, lla, jja, mmja, mmta,
                     eb, llb, jjb, mmjb, mmtb,
                     ec, llc, jjc, mmjc, mmtc,
                     ed, lld, jjd, mmjd, mmtd);
#endif
  }

#ifdef UNCOUPLE_ME_CACHE
  // Cache-passing variant (see TwBME_scalar::getME with a cache). NOTE: the
  // cache is keyed only by the four m-scheme states, so it is correct only when
  // a single operator is active per program -- see app/TwoBodyOperatorJT.cpp.
  double TwBME_general::getME(TwBME_mscheme_cache_t& cache,
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const
  {
    TwBME_mscheme_key key = pack_mscheme_2Bstate(
      ea, lla, jja, mmja, mmta, eb, llb, jjb, mmjb, mmtb,
      ec, llc, jjc, mmjc, mmtc, ed, lld, jjd, mmjd, mmtd);

#ifdef MEASURE_MSCHEME_REUSE
    g_probed.insert(key);
    ++g_probes_this;
#endif

    auto it = cache.find(key);
    if (it != cache.end())
    {
#ifdef COUNT_MSCHEME_CACHE
      ++g_ms_hits[omp_get_thread_num()];
#endif
      return it->second;
    }
#ifdef COUNT_MSCHEME_CACHE
    ++g_ms_misses[omp_get_thread_num()];
#endif
    double v = computeME(ea, lla, jja, mmja, mmta,
                         eb, llb, jjb, mmjb, mmtb,
                         ec, llc, jjc, mmjc, mmtc,
                         ed, lld, jjd, mmjd, mmtd);
    cache.emplace(key, v);
    return v;
  }

  // Two-level lookup (see TwBME_scalar::getME with local+global).
  double TwBME_general::getME(TwBME_mscheme_cache_t& local,
      TwBME_mscheme_cache_t& global,
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const
  {
    TwBME_mscheme_key key = pack_mscheme_2Bstate(
      ea, lla, jja, mmja, mmta, eb, llb, jjb, mmjb, mmtb,
      ec, llc, jjc, mmjc, mmtc, ed, lld, jjd, mmjd, mmtd);

    auto it = local.find(key);
    if (it != local.end()) return it->second;

    double v = getME(global,
                     ea, lla, jja, mmja, mmta, eb, llb, jjb, mmjb, mmtb,
                     ec, llc, jjc, mmjc, mmtc, ed, lld, jjd, mmjd, mmtd);
    local.emplace(key, v);
    return v;
  }
#endif

  double TwBME_general::computeME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb,
      qn_t ec, qn_t llc, qn_t jjc, qn_t mmjc, qn_t mmtc,
      qn_t ed, qn_t lld, qn_t jjd, qn_t mmjd, qn_t mmtd) const
  {
    qn_t la = lla / 2, lb = llb / 2, lc = llc / 2, ld = lld / 2;

    // parity selection consistent with the operator
    if (((la + lb + lc + ld + parity_) % 2) != 0) return 0.0;

    qn_t MMJab = mmja + mmjb, MMJcd = mmjc + mmjd; // 2*M
    qn_t M0 = MMJab - MMJcd;                       // 2*M0 (operator projection)

    qn_t na = (ea - la) / 2, nb = (eb - lb) / 2,
         nc = (ec - lc) / 2, nd = (ed - ld) / 2;

    auto a = orbit_map.getIndex(na, la, jja, mmta),
         b = orbit_map.getIndex(nb, lb, jjb, mmtb),
         c = orbit_map.getIndex(nc, lc, jjc, mmtc),
         d = orbit_map.getIndex(nd, ld, jjd, mmtd);

    double norm_ab = (a == b) ? SQRT2 : 1.0, norm_cd = (c == d) ? SQRT2 : 1.0,
           norm = std::sqrt(static_cast<double>(JJ_) + 1.0);

    auto& cgcs = utils::cgcs();
    double x = 0.0;
    for (qn_t JJab = std::abs(jja - jjb); JJab <= jja + jjb; JJab += 2)
    {
      if (a == b && JJab % 4 != 0) continue; // identical bra -> Jab even
      double cg_ab = cgcs.get({jja, jjb, JJab, mmja, mmjb, MMJab});
      if (cg_ab == 0.0) continue;

      for (qn_t JJcd = std::abs(jjc - jjd); JJcd <= jjc + jjd; JJcd += 2)
      {
        if (c == d && JJcd % 4 != 0) continue; // identical ket -> Jcd even
        double cg_cd = cgcs.get({jjc, jjd, JJcd, mmjc, mmjd, MMJcd});
        if (cg_cd == 0.0) continue;

        // Wigner-Eckart: CG(Jcd OpJ Jab; Mcd M0 Mab), all in 2x units
        double cg_we = cgcs.get({JJcd, JJ_, JJab, MMJcd, M0, MMJab});
        if (cg_we == 0.0) continue;

        x += cg_ab * cg_cd * cg_we *
             getME(a, b, c, d, static_cast<qn_t>(JJab / 2),
                               static_cast<qn_t>(JJcd / 2));
      }
    }

#ifdef POSITIVE_INFINITY
    double phase = ((na + nb + nc + nd) % 2 == 0) ? 1.0 : -1.0;
    return phase * x * norm_ab * norm_cd / norm;
#else
    return x * norm_ab * norm_cd / norm;
#endif
  }

  // Coupled (reduced) one-body ME. The 1-body writer loops a,b over the full
  // orbit range, so both orderings are stored -> a direct lookup suffices.
  double TwBME_general::getME(uint16_t a, uint16_t b) const
  {
    assert(!NN_only_);
    auto it = obme_map.find(pack_ab(a, b));
    if (it != obme_map.end()) return it->second;
    return 0.0;
  }

  // m-scheme one-body ME via rank-OpJ Wigner-Eckart:
  //   <a ma| O^J0_M0 |b mb> = CG(jb OpJ ja; mb M0 ma) * <a||O^J0||b>,  M0 = ma-mb
  // Reduces to the scalar one-body ME at OpJ=0 since CG(j 0 j; m 0 m) = 1.
  double TwBME_general::getME(
      qn_t ea, qn_t lla, qn_t jja, qn_t mmja, qn_t mmta,
      qn_t eb, qn_t llb, qn_t jjb, qn_t mmjb, qn_t mmtb) const
  {
    qn_t la = lla / 2, lb = llb / 2;

    // parity selection consistent with the operator
    if (((la + lb + parity_) % 2) != 0) return 0.0;

    qn_t na = (ea - la) / 2, nb = (eb - lb) / 2;
    auto a = orbit_map.getIndex(na, la, jja, mmta);
    auto b = orbit_map.getIndex(nb, lb, jjb, mmtb);

    qn_t M0 = mmja - mmjb; // 2*M0 (operator projection)

    // Wigner-Eckart: CG(jb OpJ ja; mb M0 ma), all in 2x units
    double cg = utils::cgcs().get({jjb, JJ_, jja, mmjb, M0, mmja});
    if (cg == 0.0) return 0.0;

    double R = getME(a, b); // stored reduced one-body ME

#ifdef POSITIVE_INFINITY
    double phase = ((na + nb) % 2 == 0) ? 1.0 : -1.0;
    return phase * cg * R;
#else
    return cg * R;
#endif
  }
} // namespace HO
