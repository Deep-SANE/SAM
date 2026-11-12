#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <convert.h>
#include <readwrite.h>
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>
#include <so3Coeffs/SO3_Fano.h>

#include <U3SO3xU3SO3.h>
#include <U4SU2SU2xU4SU2SU2.h>

#include <GlobalConstants.h>
#include <HOutilities.h>
#include <OneBodyOperatorJT.h>
#include <DataDirectory.h>
#include <ExpansionFiles.h>

namespace JT
{
  OneBodyOperator::OneBodyOperator(const std::string& name, int hw, int Nmax, 
                                   int valenceShell, int maxShell,
                                   const HO::Table_nljmjmt& HO_labels,
                                   const U3xU4::Table_gelfand& gelfand_labels)
    : HO_labels_(HO_labels), gelfand_labels_(gelfand_labels)
  {
    if (maxShell > 14)
      throw std::out_of_range("Sorry, we only have maxShell up to 14");

    hw_ = (hw >= 0) ? utils::ito8(hw) : 20;
    Nmax_ = (Nmax >= 0) ? utils::ito8(Nmax) : 14;
    valenceShell_ = (valenceShell > 0) ? utils::ito8(valenceShell) : 0;
    maxShell_ = (maxShell >= 0) ? utils::ito8(maxShell) : 14;

    name_ = name;
    JJ_ = 0;
    MMJ_ = 0;
    TT_ = 0;
    MMT_ = 0;
    parity_ = 0;
    on_the_fly = true;

    if (known_scalar_ops.count(name_)) { }
    else if (name_ == "E2") { JJ_ = 4; }
    else if (name_ == "M1")
    { 
      TT_ = -1;
      JJ_ = 2;
    }
    else if (name_ == "FermiPlus")
    {
      TT_ = 2;
      MMT_ = 2;
    }
    else if (name_ == "FermiMinus")
    {
      TT_ = 2;
      MMT_ = -2;
    }
    else if (name_ == "GamowTellerPlus")
    {
      JJ_ = 2;
      TT_ = 2;
      MMT_ = 2;
    }
    else if (name_ == "GamowTellerMinus")
    {
      JJ_ = 2;
      TT_ = 2;
      MMT_ = -2;
    }
    else
    {
#ifdef USE_MPI
      int mpi_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

      if (mpi_rank == 0)
#endif
      std::cout << "WARNING: This operator isn't known" << std::endl;
      
      TT_ = -1;
      OBME.emplace(name_, hw, Nmax, valenceShell, maxShell);
      on_the_fly = false;
    }
  }

  // Check the type of this operator
  void OneBodyOperator::is() const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
    {
      if (name_ == "NumberOp")
        std::cout << "Number operator N" << std::endl;
      else if (name_ == "L.L")
        std::cout << "One-body part of L.L" << std::endl;
      else if (name_ == "L.S")
        std::cout << "One-body part of L.S" << std::endl;
      else if (name_ == "CoMNumberOp")
        std::cout << "One-body part of the CoM number operator without 1/A"
                  << std::endl;
      else if (name_ == "Kinetic")
        std::cout
            << "One-body part of the relative kinetic energy without (A-1)/A"
            << std::endl;
      else if (name_ == "Rrel")
        std::cout << "One-body part of the intrinsic mass radius squared without "
                     "(A-1)/A"
                  << std::endl;
      else if (name_ == "Radius")
        std::cout << "Point-nucleon radius" // squared without CoM correction 
                  << std::endl;
      else if (name_ == "Radius_p")
        std::cout << "Point-proton radius" // squared without CoM correction 
                  << std::endl;
      else if (name_ == "Radius_n")
        std::cout << "Point-neutron radius" // squared without CoM correction 
                  << std::endl;
      else if (name_ == "E2")
        std::cout << "E2 transition operator"
                  << std::endl;
      else if (name_ == "M1")
        std::cout << "M1 transition operator" << std::endl;
      else if (name_ == "FermiPlus")
        std::cout << "Fermi for beta+" << std::endl;
      else if (name_ == "FermiMinus")
        std::cout << "Fermi for beta-" << std::endl;
      else if (name_ == "GamowTellerPlus")
        std::cout << "Gamow-Teller for beta+" << std::endl;
      else if (name_ == "GamowTellerMinus")
        std::cout << "Gamow-Teller for beta-" << std::endl;
      else if (on_the_fly == false)
        std::cout << "One-body potential with " << name_ << std::endl;
    }

    if (name_ != "NumberOp" && name_ != "L.L" && name_ != "L.S" && 
        name_ != "CoMNumberOp" && name_ != "Kinetic" && name_ != "Rrel" &&
        name_ != "Radius" && name_ != "Radius_p" && name_ != "Radius_n" &&
        name_ != "E2" && name_ != "M1" && name_ != "FermiPlus" && 
        name_ != "FermiMinus" && name_ != "GamowTellerPlus" &&
        name_ != "GamowTellerMinus" && on_the_fly == true)
      throw std::invalid_argument("This operator has not been implemented");
  }

  std::ostream& operator<<(std::ostream& s, const OneBodyOperator& x)
  {
    if (x.name_ == "NumberOp")
      s << "Number operator N";
    else if (x.name_ == "L.L")
      s << "One-body part of L.L";
    else if (x.name_ == "L.S")
      s << "One-body part of L.S";
    else if (x.name_ == "CoMNumberOp")
      s << "One-body part of the CoM number operator without 1/A";
    else if (x.name_ == "Kinetic")
      s << "One-body part of the relative kinetic energy without (A-1)/A";
    else if (x.name_ == "Rrel")
      s << "One-body part of the intrinsic mass radius squared without (A-1)/A";
    else if (x.name_ == "Radius")
      s << "Point-nucleon radius"; // squared without CoM correction
    else if (x.name_ == "Radius_p")
      s << "Point-proton radius"; // squared without CoM correction
    else if (x.name_ == "Radius_n")
      s << "Point-neutron radius"; // squared without CoM correction
    else if (x.name_ == "E2")
      s << "E2 transition operator";
    else if (x.name_ == "M1")
      s << "M1 transition operator";
    else if (x.name_ == "FermiPlus")
      s << "Fermi for beta+";
    else if (x.name_ == "FermiMinus")
      s << "Fermi for beta-";
    else if (x.name_ == "GamowTellerPlus")
      s << "Gamow-Teller for beta+";
    else if (x.name_ == "GamowTellerMinus")
      s << "Gamow-Teller for beta-";
    else if (x.on_the_fly == false)
      s << "One-body potential with " << x.name_ << std::endl;
    else
      throw std::invalid_argument("This operator has not been implemented");

    return s;
  }

  std::string OneBodyOperator::dirname() const
  {
    std::string dir = sam::data_dir() + "/OneBodyOperatorJT/" + name_;

    std::filesystem::create_directories(dir);
    return dir;
  }

  // Write to binary file
  void OneBodyOperator::writeBinary(const std::string& filename, 
                     const U3xU4::MultiShellTensors_phys& tensors) const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::cerr << "Start writing " << name_ << " to binary file on rank " 
              << mpi_rank << std::endl;
#else
    std::cout << "Start writing " << name_ << " to binary file" << std::endl;
#endif
    
    if (valenceShell_ > 0)
      throw std::runtime_error("Set valenceShell to 0 for reusability");

    if (Nmax_ != maxShell_)
      throw std::runtime_error("Set Nmax and maxShell equal");

    std::ofstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for writing: " + filename);

    utils::writeBinary(s, maxShell_);

    // Number of tensors
    size_t n_tensors = tensors.size();
    utils::writeBinary(s, n_tensors);

    for (const auto& tensor : tensors) tensor.writeBinary(s);

#ifdef USE_MPI
    std::cerr << "Finish writing to binary file on rank " << mpi_rank << std::endl;
#else
    std::cout << "Finish writing to binary file" << std::endl;
#endif
  }

  // Read from binary file
  U3xU4::MultiShellTensors_phys OneBodyOperator::readBinary(
    const std::string& filename)
  {
    std::ifstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for reading: " + filename);

    HO::qn_t maxShell;
    utils::readBinary(s, maxShell);

#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::cerr << "Start reading " << name_
              << " from binary file with capacity maxShell = " 
              << static_cast<int>(maxShell) << " on rank " << mpi_rank << std::endl;
#else
    std::cout << "Start reading " << name_ 
              << " from binary file with capacity maxShell = " 
              << static_cast<int>(maxShell) << std::endl;
#endif
    
    // stop if the file does not reach this calculation's shells: a one-body
    // tensor a+_m a~_n needs both shells m, n <= maxShell_
    if (maxShell_ > maxShell)
      sam::throw_expansion_too_small(filename, maxShell, maxShell_,
                                     "OneBodyOperatorJT");

    // Read number of tensors
    size_t n_tensors;
    utils::readBinary(s, n_tensors);

    U3xU4::MultiShellTensors_phys tensors;
    tensors.reserve(n_tensors);

    for (size_t i = 0; i < n_tensors; ++i)
    { 
      U3xU4::MultiShellTensor_phys tensor(HO_labels_);
      tensor.readBinary(s);

      // tensor cannot reside beyond maxShell
      if (tensor.Gammas().back().shell() - valenceShell_ > Nmax_) continue;

      // tensor cannot shift more quanta than Nmax
      if (abs(tensor.shift_quanta()) > Nmax_) continue;

      if (on_the_fly)
      {
        if (name_ == "Kinetic")
          tensor.enhanceStrengths(static_cast<double>(hw_) / 20.0);
        else if (name_ == "Rrel" or name_ == "Radius" or name_ == "Radius_p" or 
                 name_ == "Radius_n" or name_ == "E2")
          tensor.enhanceStrengths(20.0 / static_cast<double>(hw_));
      }

      tensors.push_back(std::move(tensor));
    }
#ifdef USE_MPI
    std::cerr << "Finish reading from binary file on rank " << mpi_rank << std::endl;
#else
    std::cout << "Finish reading from binary file" << std::endl;
#endif

    return tensors;
  }

  // Matrix representation
  utils::CSRMatrix
  OneBodyOperator::matrix_CSR(const U3xU4::ManyNucleonBasis& basis)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements_CSR(tensors, basis, gelfand_labels_);
  }

  Eigen::SparseMatrix<double>
  OneBodyOperator::matrix_Eigen(const U3xU4::ManyNucleonBasis& basis)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements_Eigen(tensors, basis, gelfand_labels_);
  }

  Eigen::SparseMatrix<double> OneBodyOperator::matrix_Eigen(
    const U3xU4::ManyNucleonBasis& bras,
    const U3xU4::ManyNucleonBasis& kets)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements_Eigen(tensors, bras, kets, gelfand_labels_);
  }

  // Tensor Analyzers
  std::vector<std::pair<u3::irrep, sam::strength_stats>>
    OneBodyOperator::U3_tensor_analyzer_write()
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_U3.dat";

    auto tensors = expand();
    return sam::U3_tensor_analyzer(tensors, filename);
  }
  
  std::vector<std::pair<u3::irrep, sam::strength_stats>>
    OneBodyOperator::U3_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::U3_tensor_analyzer(tensors);
  }

  std::vector<std::pair<u4::irrep, sam::strength_stats>>
    OneBodyOperator::U4_tensor_analyzer_write()
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_U4.dat";

    auto tensors = expand();
    return sam::U4_tensor_analyzer(tensors, filename);
  }
  
  std::vector<std::pair<u4::irrep, sam::strength_stats>>
    OneBodyOperator::U4_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::U4_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    OneBodyOperator::LL_tensor_analyzer_write()
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_LL.dat";

    auto tensors = expand();
    return sam::LL_tensor_analyzer(tensors, filename);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    OneBodyOperator::LL_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::LL_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    OneBodyOperator::SS_tensor_analyzer_write()
  {
    std::string base_dir = dirname() + "/Analysis";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_SS.dat";

    auto tensors = expand();
    return sam::SS_tensor_analyzer(tensors, filename);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    OneBodyOperator::SS_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::SS_tensor_analyzer(tensors);
  }

} // namespace JT
