#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <omp.h>

#include <convert.h>
#include <print.h>
using utils::print;
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>
#include <so3Coeffs/SO3_Fano.h>

#include <sam.h>

#include <TwoBodyOperatorJT.h>
#include <DataDirectory.h>
#include <ExpansionFiles.h>

namespace JT
{
  TwoBodyOperator::TwoBodyOperator(const std::string& name, int hw, int Nmax, 
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
    symmetric_ = true;

    if (known_scalar_ops.count(name)) 
    {
      on_the_fly = true;
    }
    else if (known_potentials.count(name))
    {
      TT_ = -1;
      TBME_scalar.emplace(name, hw, Nmax, valenceShell, maxShell);
      on_the_fly = false;
    }
    else if (known_general_ops.count(name))
    {
      TBME_general.emplace(name, hw, Nmax, valenceShell, maxShell);
      on_the_fly = false;
      symmetric_ = false;

      if (name.find("GamowTellerMinus") != std::string::npos)
      {
        JJ_ = 2;
        TT_ = 2;
        MMT_ = -2;
      }
    }
    else // Unknown nuclear potentials, e.g., from NO2B truncation
    {
#ifdef USE_MPI
      int mpi_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

      if (mpi_rank == 0)
#endif
      std::cout << "WARNING: This operator isn't known" << std::endl;

      TT_ = -1;
      on_the_fly = false;

      TBME_scalar.emplace(name, hw, Nmax, valenceShell, maxShell);
      // TBME_general.emplace(name, hw, Nmax, valenceShell, maxShell);   
    }
  }

  void TwoBodyOperator::is() const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
    {
      if (name_ == "L.L")
        std::cout << "Two-body part of L.L" << std::endl;
      else if (name_ == "L.S")
        std::cout << "Two-body part of L.S" << std::endl;
      else if (name_ == "CoMNumberOp")
        std::cout << "Two-body part of CoM number operator without 1/A"
                  << std::endl;
      else if (name_ == "Kinetic")
        std::cout << "Two-body part of the relative kinetic energy without 1/A"
                  << std::endl;
      else if (name_ == "Rrel")
        std::cout
            << "Two-body part of the intrinsic mass radius squared without -1/A"
            << std::endl;
      else if (on_the_fly == false)
        std::cout << "Two-body operator: " << name_ << std::endl;
    }

    if (name_ != "L.L" && name_ != "L.S" && name_ != "CoMNumberOp" &&
        name_ != "Kinetic" && name_ != "Rrel" && on_the_fly == true)
      throw std::invalid_argument("This operator has not been implemented");
  }

  std::ostream& operator<<(std::ostream& s, const TwoBodyOperator& x)
  {
    if (x.name_ == "L.L")
      s << "Two-body part of L.L";
    else if (x.name_ == "L.S")
      s << "Two-body part of L.S";
    else if (x.name_ == "CoMNumberOp")
      s << "Two-body part of CoM number operator without 1/A";
    else if (x.name_ == "Kinetic")
      s << "Two-body part of the relative kinetic energy without 1/A";
    else if (x.name_ == "Rrel")
      s << "Two-body part of the intrinsic mass radius squared without -1/A";
    else if (x.on_the_fly == false)
      s << "Two-body operator: " << x.name_;
    else
      throw std::invalid_argument("This operator has not been implemented");

    return s;
  }

  std::string TwoBodyOperator::dirname() const
  {
    std::string dir = sam::data_dir() + "/TwoBodyOperatorJT/" + name_;
    
    std::filesystem::create_directories(dir);
    return dir;
  }

  // Write to binary file
  void TwoBodyOperator::writeBinary(const std::string& filename, 
                     const U3xU4::MultiShellTensors_phys& tensors) const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::cerr << "Start writing " << name_ 
              << " to binary file on rank " << mpi_rank << std::endl;
#else
    std::cout << "Start writing " << name_ << " to binary file" << std::endl;
#endif

    if (valenceShell_ > 0) throw std::runtime_error("Set valenceShell to 0 for reusability");

    // if (Nmax_ != maxShell_) hrow std::runtime_error("Set Nmax and maxShell equal");

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
  U3xU4::MultiShellTensors_phys TwoBodyOperator::readBinary(
    const std::string& filename)
  {
    std::ifstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for reading: " + filename);

    HO::qn_t maxShell;
    utils::readBinary(s, maxShell);

    // stop if the file does not reach this calculation's shells
    if ((maxShell_ > maxShell) or (maxShell_ - 2 * valenceShell_ > maxShell))
      sam::throw_expansion_too_small(filename, maxShell, maxShell_,
                                     "TwoBodyOperatorJT");

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

    // Read number of tensors
    size_t n_tensors;
    utils::readBinary(s, n_tensors);

    U3xU4::MultiShellTensors_phys tensors;
    tensors.reserve(n_tensors);

    for (size_t i = 0; i < n_tensors; ++i)
    { 
      U3xU4::MultiShellTensor_phys tensor(HO_labels_);
      tensor.readBinary(s);

      int ab_Nhw = 0, cd_Nhw = 0, ads = 0, ats = 0;
      for (const auto& G : tensor.Gammas())
      {
        ab_Nhw += ((G.shell() - valenceShell_) * G.num_ads());
        cd_Nhw += ((G.shell() - valenceShell_) * G.num_ats());

        ads += G.num_ads();
        ats += G.num_ats();
      }
      
      // number of quanta shifted by a+'s and a-'s cannot exceed Nmax
      if (ab_Nhw > Nmax_ || cd_Nhw > Nmax_ || abs(ab_Nhw - cd_Nhw) > Nmax_) 
        continue;

      assert(ads == 2);
      assert(ats == 2);

      tensors.push_back(std::move(tensor));
    }

#ifdef USE_MPI
    std::cerr << "Finish reading to binary file on rank " << mpi_rank << std::endl;
#else
    std::cout << "Finish reading to binary file" << std::endl;
#endif

    scaleStrengths(tensors);
    return tensors;
  }

  U3xU4::MultiShellTensors_phys TwoBodyOperator::readBinary_MPI(
    const std::string& filename)
  {
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::ifstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for reading: " + filename);

    HO::qn_t maxShell; /* = Nmax */
    utils::readBinary(s, maxShell);

    // stop if the file does not reach this calculation's shells
    if ((maxShell_ > maxShell) or (maxShell_ - 2 * valenceShell_ > maxShell))
      sam::throw_expansion_too_small(filename, maxShell, maxShell_,
                                     "TwoBodyOperatorJT");

    std::cerr << "Start reading " << name_ 
              << " from binary file with capacity maxShell = " 
              << static_cast<int>(maxShell) << " on rank " << mpi_rank << std::endl;

    // Read number of tensors
    size_t n_tensors;
    utils::readBinary(s, n_tensors);

    U3xU4::MultiShellTensors_phys tensors;
    tensors.reserve(n_tensors);

    for (size_t i = 0; i < n_tensors; ++i)
    { 
      U3xU4::MultiShellTensor_phys tensor(HO_labels_);
      tensor.readBinary(s);

      int ab_Nhw = 0, cd_Nhw = 0, ads = 0, ats = 0;
      for (const auto& G : tensor.Gammas())
      {
        ab_Nhw += ((G.shell() - valenceShell_) * G.num_ads());
        cd_Nhw += ((G.shell() - valenceShell_) * G.num_ats());

        ads += G.num_ads();
        ats += G.num_ats();
      }
      
      // number of quanta shifted by a+'s and a-'s cannot exceed Nmax
      if (ab_Nhw > Nmax_ || cd_Nhw > Nmax_ || abs(ab_Nhw - cd_Nhw) > Nmax_) 
        continue;

      assert(ads == 2);
      assert(ats == 2);

      tensors.push_back(std::move(tensor));
    }

    std::cerr << "Finish reading from binary file on rank " << mpi_rank << std::endl;

    scaleStrengths(tensors);
    return tensors;
  }

  // Matrix representation
  utils::CSRMatrix 
  TwoBodyOperator::matrix_CSR(const U3xU4::ManyNucleonBasis& basis)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements_CSR(tensors, basis, gelfand_labels_);
  }

  Eigen::SparseMatrix<double> 
  TwoBodyOperator::matrix_Eigen(const U3xU4::ManyNucleonBasis& basis)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements_Eigen(tensors, basis, gelfand_labels_);
  }
  
  Eigen::SparseMatrix<double> TwoBodyOperator::matrix_Eigen(
    const U3xU4::ManyNucleonBasis& bras, 
    const U3xU4::ManyNucleonBasis& kets)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements_Eigen(tensors, bras, kets, gelfand_labels_);
  }

  // Tensor Analyzers
  std::vector<std::pair<u3::irrep, sam::strength_stats>>
    TwoBodyOperator::U3_tensor_analyzer_write()
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
    TwoBodyOperator::U3_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::U3_tensor_analyzer(tensors);
  }

  std::vector<std::pair<u4::irrep, sam::strength_stats>>
    TwoBodyOperator::U4_tensor_analyzer_write()
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
    TwoBodyOperator::U4_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::U4_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::LL_tensor_analyzer_write()
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
    TwoBodyOperator::LL_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::LL_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    TwoBodyOperator::SS_tensor_analyzer_write()
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
    TwoBodyOperator::SS_tensor_analyzer()
  {
    auto tensors = expand();
    return sam::SS_tensor_analyzer(tensors);
  }
} // namespace JT
