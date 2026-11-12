#include <ranges>

#include <OperatorJT.h>
#include <DataDirectory.h>

namespace JT
{
  Operator::Operator(const std::string& name, int N, int Z, int hw, int Nmax,
                     const HO::Table_nljmjmt& HO_labels,
                     const U3xU4::Table_gelfand& gelfand_labels)
      : name_(name), N_(N), Z_(Z), hw_(hw), Nmax_(Nmax), 
        HO_labels_(HO_labels), gelfand_labels_(gelfand_labels)
  {
    valenceShell_ = HO::valenceShell(N + Z);
    maxShell_ = valenceShell_ + Nmax_;
    Lawson = 100.0;

    bool Hamiltonian_type = false;
    if (name.find("Hamiltonian") != std::string::npos)
    {
      for (const auto& potential : two_body)
      	if (name.find(potential) != std::string::npos)
        {
          Hamiltonian_type = true;
      	  potential_name_ = potential;
        }
    }

    if (!one_body.count(name) && !two_body.count(name) &&
        !one_plus_two_body.count(name) && !Hamiltonian_type)
      potential_name_ = "custom_potential";

    for (auto& c : name_) 
      if (c == ' ') c = '_';
  }

  void Operator::is() const
  {
#ifdef USE_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
#endif
    {
      if (one_body.count(name_))
      {
        JT::OneBodyOperator 
          op(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
        op.is();
      }
      else if (two_body.count(name_))
      {
        JT::TwoBodyOperator 
          op(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
        op.is();
      }
      else if (name_ == "L.L" || name_ == "L.S")
        std::cout << name_ << " operator" << std::endl;
    	else if (name_ == "CoMNumberOp")
        std::cout << "HO center-of-mass number operator" << std::endl;
    	else if (name_ == "Kinetic")
        std::cout << "Relative kinetic energy" << std::endl;
    	else if (name_ == "Rrel")
        std::cout << "Intrinsic mass root-mean-squared radius" << std::endl;
      else if (name_ == "Radius")
        std::cout << "Point-nucleon root-mean-squared radius" << std::endl;
      else if (name_ == "Radius_p")
        std::cout << "Point-proton root-mean-squared radius" << std::endl;
      else if (name_ == "Radius_n")
        std::cout << "Point-neutron root-mean-squared radius" << std::endl;
      else if (name_ == "GamowTellerMinus_2B_N3LO_EM500_bare")
        std::cout << "1+2B Gamow-Teller for beta- from N3LO_EM500_bare" << std::endl;
      else if (name_ == "GamowTellerPlus_2B_N3LO_EM500_bare")
        std::cout << "1+2B Gamow-Teller for beta+ from N3LO_EM500_bare\n"
                  << "WARNING: expand() is unavailable" << std::endl;
    	else
    	  std::cout << name_ << " with Lawson term x " << Lawson << std::endl;
    }
  }

  std::ostream& operator<<(std::ostream& s, const Operator& x)
  {
  	if (x.one_body.count(x.name_))
    {
      JT::OneBodyOperator op(x.name_, x.hw_, x.Nmax_, x.valenceShell_, x.maxShell_, 
                                                  x.HO_labels_, x.gelfand_labels_);
      s << op;
    }
    else if (x.two_body.count(x.name_))
    {
      JT::TwoBodyOperator op(x.name_, x.hw_, x.Nmax_, x.valenceShell_, x.maxShell_,
                                                  x.HO_labels_, x.gelfand_labels_);
      s << op;
    }
    else if (x.name_ == "L.L" || x.name_ == "L.S") s << x.name_ << " operator";
  	else if (x.name_ == "CoMNumberOp") s << "HO center-of-mass number operator";
  	else if (x.name_ == "Kinetic") s << "Relative kinetic energy";
  	else if (x.name_ == "Rrel") s << "Intrinsic mass root-mean-squared radius";
    else if (x.name_ == "Radius") s << "Point-nucleon root-mean-squared radius";
    else if (x.name_ == "Radius_p") s << "Point-proton root-mean-squared radius";
    else if (x.name_ == "Radius_n") s << "Point-neutron root-mean-squared radius";
    else if (x.name_ == "GamowTellerMinus_2B_N3LO_EM500_bare")
      s << "1+2B Gamow-Teller for beta- from N3LO_EM500_bare";
    else if (x.name_ == "GamowTellerPlus_2B_N3LO_EM500_bare")
      s << "1+2B Gamow-Teller for beta+ from N3LO_EM500_bare\n"
        << "WARNING: expand() is unavailable";
  	else s << x.name_ << " with Lawson term x " << x.Lawson;

  	return s;
  }

  std::string Operator::dirname() const
  {
    std::string dir = sam::data_dir() + "/OperatorJT/" + name_;
    std::filesystem::create_directories(dir);

    return dir;
  }
  
  std::string Operator::matrix_file(const U3xU4::ManyNucleonBasis& basis) const
  {
    std::string dir = dirname() + "/Z" + std::to_string(Z_) 
                                + "/N" + std::to_string(N_)
                                + "/JJ" + std::to_string(basis.JJ()) 
                                + "/Nmax" + std::to_string(Nmax_);

    std::filesystem::create_directories(dir);

    return dir + "/hw" + std::to_string(hw_) + ".mat";
  }

  std::string Operator::wfns_file(const U3xU4::ManyNucleonBasis& basis) const
  {
    std::string dir = dirname() + "/Z" + std::to_string(Z_) 
                                + "/N" + std::to_string(N_)
                                + "/JJ" + std::to_string(basis.JJ()) 
                                + "/Nmax" + std::to_string(Nmax_);

    std::filesystem::create_directories(dir);

    return dir + "/hw" + std::to_string(hw_) + ".wfns";
  }

  std::string Operator::matrix_file(const U3xU4::ManyNucleonBasis& basis,
                                    std::string custom_dir) const
  {
    std::string dir = dirname() + "/Z" + std::to_string(Z_) 
                                + "/N" + std::to_string(N_)
                                + "/JJ" + std::to_string(basis.JJ()) 
                                + "/Nmax" + std::to_string(Nmax_)
                                + "/" + custom_dir;

    std::filesystem::create_directories(dir);

    return dir + "/hw" + std::to_string(hw_) + ".mat";
  }

  std::string Operator::wfns_file(const U3xU4::ManyNucleonBasis& basis,
                                    std::string custom_dir) const
  {
    std::string dir = dirname() + "/Z" + std::to_string(Z_) 
                                + "/N" + std::to_string(N_)
                                + "/JJ" + std::to_string(basis.JJ()) 
                                + "/Nmax" + std::to_string(Nmax_)
                                + "/" + custom_dir;

    std::filesystem::create_directories(dir);

    return dir + "/hw" + std::to_string(hw_) + ".wfns";
  }

  // Matrix representation
  utils::CSRMatrix
  Operator::matrix_CSR(const U3xU4::ManyNucleonBasis& basis)
  {
    // auto tensors = expand_old();
    auto tensors = expand();
    return U3xU4::MatrixElements_CSR(tensors, basis, gelfand_labels_);
  }

  Eigen::SparseMatrix<double> 
  Operator::matrix_Eigen(const U3xU4::ManyNucleonBasis& basis)
  {
    // auto tensors = expand_old();
    auto tensors = expand();
    return U3xU4::MatrixElements_Eigen(tensors, basis, gelfand_labels_);
  }

  Eigen::SparseMatrix<double> Operator::matrix_Eigen(
    const U3xU4::ManyNucleonBasis& bras,
    const U3xU4::ManyNucleonBasis& kets)
  {
    if (name_ == "GamowTellerPlus_2B_N3LO_EM500_bare")
    {
      JT::Operator op_temp("GamowTellerMinus_2B_N3LO_EM500_bare",
                           N_, Z_, hw_, Nmax_, HO_labels_, gelfand_labels_);
      auto tensors = op_temp.expand();
      auto mat = U3xU4::MatrixElements_Eigen(tensors, kets, bras, gelfand_labels_);
      return mat.transpose();
    }
    else
    {
      // auto tensors = expand_old();
      auto tensors = expand();
      return U3xU4::MatrixElements_Eigen(tensors, bras, kets, gelfand_labels_);
    }
  }

  // Rectangular CSR, block-distributed across MPI ranks. Memory-lean
  // alternative to matrix_Eigen(bras, kets) for 2-body two-basis operators.
  //
  // NOTE on layout: for the general operator this returns the (bras_dim x
  // kets_dim) matrix. GamowTellerPlus (2B) = (GamowTellerMinus)^T; rather than
  // transpose the distributed CSR (which would need cross-rank communication)
  // this returns the Minus matrix in (kets_dim x bras_dim) layout, i.e. the
  // transpose of the requested operator. expectationValue (two-basis) accounts
  // for this by swapping the bra/ket roles in the vec-mat-vec.
  utils::CSRMatrix Operator::matrix_CSR(
    const U3xU4::ManyNucleonBasis& bras,
    const U3xU4::ManyNucleonBasis& kets)
  {
    if (name_ == "GamowTellerPlus_2B_N3LO_EM500_bare")
    {
      JT::Operator op_temp("GamowTellerMinus_2B_N3LO_EM500_bare",
                           N_, Z_, hw_, Nmax_, HO_labels_, gelfand_labels_);
      auto tensors = op_temp.expand();
      return U3xU4::MatrixElements_CSR(tensors, kets, bras, gelfand_labels_);
    }
    else
    {
      auto tensors = expand();
      return U3xU4::MatrixElements_CSR(tensors, bras, kets, gelfand_labels_);
    }
  }

} // namespace JT
