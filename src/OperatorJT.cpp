#include <OperatorJT.h>

namespace JT
{
  Operator::Operator(const std::string& name, int N, int Z, int hw, int Nmax,
                     const HO::Table_nljmjmt& labels)
      : name_(name), N_(N), Z_(Z), hw_(hw), Nmax_(Nmax), labels_(labels)
  {
    valenceShell_ = HO::valenceShell(N + Z);
    maxShell_ = valenceShell_ + Nmax_;

    bool Hamiltonian_type = false;
    if (name.find("Hamiltonian") != std::string::npos)
    {
      for (const auto& potential : two_body)
      {
      	if (name.find(potential) != std::string::npos)
      	{
      	  potential_name_ = potential;
      	  Hamiltonian_type = true;
      	}
      }
    }

    if (!one_body.count(name) && !two_body.count(name) &&
        !one_plus_two_body.count(name) && !Hamiltonian_type)
      throw std::invalid_argument(
          "Name is incorrect or operator is not implemented");

    for (auto& c : name_) 
        if (c == ' ') c = '_';
  }

  void Operator::is() const
  {
    if (one_body.count(name_))
    {
      JT::OneBodyOperator op(name_, hw_, Nmax_, valenceShell_, maxShell_, labels_);
      op.is();
    }
    else if (two_body.count(name_))
    {
      JT::TwoBodyOperator op(name_, hw_, Nmax_, valenceShell_, maxShell_, labels_);
      op.is();
    }
    else if (name_ == "L.L" || name_ == "L.S")
      std::cout << name_ << " operator" << std::endl;
  	else if (name_ == "CoMNumberOp")
      std::cout << "HO center-of-mass number operator" << std::endl;
  	else if (name_ == "Kinetic")
      std::cout << "Relative kinetic energy" << std::endl;
  	else if (name_ == "R.R")
      std::cout << "Intrinsic mass radius squared" << std::endl;
  	else
  	  std::cout << name_ << " with Lawson term x 1000" << std::endl;
  }

  std::ostream& operator<<(std::ostream& s, const Operator& x)
  {
  	if (x.one_body.count(x.name_))
    {
      JT::OneBodyOperator 
      	op(x.name_, x.hw_, x.Nmax_, x.valenceShell_, x.maxShell_, x.labels_);
      s << op;
    }
    else if (x.two_body.count(x.name_))
    {
      JT::TwoBodyOperator 
      	op(x.name_, x.hw_, x.Nmax_, x.valenceShell_, x.maxShell_, x.labels_);
      s << op;
    }
    else if (x.name_ == "L.L" || x.name_ == "L.S") s << x.name_ << " operator";
  	else if (x.name_ == "CoMNumberOp") s << "HO center-of-mass number operator";
  	else if (x.name_ == "Kinetic") s << "Relative kinetic energy";
  	else if (x.name_ == "R.R") s << "Intrinsic mass radius squared";
  	else s << x.name_ << " with Lawson term x 1000";

  	return s;
  }

  std::string Operator::dirname() const
  {
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");
      
    std::string dir = std::string(sam_dir) + "/data/OperatorJT/" + name_;
    std::filesystem::create_directories(dir);

    return dir;
  }
  
  std::string Operator::filename(const U3xU4::ManyNucleonBasis& basis) const
  {
    std::string dir = dirname() + "/Z" + std::to_string(Z_) 
                                + "/N" + std::to_string(N_)
                                + "/JJ" + std::to_string(basis.JJ()) 
                                + "/Nmax" + std::to_string(Nmax_);

    std::filesystem::create_directories(dir);

    return dir + "/hw" + std::to_string(hw_);       
  }

  U3xU4::MultiShellTensors_phys Operator::expand()
  {
  	if (one_body.count(name_))
    {
      JT::OneBodyOperator op(name_, hw_, Nmax_, valenceShell_, maxShell_, labels_);
      return op.expand();
    }
    else if (two_body.count(name_))
    {
      JT::TwoBodyOperator op(name_, hw_, Nmax_, valenceShell_, maxShell_, labels_);
      return op.expand();
    }
    else if (one_plus_two_body.count(name_))
    {
      JT::OneBodyOperator op1(name_, hw_, Nmax_, valenceShell_, maxShell_, labels_);
      auto tensors1 = op1.expand();

      JT::TwoBodyOperator op2(name_, hw_, Nmax_, valenceShell_, maxShell_, labels_);
      auto tensors2 = op2.expand();

      double coeff1, coeff2;

      if (name_ == "L.L" || name_ == "L.S")
      {
      	coeff1 = 1.0;
      	coeff2 = 1.0;
      }
      else if (name_ == "CoMNumberOp")
      {
      	double A = N_ + Z_;
      	coeff1 = 1.0 / A;
      	coeff2 = 1.0 / A;
      }
      else if (name_ == "Kinetic")
      {
      	double A = N_ + Z_;
      	coeff1 = (A - 1.0) / A;
      	coeff2 = 1.0 / A;
      }
      else if (name_ == "R.R")
      {
      	double A = N_ + Z_;
      	coeff1 = (A - 1.0) / A;
      	coeff2 = -1.0 / A;
      }

      U3xU4::MultiShellTensors_phys tensors;
      tensors.reserve(tensors1.size() + tensors2.size());

      for (auto& tensor1 : tensors1)
      {
      	tensor1.enhanceStrengths(coeff1);
      	tensors.push_back(std::move(tensor1));
      }

      for (auto& tensor2 : tensors2)
      {
      	tensor2.enhanceStrengths(coeff2);
      	tensors.push_back(std::move(tensor2));
      }

      return tensors;
    }
    else // Hamiltonian
    {
      JT::OneBodyOperator 
      	CoM1("CoMNumberOp", hw_, Nmax_, valenceShell_, maxShell_, labels_),
      	KE1("Kinetic", hw_, Nmax_, valenceShell_, maxShell_, labels_);

      auto com1s = CoM1.expand();
      auto ke1s = KE1.expand();

  	  JT::TwoBodyOperator 
  	  	CoM2("CoMNumberOp", hw_, Nmax_, valenceShell_, maxShell_, labels_),
      	KE2("Kinetic", hw_, Nmax_, valenceShell_, maxShell_, labels_),
      	NN(potential_name_, hw_, Nmax_, valenceShell_, maxShell_, labels_);

      auto com2s = CoM2.expand();
      auto ke2s = KE2.expand();
      auto nns = NN.expand();

      U3xU4::MultiShellTensors_phys tensors;
      tensors.reserve(com1s.size() + com2s.size() + 
      				  ke1s.size() + ke2s.size() + nns.size());

      double A = N_ + Z_, com_coeff = 1000.0 / A, 
      		 ke1_coeff = (A - 1.0) / A, ke2_coeff = 1.0 / A;

      for (auto& com1 : com1s)
      {
      	com1.enhanceStrengths(com_coeff);
      	tensors.push_back(std::move(com1));
      }

      for (auto& ke1 : ke1s)
      {
      	ke1.enhanceStrengths(ke1_coeff);
      	bool found = false;
      	for (auto& tensor : tensors)
      	{
      	  if (ke1 == tensor)
      	  {
      	  	tensor.addLLSSTTstrengths(ke1.LLSSTTstrengths());
      	  	found = true;
      	  	break;
      	  }
      	}
      	if (!found)
      	  tensors.push_back(std::move(ke1));
      }

      for (auto& com2 : com2s)
      {
      	com2.enhanceStrengths(com_coeff);
      	tensors.push_back(std::move(com2));
      }

      for (auto& ke2 : ke2s)
      {
      	ke2.enhanceStrengths(ke2_coeff);
      	bool found = false;
      	for (auto& tensor : tensors)
      	{
      	  if (ke2 == tensor)
      	  {
      	  	tensor.addLLSSTTstrengths(ke2.LLSSTTstrengths());
      	  	found = true;
      	  	break;
      	  }
      	}
      	if (!found)
      	  tensors.push_back(std::move(ke2));
      }

      for (auto& nn : nns)
      {
      	bool found = false;
      	for (auto& tensor : tensors)
      	{
      	  if (nn == tensor)
      	  {
      	  	tensor.addLLSSTTstrengths(nn.LLSSTTstrengths());
      	  	found = true;
      	  	break;
      	  }
      	}
      	if (!found)
      	  tensors.push_back(std::move(nn));
      }

      return tensors;
    }
  }

  // Matrix representation
  Eigen::SparseMatrix<double> Operator::matrix(
    const U3xU4::ManyNucleonBasis& basis)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements(tensors,basis);
  }
  Eigen::SparseMatrix<double> Operator::matrix(
    const U3xU4::ManyNucleonBasis& bras,
    const U3xU4::ManyNucleonBasis& kets)
  {
    auto tensors = expand();
    return U3xU4::MatrixElements(tensors, bras, kets);
  }
} // namespace JT
