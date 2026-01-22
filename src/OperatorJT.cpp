#include <OperatorJT.h>

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
  	else
  	  std::cout << name_ << " with Lawson term x 1000" << std::endl;
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
      JT::OneBodyOperator 
        op(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
      return op.expand();
    }
    else if (two_body.count(name_))
    {
      JT::TwoBodyOperator 
        op(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
      return op.expand();
    }
    else if (one_plus_two_body.count(name_))
    {
      JT::OneBodyOperator 
        op1(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
      auto tensors1 = op1.expand();

      JT::TwoBodyOperator 
        op2(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
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
      else if (name_ == "Rrel")
      {
      	double A = N_ + Z_;
      	coeff1 = (A - 1.0) / A;
      	coeff2 = -1.0 / A;
      }

      U3xU4::MultiShellTensors_phys tensors;
      tensors.reserve(tensors1.size() + tensors2.size());

      for (auto& tensor1 : tensors1)
      {
        if (tensor1.shift_quanta() < 0) continue;

      	tensor1.enhanceStrengths(coeff1);
      	tensors.push_back(std::move(tensor1));
      }

      for (auto& tensor2 : tensors2)
      {
        if (tensor2.shift_quanta() < 0) continue;

      	tensor2.enhanceStrengths(coeff2);
      	tensors.push_back(std::move(tensor2));
      }

      return tensors;
    }
    else // Hamiltonian
    {
      // For Hamiltonian-type operators, the matrix is computed only in 
      // the lower triangle.
      // Recall this condition: if (bra_id < ket_id) continue;
      // This means that the number of quanta carried by bra irrep is always
      // larger or equal than that of the ket irrep. Thus the tensors must
      // always either create or keep quanta.

      JT::OneBodyOperator 
      	CoM1("CoMNumberOp", hw_, Nmax_, valenceShell_, maxShell_,
                                    HO_labels_, gelfand_labels_),
      	KE1("Kinetic", hw_, Nmax_, valenceShell_, maxShell_,
                                    HO_labels_, gelfand_labels_);

      auto com1s = CoM1.expand();
      auto ke1s = KE1.expand();

  	  JT::TwoBodyOperator 
  	  	CoM2("CoMNumberOp", hw_, Nmax_, valenceShell_, maxShell_,
                                    HO_labels_, gelfand_labels_),
      	KE2("Kinetic", hw_, Nmax_, valenceShell_, maxShell_,
                                    HO_labels_, gelfand_labels_),
      	NN(potential_name_, hw_, Nmax_, valenceShell_, maxShell_,
                                    HO_labels_, gelfand_labels_);

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
        if (com1.shift_quanta() < 0) continue;

      	com1.enhanceStrengths(com_coeff);
      	tensors.push_back(std::move(com1));
      }

      for (auto& ke1 : ke1s)
      {
        if (ke1.shift_quanta() < 0) continue;

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
        if (com2.shift_quanta() < 0) continue;

      	com2.enhanceStrengths(com_coeff);
      	tensors.push_back(std::move(com2));
      }

      for (auto& ke2 : ke2s)
      {
        if (ke2.shift_quanta() < 0) continue;

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
        if (nn.shift_quanta() < 0) continue;

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

    // return U3xU4::MatrixElements(tensors, basis);
    return U3xU4::MatrixElements(tensors, basis, gelfand_labels_);
  }
  Eigen::SparseMatrix<double> Operator::matrix(
    const U3xU4::ManyNucleonBasis& bras,
    const U3xU4::ManyNucleonBasis& kets)
  {
    auto tensors = expand();
    
    // return U3xU4::MatrixElements(tensors, bras, kets);
    return U3xU4::MatrixElements(tensors, bras, kets, gelfand_labels_);
  }

    // Expectation value
  using Col_t = Eigen::MatrixXd::ColXpr;
  // within a single basis; users can pass multiple eigenstate pairs
  std::vector<double> Operator::expectationValue(
    const U3xU4::ManyNucleonBasis& basis, 
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    std::vector<double> res;
    res.reserve(states.size());

    auto dim = basis.dim();
    
    auto mat = matrix(basis);
    
    for (auto& [bra, ket] : states)
    {
      assert(bra.size() == dim);
      assert(ket.size() == dim);
      double val = bra.transpose() * mat * ket;
      res.push_back(val);
    }

    if (name_ == "M1")
    {
      double factor = std::sqrt(4.0 * PI / 3.0);
      for (auto& r : res) r *= factor;
    }

    if (name_ == "Radius") 
    {
      double A = N_ + Z_, com = 1.5 * HO::oscilatorLength2(hw_, -1);
      for (auto& r2 : res) r2 = std::sqrt((r2 - com) / A);
    }

    if (name_ == "Radius_p") 
    {
      double A = N_ + Z_, com = 1.5 * HO::oscilatorLength2(hw_, -1);
      for (auto& r2 : res) r2 = std::sqrt(r2 / Z_ - com / A);
    }

    if (name_ == "Radius_n") 
    {
      double A = N_ + Z_, com = 1.5 * HO::oscilatorLength2(hw_, -1);
      for (auto& r2 : res) r2 = std::sqrt(r2 / N_ - com / A);
    }

    return res;
  }

  // transition between two bases; users can pass multiple eigenstate pairs
  std::vector<double> Operator::expectationValue(
    const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    std::vector<double> res;
    res.reserve(states.size());

    auto bras_dim = bras.dim();
    auto kets_dim = kets.dim();
    
    auto mat = matrix(bras, kets);

    for (auto& [bra, ket] : states)
    {
      assert(bra.size() == bras_dim);
      assert(ket.size() == kets_dim);
      double val = bra.transpose() * mat * ket;
      res.push_back(val);
    }

    return res;
  }

  // Transition strength
  // within a single basis; users can pass multiple eigenstate pairs
  std::vector<double> Operator::transitionStrength(
    const U3xU4::ManyNucleonBasis& basis, 
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    auto& cgcs = qcx_utils::cgcs();
    
    auto res = expectationValue(basis, states);
    
    double factor = 1.0;

    if (name_ == "E2")
    {
      auto JJ = basis.JJ();
      auto cgc = cgcs.get({JJ, 4, JJ, JJ, 0, JJ});
      factor = 5.0 / (16.0 * PI) / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }
    else if (name_ == "M1")
    {
      auto JJ = basis.JJ();
      auto cgc = cgcs.get({JJ, 2, JJ, JJ, 0, JJ});
      factor = 1.0 / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }

    return res;
  }
  // transition between two bases; users can pass multiple eigenstate pairs
  std::vector<double> Operator::transitionStrength(
    const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    auto& cgcs = qcx_utils::cgcs();

    auto JJf = bras.JJ();
    auto JJi = kets.JJ();
    auto MMJ = std::min(JJf, JJi);

    auto res = expectationValue(bras, kets, states);

    if (name_ == "E2")
    { 
      auto cgc = cgcs.get({JJi, 4, JJf, MMJ, 0, MMJ});
      double factor = 5.0 / (16.0 * PI) * (JJf + 1.0) / (JJi + 1.0) / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }
    else if (name_ == "M1")
    {
      auto cgc = cgcs.get({JJi, 2, JJf, MMJ, 0, MMJ});
      double factor = (JJf + 1.0) / (JJi + 1.0) / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }
    else if (name_ == "GamowTellerPlus" || name_ == "GamowTellerMinus")
    {
      auto cgc = cgcs.get({JJi, 2, JJf, MMJ, 0, MMJ});
      double factor = std::sqrt(JJf + 1.0) / cgc;
      for (auto& r : res) r *= factor;
    }

    return res;
  }
} // namespace JT
