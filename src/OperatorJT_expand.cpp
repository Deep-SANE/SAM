#include <ranges>

#include <OperatorJT.h>

namespace JT
{
  U3xU4::MultiShellTensors_phys Operator::expand_old()
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
      if (name_ == "GamowTellerPlus_2B_N3LO_EM500_bare")
        throw std::runtime_error("Expansion unavailable");
      else if (name_ == "GamowTellerMinus_2B_N3LO_EM500_bare")
      {
        JT::OneBodyOperator 
          op1("GamowTellerMinus", hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
        auto tensors1 = op1.expand();

        JT::TwoBodyOperator 
          op2(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
        auto tensors2 = op2.expand();

        double coeff2 = - SQRT2 / gAx;

        U3xU4::MultiShellTensors_phys tensors;
        tensors.reserve(tensors1.size() + tensors2.size());

        for (auto& tensor1 : tensors1)
        {
          tensors.push_back(std::move(tensor1));
        }

        for (auto& tensor2 : tensors2)
        {
          tensor2.enhanceStrengths(coeff2);
          tensors.push_back(std::move(tensor2));
        }

        return tensors;
      }
      else
      {
        JT::OneBodyOperator 
          op1(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
        auto tensors1 = op1.expand();

        JT::TwoBodyOperator 
          op2(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
        auto tensors2 = op2.expand();

        double coeff1 = 0.0, coeff2 = 0.0;

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

      double A = N_ + Z_, com_coeff = Lawson / A, 
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

  U3xU4::MultiShellTensors_ShareGammas_phys Operator::expand()
  {
    if (one_body.count(name_))
    {
      JT::OneBodyOperator 
        op(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);

      auto op_tensors = op.expand();

      U3xU4::MultiShellTensors_ShareGammas_phys tensors;
      tensors.reserve(op_tensors.size());

      for (auto& op_tensor : op_tensors)
      {
        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(op_tensor))
          {
            tensor.addOmegasLLSSTTstrengths(op_tensor);
            found = true;
            break;
          }
        }
        if (!found)
          tensors.push_back(std::move(op_tensor));
      }
      return tensors;
    }
    else if (two_body.count(name_))
    {
      JT::TwoBodyOperator 
        op(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
      
      auto op_tensors = op.expand();

      U3xU4::MultiShellTensors_ShareGammas_phys tensors;
      tensors.reserve(op_tensors.size());

      for (auto& op_tensor : op_tensors)
      {
        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(op_tensor))
          {
            tensor.addOmegasLLSSTTstrengths(op_tensor);
            found = true;
            break;
          }
        }
        if (!found)
          tensors.push_back(std::move(op_tensor));
      }
      return tensors;
    }
    else if (one_plus_two_body.count(name_))
    {
      if (name_ == "GamowTellerPlus_2B_N3LO_EM500_bare")
        throw std::runtime_error("Expansion unavailable");
      else if (name_ == "GamowTellerMinus_2B_N3LO_EM500_bare")
      {
        JT::OneBodyOperator 
          op1("GamowTellerMinus", hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
        auto tensors1 = op1.expand();

        JT::TwoBodyOperator 
          op2(name_, hw_, Nmax_, valenceShell_, maxShell_, HO_labels_, gelfand_labels_);
        auto tensors2 = op2.expand();

        double coeff2 = -SQRT2 / gAx;

        U3xU4::MultiShellTensors_ShareGammas_phys tensors;
        tensors.reserve(tensors1.size() + tensors2.size());

        // List 1
        for (auto& tensor1 : tensors1)
        {
          bool found = false;
          for (auto& tensor : tensors | std::views::reverse)
          {
            if (tensor.shareGammas(tensor1))
            {
              tensor.addOmegasLLSSTTstrengths(tensor1);
              found = true;
              break;
            }
          }
          if (!found)
            tensors.push_back(std::move(tensor1));
        }

        // List 2
        for (auto& tensor2 : tensors2)
        {
          tensor2.enhanceStrengths(coeff2);
          bool found = false;
          for (auto& tensor : tensors | std::views::reverse)
          {
            if (tensor.shareGammas(tensor2))
            {
              tensor.addOmegasLLSSTTstrengths(tensor2);
              found = true;
              break;
            }
          }
          if (!found)
            tensors.push_back(std::move(tensor2));
        }

        return tensors;
      }
      else
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

        U3xU4::MultiShellTensors_ShareGammas_phys tensors;
        tensors.reserve(tensors1.size() + tensors2.size());

        // List 1
        for (auto& tensor1 : tensors1)
        {
          if (tensor1.shift_quanta() < 0) continue;

          tensor1.enhanceStrengths(coeff1);
          bool found = false;
          for (auto& tensor : tensors | std::views::reverse)
          {
            if (tensor.shareGammas(tensor1))
            {
              tensor.addOmegasLLSSTTstrengths(tensor1);
              found = true;
              break;
            }
          }
          if (!found)
            tensors.push_back(std::move(tensor1));
        }

        // List 2
        for (auto& tensor2 : tensors2)
        {
          if (tensor2.shift_quanta() < 0) continue;

          tensor2.enhanceStrengths(coeff2);
          bool found = false;
          for (auto& tensor : tensors | std::views::reverse)
          {
            if (tensor.shareGammas(tensor2))
            {
              tensor.addOmegasLLSSTTstrengths(tensor2);
              found = true;
              break;
            }
          }
          if (!found)
            tensors.push_back(std::move(tensor2));
        }

        return tensors;
      }
    }
    else if (potential_name_ == "custom_potential")
    {
      JT::OneBodyOperator 
        CoM1("CoMNumberOp", hw_, Nmax_, valenceShell_, maxShell_,
                                    HO_labels_, gelfand_labels_),
        V1(name_, hw_, Nmax_, valenceShell_, maxShell_,
                                    HO_labels_, gelfand_labels_);

      auto com1s = CoM1.expand();
      auto v1s = V1.expand();

      JT::TwoBodyOperator 
        CoM2("CoMNumberOp", hw_, Nmax_, valenceShell_, maxShell_,
                                    HO_labels_, gelfand_labels_),
        V2(name_, hw_, Nmax_, valenceShell_, maxShell_,
                                    HO_labels_, gelfand_labels_);

      auto com2s = CoM2.expand();
      auto v2s = V2.expand();

      U3xU4::MultiShellTensors_ShareGammas_phys tensors;
      tensors.reserve(com1s.size() + com2s.size() + 
                v1s.size() + v2s.size());

      double A = N_ + Z_, com_coeff = Lawson / A;
  
      for (auto& com1 : com1s)
      {
        if (com1.shift_quanta() < 0) continue;

        com1.enhanceStrengths(com_coeff);
        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(com1))
          {
            tensor.addOmegasLLSSTTstrengths(com1);
            found = true;
            break;
          }
        }
        if (!found)
          tensors.push_back(std::move(com1));
      }

      for (auto& v1 : v1s)
      {
        if (v1.shift_quanta() < 0) continue;

        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(v1))
          {
            tensor.addOmegasLLSSTTstrengths(v1);
            found = true;
            break;
          }
        }
        if (!found)
          tensors.push_back(std::move(v1));
      }

      for (auto& com2 : com2s)
      {
        if (com2.shift_quanta() < 0) continue;

        com2.enhanceStrengths(com_coeff);
        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(com2))
          {
            tensor.addOmegasLLSSTTstrengths(com2);
            found = true;
            break;
          }
        }
        if (!found)
          tensors.push_back(std::move(com2));
      }

      for (auto& v2 : v2s)
      {
        if (v2.shift_quanta() < 0) continue;

        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(v2))
          {
            tensor.addOmegasLLSSTTstrengths(v2);
            found = true;
            break;
          }
        }
        if (!found)
          tensors.push_back(std::move(v2));
      }

      return tensors;
    }
    else // Hamiltonian with known potential
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

      U3xU4::MultiShellTensors_ShareGammas_phys tensors;
      tensors.reserve(com1s.size() + com2s.size() + 
                ke1s.size() + ke2s.size() + nns.size());

      double A = N_ + Z_, com_coeff = Lawson / A, 
           ke1_coeff = (A - 1.0) / A, ke2_coeff = 1.0 / A;

      for (auto& com1 : com1s)
      {
        if (com1.shift_quanta() < 0) continue;

        com1.enhanceStrengths(com_coeff);
        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(com1))
          {
            tensor.addOmegasLLSSTTstrengths(com1);
            found = true;
            break;
          }
        }
        if (!found)
          tensors.push_back(std::move(com1));
      }

      for (auto& ke1 : ke1s)
      {
        if (ke1.shift_quanta() < 0) continue;

        ke1.enhanceStrengths(ke1_coeff);
        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(ke1))
          {
            tensor.addOmegasLLSSTTstrengths(ke1);
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
        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(com2))
          {
            tensor.addOmegasLLSSTTstrengths(com2);
            found = true;
            break;
          }
        }
        if (!found)
          tensors.push_back(std::move(com2));
      }

      for (auto& ke2 : ke2s)
      {
        if (ke2.shift_quanta() < 0) continue;

        ke2.enhanceStrengths(ke2_coeff);
        bool found = false;
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(ke2))
          {
            tensor.addOmegasLLSSTTstrengths(ke2);
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
        for (auto& tensor : tensors | std::views::reverse)
        {
          if (tensor.shareGammas(nn))
          {
            tensor.addOmegasLLSSTTstrengths(nn);
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

} // namespace JT
