#pragma once

#include <qcx_Int.h>

#include <HOutilities.h>
#include <OneBodyOperatorJT.h>
#include <TwoBodyOperatorJT.h>
#include <U3xU4ManyNucleonBasis.h>

namespace JT
{
  class Operator
  {
  private:
    const HO::Table_nljmjmt& labels_;

    int N_, Z_, hw_, Nmax_, valenceShell_, maxShell_;

    std::string name_;

    const std::unordered_set<std::string> one_body = {"E2", "M1",
        "FermiPlus", "FermiMinus", "GamowTellerPlus", "GamowTellerMinus"};

    const std::unordered_set<std::string> two_body = {
        "N2LO_EMN500_bare",   "N3LO_EMN500_bare",   "N4LO_EMN500_bare",
        "N3LO_EM500_bare",    "N2LO_opt_bare",      "N2LO_sat_bare",
        "N2LO_EMN500_srg1.0", "N3LO_EMN500_srg1.0", "N4LO_EMN500_srg1.0",
        "N3LO_EM500_srg1.0",  "N2LO_opt_srg1.0",    "N2LO_sat_srg1.0",
        "N2LO_EMN500_srg2.0", "N3LO_EMN500_srg2.0", "N4LO_EMN500_srg2.0",
        "N3LO_EM500_srg2.0",  "N2LO_opt_srg2.0",    "N2LO_sat_srg2.0",
        "N2LO_EMN500_srg4.0", "N3LO_EMN500_srg4.0", "N4LO_EMN500_srg4.0",
        "N3LO_EM500_srg4.0",  "N2LO_opt_srg4.0",    "N2LO_sat_srg4.0"};

    const std::unordered_set<std::string> one_plus_two_body = {
        "L.L", "L.S", "CoMNumberOp", "Kinetic", "R.R"};

    // Note: if you want a Hamiltonian, just define
    // std::string name = "Hamiltonian with " + potential_name;
    std::string potential_name_;

  public:
    Operator(const std::string& name, int N, int Z, int hw, int Nmax,
             const HO::Table_nljmjmt& labels);

    // Print
    void is() const;
    friend std::ostream& operator<<(std::ostream& s, const Operator& x);

    // expand into a sum of U3xU4::MultiShellTensor_phys
    U3xU4::MultiShellTensors_phys expand(sam::Databases& dbs);

    // Matrix representation
    // symmetric
    Eigen::SparseMatrix<double>
    matrix(const U3xU4::ManyNucleonBasis& basis, sam::Databases& dbs);
    // rectangular
    Eigen::SparseMatrix<double>
    matrix(const U3xU4::ManyNucleonBasis& bras, 
           const U3xU4::ManyNucleonBasis& kets, sam::Databases& dbs);

    // Get binary file name for writing matrix
    std::string filename(const U3xU4::ManyNucleonBasis& basis) const
    {
      const char* sam_dir = std::getenv("SAM_DIR");
      if (!sam_dir)
        throw std::runtime_error("Environment variable SAM_DIR not set");
      std::string res = std::string(sam_dir) + "/data/OperatorJT/" + name_
                           + "/Z" + std::to_string(Z_) 
                           + "/N" + std::to_string(N_)
                           + "/JJ" + std::to_string(basis.JJ()) 
                           + "/Nmax" + std::to_string(Nmax_)
                           + "/hw" + std::to_string(hw_);
      return res;             
    }
  };
} // namespace JT
