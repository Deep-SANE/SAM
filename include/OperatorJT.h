#pragma once

#include <utility>

#include <Int.h>

#include <HOutilities.h>
#include <OneBodyOperatorJT.h>
#include <TwoBodyOperatorJT.h>
#include <U3xU4ManyNucleonBasis.h>

namespace JT
{
  class Operator
  {
  private:
    const HO::Table_nljmjmt& HO_labels_;
    const U3xU4::Table_gelfand& gelfand_labels_;

    int N_, Z_, hw_, Nmax_, valenceShell_, maxShell_;
    double Lawson;

    std::string name_;

    const std::unordered_set<std::string> one_body = {
      "NumberOp", "E2", "M1",
      "FermiPlus", "FermiMinus", "GamowTellerPlus", "GamowTellerMinus",
      "Radius", "Radius_p", "Radius_n"
    };

    const std::unordered_set<std::string> two_body = {
      "N2LO_EMN500_bare",   "N3LO_EMN500_bare",   "N4LO_EMN500_bare",
      "N3LO_EM500_bare",    "N2LO_opt_bare",      "N2LO_sat_bare",
      "N2LO_EMN500_srg1.0", "N3LO_EMN500_srg1.0", "N4LO_EMN500_srg1.0",
      "N3LO_EM500_srg1.0",  "N2LO_opt_srg1.0",    "N2LO_sat_srg1.0",
      "N2LO_EMN500_srg2.0", "N3LO_EMN500_srg2.0", "N4LO_EMN500_srg2.0",
      "N3LO_EM500_srg2.0",  "N2LO_opt_srg2.0",    "N2LO_sat_srg2.0",
      "N2LO_EMN500_srg4.0", "N3LO_EMN500_srg4.0", "N4LO_EMN500_srg4.0",
      "N3LO_EM500_srg4.0",  "N2LO_opt_srg4.0",    "N2LO_sat_srg4.0"
    };

    const std::unordered_set<std::string> one_plus_two_body = {
      "L.L", "L.S", "CoMNumberOp", "Kinetic", "Rrel",
      "GamowTellerMinus_2B_N3LO_EM500_bare",
      "GamowTellerPlus_2B_N3LO_EM500_bare"
    };

    // Note: if you want a Hamiltonian, just define
    // std::string name = "Hamiltonian with " + potential_name;
    std::string potential_name_;

  public:
    Operator(const std::string& name, int N, int Z, int hw, int Nmax,
             const HO::Table_nljmjmt& HO_labels, 
             const U3xU4::Table_gelfand& gelfand_labels);

    // Print
    void is() const;
    friend std::ostream& operator<<(std::ostream& s, const Operator& x);

    // Get binary file names
    std::string dirname() const;
    std::string matrix_file(const U3xU4::ManyNucleonBasis& basis) const;
    std::string wfns_file(const U3xU4::ManyNucleonBasis& basis) const;
    std::string matrix_file(const U3xU4::ManyNucleonBasis& basis,
                            std::string custom_dir) const;
    std::string wfns_file(const U3xU4::ManyNucleonBasis& basis,
                          std::string custom_dir) const;

    // expand into a sum of U3xU4::MultiShellTensor_phys
    U3xU4::MultiShellTensors_phys expand_old();

    // combine the expansions of the one- and two-body pieces; they are loaded
    // from the data directory with their expand(), which never expands and
    // stops with an error if a piece's file is missing or too small
    U3xU4::MultiShellTensors_ShareGammas_phys expand();

    // set scale for Lawson operator
    void set_Lawson(double beta) { Lawson = beta; }

    // Matrix representation
    // symmetric
    utils::CSRMatrix
    matrix_CSR(const U3xU4::ManyNucleonBasis& basis);

    Eigen::SparseMatrix<double>
    matrix_Eigen(const U3xU4::ManyNucleonBasis& basis);

    // rectangular
    utils::CSRMatrix
    matrix_CSR(const U3xU4::ManyNucleonBasis& bras,
               const U3xU4::ManyNucleonBasis& kets);
    
    Eigen::SparseMatrix<double>
    matrix_Eigen(const U3xU4::ManyNucleonBasis& bras,
                 const U3xU4::ManyNucleonBasis& kets);

    // Expectation value.  The plain names default to the CSR path (symmetric
    // block-distributed CSR for one basis, rectangular for two); the _Eigen
    // variants are the retiring dense-Eigen path.
    using Col_t = Eigen::MatrixXd::ColXpr;
    // within a single basis; users can pass multiple eigenstate pairs
    std::vector<double> expectationValue(const U3xU4::ManyNucleonBasis& basis,
                            const std::vector<std::pair<Col_t, Col_t>>& states);
    std::vector<double> expectationValue_Eigen(const U3xU4::ManyNucleonBasis& basis,
                            const std::vector<std::pair<Col_t, Col_t>>& states);
    // between two bases; users can pass multiple eigenstate pairs
    std::vector<double> expectationValue(
      const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
      const std::vector<std::pair<Col_t, Col_t>>& states);
    std::vector<double> expectationValue_Eigen(
      const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
      const std::vector<std::pair<Col_t, Col_t>>& states);

    // Transition strength (same CSR-default / _Eigen-retiring convention)
    // within a single basis; users can pass multiple eigenstate pairs
    std::vector<double> transitionStrength(const U3xU4::ManyNucleonBasis& basis,
      const std::vector<std::pair<Col_t, Col_t>>& states);
    std::vector<double> transitionStrength_Eigen(const U3xU4::ManyNucleonBasis& basis,
      const std::vector<std::pair<Col_t, Col_t>>& states);
    // between two bases; users can pass multiple eigenstate pairs
    std::vector<double> transitionStrength(
      const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
      const std::vector<std::pair<Col_t, Col_t>>& states);
    std::vector<double> transitionStrength_Eigen(
      const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
      const std::vector<std::pair<Col_t, Col_t>>& states);

  };
} // namespace JT
