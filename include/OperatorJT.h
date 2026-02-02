#pragma once

#include <utility>

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
    const HO::Table_nljmjmt& HO_labels_;
    const U3xU4::Table_gelfand& gelfand_labels_;

    int N_, Z_, hw_, Nmax_, valenceShell_, maxShell_;

    std::string name_;

    const std::unordered_set<std::string> one_body = {"E2", "M1",
        "FermiPlus", "FermiMinus", "GamowTellerPlus", "GamowTellerMinus",
        "Radius", "Radius_p", "Radius_n"};

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
        "L.L", "L.S", "CoMNumberOp", "Kinetic", "Rrel"};

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

    // Get binary file name for writing matrix
    std::string dirname() const;
    std::string filename(const U3xU4::ManyNucleonBasis& basis) const;

    // expand into a sum of U3xU4::MultiShellTensor_phys
    U3xU4::MultiShellTensors_phys expand_old();

    // expand into a sum of U3xU4::MultiShellTensor_phys
    U3xU4::MultiShellTensors_ShareGammas_phys expand();

    // Matrix representation
    // symmetric
    Eigen::SparseMatrix<double>
    matrix(const U3xU4::ManyNucleonBasis& basis);
    // rectangular
    Eigen::SparseMatrix<double>
    matrix(const U3xU4::ManyNucleonBasis& bras, 
           const U3xU4::ManyNucleonBasis& kets);

    // Expectation value
    using Col_t = Eigen::MatrixXd::ColXpr;
    // within a single basis; users can pass multiple eigenstate pairs
    std::vector<double> expectationValue(const U3xU4::ManyNucleonBasis& basis, 
                            const std::vector<std::pair<Col_t, Col_t>>& states);
    // between two bases; users can pass multiple eigenstate pairs
    std::vector<double> expectationValue(
      const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
      const std::vector<std::pair<Col_t, Col_t>>& states);

    // Transition strength
    // within a single basis; users can pass multiple eigenstate pairs
    std::vector<double> transitionStrength(const U3xU4::ManyNucleonBasis& basis, 
      const std::vector<std::pair<Col_t, Col_t>>& states);
    // between two bases; users can pass multiple eigenstate pairs
    std::vector<double> transitionStrength(
      const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
      const std::vector<std::pair<Col_t, Col_t>>& states);

  };
} // namespace JT
