#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <HOutilities.h>
#include <TensorAnalyzer.h>
#include <U3xU4ManyNucleonBasis.h>
#include <U3xU4MultiShellTensors.h>

namespace JT
{
  /* Class for holding one-body operator in JT-coupled scheme
   JJ TT					       (lam0 mu0) (P0 P0'P0'')
  O  		= sum of [Ta x Tb]
   MMJ MMT               kappa0 2L0 2S0 2T0 2MT0 2J0 2MJ0
   where MMJ = MMJ0 and MMT = MMT0
  */
  class OneBodyOperator
  {
  private:
    const HO::Table_nljmjmt& HO_labels_;
    const U3xU4::Table_gelfand& gelfand_labels_;

    HO::qn_t JJ_, MMJ_, TT_, MMT_, parity_, // (0=even,1=odd)
             hw_, Nmax_, valenceShell_, maxShell_;

    std::string name_;
    bool on_the_fly;

    std::optional<HO::OBME_scalar> OBME;

    const std::unordered_set<std::string> known_scalar_ops = {
        "NumberOp", "L.L", "L.S", "CoMNumberOp", "Kinetic", "Rrel", 
        "Radius", "Radius_p", "Radius_n"};

  public:
    // Constructor
    OneBodyOperator(const std::string& name, int hw, int Nmax, 
                    int valenceShell, int maxShell,
                    const HO::Table_nljmjmt& HO_labels,
                    const U3xU4::Table_gelfand& gelfand_labels);

    // Getters
    int JJ() const { return static_cast<int>(JJ_); }
    int MMJ() const { return static_cast<int>(MMJ_); }
    int TT() const { return static_cast<int>(TT_); }
    int MMT() const { return static_cast<int>(MMT_); }
    int parity() const { return static_cast<int>(parity_); }

    // Print
    void is() const;
    friend std::ostream& operator<<(std::ostream& s,
                                    const OneBodyOperator& x);

    // Name of the directory that contains data of this operator
    std::string dirname() const;

  private:
    // ME: < na lla jja mmja mmta | O | nb llb jjb mmjb mmtb >
    // recommended, since I haven't implemented RME for all operators
    double sphericalME(int na, int lla, int jja, int mmja, int mmta, int nb,
                       int llb, int jjb, int mmjb, int mmtb) const;
  
    // RME:             < na lla jja || O || nb llb jjb >
    // = ME / ( < jb mjb ; J MJ | ja mja > * < 1/2 mtb ; T MT | 1/2 mta > )
    double sphericalRME(int na, int lla, int jja, int nb, int llb, int jjb) const;

    // get strengths for U3xU4::OneBodyMultiShellTensor
    std::pair<bool, std::vector<double>> getTensorStrength_ME(
        int lam0, int mu0, int P10, int P20, int P30, int kappa0, int LL0,
        int SS0, int na, int nb) const;

    std::pair<bool, std::vector<double>> getTensorStrength_RME(
        int lam0, int mu0, int P10, int P20, int P30, int kappa0, int LL0,
        int SS0, int na, int nb) const;

    // set strengths for U3xU4::MultiShellTensor_phys, use ME
    void setTensorStrength(U3xU4::MultiShellTensor_phys& tensor,
                           int na, int nb) const;

    // selection rule
    bool LLselection(const int& LL) const;
    bool SSselection(const int& SS) const;
    bool U3selection(const u3::irrep& irr) const;
    bool U4selection(const u4::irrep& irr) const;

  public:
    // Expand into a sum of U3xU4::OneBodyMultiShellTensor
    U3xU4::OneBodyMultiShellTensors expand_old();

    // Expand into a sum of U3xU4::MultiShellTensor_phys. This is the expansion
    // step, called from the expansion program (app/tutorial/OneBodyOperatorJT.cpp),
    // which writes the result with writeBinary().
    U3xU4::MultiShellTensors_phys expand_new();

    // Load the expansion from the data directory; never expands.
    U3xU4::MultiShellTensors_phys expand();
    /* Note for expand():
    - The filename is <data dir>/OneBodyOperatorJT/<name>/Expansion/<name>_hw<hw>.
      If the file exists, it is read with readBinary().
    - If the file is missing, or reaches a smaller maxShell than this operator
      needs, expand() stops with an error that says how to get the file:
      download it (`make data`) or run the expansion program separately.
    - For operators whose matrix elements are computed on the fly:
      + the expansion is always done at hw_ = 20, thus filename will always
        end with hw20
      + when readBinary() is called, strength of tensors are scaled accordingly
    */

    // Write to binary file
    void writeBinary(const std::string& filename, 
                     const U3xU4::MultiShellTensors_phys& tensors) const;

    // Read from binary file; stops with an error if the file reaches a smaller
    // maxShell than this operator needs
    U3xU4::MultiShellTensors_phys readBinary(
      const std::string& filename);

    // Matrix representation
    // symmetric
    utils::CSRMatrix
    matrix_CSR(const U3xU4::ManyNucleonBasis& basis);

    Eigen::SparseMatrix<double>
    matrix_Eigen(const U3xU4::ManyNucleonBasis& basis);

    // rectangular
    Eigen::SparseMatrix<double>
    matrix_Eigen(const U3xU4::ManyNucleonBasis& bras, 
                 const U3xU4::ManyNucleonBasis& kets);

    // Tensor Analyzers
    std::vector<std::pair<u3::irrep, sam::strength_stats>>
      U3_tensor_analyzer_write();
    std::vector<std::pair<u3::irrep, sam::strength_stats>>
      U3_tensor_analyzer();

    std::vector<std::pair<u4::irrep, sam::strength_stats>>
      U4_tensor_analyzer_write();
    std::vector<std::pair<u4::irrep, sam::strength_stats>>
      U4_tensor_analyzer();

    std::vector<std::pair<int, sam::strength_stats>>
      LL_tensor_analyzer_write();
    std::vector<std::pair<int, sam::strength_stats>>
      LL_tensor_analyzer();

    std::vector<std::pair<int, sam::strength_stats>>
      SS_tensor_analyzer_write();
    std::vector<std::pair<int, sam::strength_stats>>
      SS_tensor_analyzer();
  };

} // namespace JT
