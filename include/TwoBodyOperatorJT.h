#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <Databases.h>
#include <HOutilities.h>
#include <TensorAnalyzer.h>
#include <U3xU4ManyNucleonBasis.h>
#include <U3xU4MultiShellTensors.h>

namespace JT
{
  class TwoBodyOperator
  {
  private:
    const HO::Table_nljmjmt& labels_;

    HO::qn_t JJ_, MMJ_, TT_, MMT_, parity_,
             hw_, Nmax_, valenceShell_, maxShell_;
    // Note that nuclear potentials mix isospin, thus we assign TT = -1

    std::string name_;

    bool on_the_fly;

    std::optional<HO::TwBME_scalar> TBME; // only read input file during
                                         // expand() or writeBinary()

    const std::unordered_set<std::string> known_scalar_ops = {
                        "L.L", "L.S", "CoMNumberOp", "Kinetic", "R.R"};

    const std::unordered_set<std::string> known_potentials = {
        "N2LO_EMN500_bare", "N3LO_EMN500_bare", "N4LO_EMN500_bare",
        "N3LO_EM500_bare",  "N2LO_opt_bare",    "N2LO_sat_bare", 
        "N2LO_EMN500_srg1.0", "N3LO_EMN500_srg1.0", "N4LO_EMN500_srg1.0",
        "N3LO_EM500_srg1.0",  "N2LO_opt_srg1.0",    "N2LO_sat_srg1.0",
        "N2LO_EMN500_srg2.0", "N3LO_EMN500_srg2.0", "N4LO_EMN500_srg2.0",
        "N3LO_EM500_srg2.0",  "N2LO_opt_srg2.0",    "N2LO_sat_srg2.0",
        "N2LO_EMN500_srg4.0", "N3LO_EMN500_srg4.0", "N4LO_EMN500_srg4.0",
        "N3LO_EM500_srg4.0",  "N2LO_opt_srg4.0",    "N2LO_sat_srg4.0"};

  public:
    // Constructor
    TwoBodyOperator(const std::string& name, int hw, int Nmax, 
      int valenceShell, int maxShell, const HO::Table_nljmjmt& labels);

    // Getters
    int JJ() const { return static_cast<int>(JJ_); }
    int MMJ() const { return static_cast<int>(MMJ_); }
    int TT() const { return static_cast<int>(TT_); }
    int MMT() const { return static_cast<int>(MMT_); }
    int parity() const { return static_cast<int>(parity_); }

    // Print
    void is() const;
    friend std::ostream& operator<<(std::ostream& s,
                                    const TwoBodyOperator& x);

  private:
    // ME: / na lla jja mmja mmta | O | nc llc jjc mmjc mmtc \
    // 	   \ nb llb jjb mmjb mmtb |	  | nd lld jjb mmjd mmtd /
    double sphericalME(int na, int lla, int jja, int mmja, int mmta, int nb,
                       int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                       int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                       int mmjd, int mmtd, sam::Databases& dbs) const;
    // Antisymmetrized matrix element
    double sphericalME_AS(int na, int lla, int jja, int mmja, int mmta, int nb,
                          int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                          int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                          int mmjd, int mmtd, sam::Databases& dbs) const;

    // Generate a list of all {na, nb, nc, nd} combinations such that
    // 0 <= na, nb, nc, nd <= maxShell
    std::tuple<std::vector<std::array<int, 4>>,
               std::vector<std::vector<std::vector<int>>>,
               std::vector<std::vector<int>>>
    GenerateAllCombinationsHOShells(int maxShell) const;
    // Restrict:
    // 0 <= (na - valenceShell + (nb - valenceShell) <= Nmax
    // 0 <= (nc - valenceShell + (nd - valenceShell) <= Nmax
    std::tuple<std::vector<std::array<int, 4>>,
               std::vector<std::vector<std::vector<int>>>,
               std::vector<std::vector<int>>>
    GenerateAllCombinationsHOShells(int Nmax, int valenceShell,
                                    int maxShell) const;
    // Note that for both cases, permutation on {na, nb, nc, nd} is applied
    // to obtain a combination in the order of increasing shells. Different
    // unpermuted combinations may be permuted to the same one. These two
    // methods only return permuted combinations together with their phases
    // and arrays for the purpose of inverting the permutations.


    // Perform permutation of tensor in increasing shell.
    // For tensors of the same shell, ad's are in the front of at's.
    // Exchange of 2 tensors always incurs a phase of -1.0.
    // No exchange of ad and a of the same shell occurs.
    std::tuple<std::array<int, 4>, std::vector<int>, int>
    sort_shells(const std::array<int, 4>& original) const;

    // Restores the original ordering using inverse permutation
    std::array<int, 4>
    inverse_permutation(const std::array<int, 4>& sorted,
                        const std::vector<int>& inverse) const;

    // Group tensors of the same shell
    // Results are contained in a list of 3-element arrays:
    // first: shell number
    // second: number of ad's
    // thrid: number of at's
    std::vector<std::array<int, 3>>
    group_same_shell(const std::array<int, 4>& sorted) const;

    // Set strengths for U3xU4::TwoBodyMultiShellTensor
    // It returns a flag, which indicate whether this tensor can be ignored.
    bool setTensorStrength(U3xU4::TwoBodyMultiShellTensor& tensor,
                           const std::vector<std::vector<int>>& inverse,
                           const std::vector<int>& phase,
                           sam::Databases& dbs) const;

    // Set strengths for U3xU4::MultiShellTensor_phys
    void setTensorStrength(U3xU4::MultiShellTensor_phys& tensor,
                           const std::vector<std::vector<int>>& inverse,
                           const std::vector<int>& phase,
                           sam::Databases& dbs) const;

    // selection rule
    bool LLselection(const int& LL) const;
    bool SSselection(const int& SS) const;
    bool U3selection(const u3::irrep& irr) const;
    bool U4selection(const u4::irrep& irr) const;

  public:
    // Expand into a sum of U3xU4::TwoBodyMultiShellTensors
    U3xU4::TwoBodyMultiShellTensors expand_old(sam::Databases& dbs);

    // expand into a sum of U3xU4::MultiShellTensor_phys
    U3xU4::MultiShellTensors_phys expand_new(sam::Databases& dbs);
    U3xU4::MultiShellTensors_phys expand(sam::Databases& dbs);

    // Write to binary file
    void writeBinary(const std::string& filename, 
                     const U3xU4::MultiShellTensors_phys& tensors) const;

    // Read from binary file
    U3xU4::MultiShellTensors_phys readBinary(
      const std::string& filename, sam::Databases& dbs);

    // Matrix representation
    // symmetric
    Eigen::SparseMatrix<double>
    matrix(const U3xU4::ManyNucleonBasis& basis, sam::Databases& dbs);
    // rectangular
    Eigen::SparseMatrix<double>
    matrix(const U3xU4::ManyNucleonBasis& bras, 
           const U3xU4::ManyNucleonBasis& kets, sam::Databases& dbs);

    // Tensor Analyzers
    std::vector<std::pair<u3::irrep, sam::strength_stats>>
      U3_tensor_analyzer_write(sam::Databases& dbs);
    std::vector<std::pair<u3::irrep, sam::strength_stats>>
      U3_tensor_analyzer(sam::Databases& dbs);

    std::vector<std::pair<u4::irrep, sam::strength_stats>>
      U4_tensor_analyzer_write(sam::Databases& dbs);
    std::vector<std::pair<u4::irrep, sam::strength_stats>>
      U4_tensor_analyzer(sam::Databases& dbs);

    std::vector<std::pair<int, sam::strength_stats>>
      LL_tensor_analyzer_write(sam::Databases& dbs);
    std::vector<std::pair<int, sam::strength_stats>>
      LL_tensor_analyzer(sam::Databases& dbs);

    std::vector<std::pair<int, sam::strength_stats>>
      SS_tensor_analyzer_write(sam::Databases& dbs);
    std::vector<std::pair<int, sam::strength_stats>>
      SS_tensor_analyzer(sam::Databases& dbs);
  };

} // namespace JT
