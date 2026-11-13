#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <HOutilities.h>
#include <TensorAnalyzer.h>
#include <U3xU4ManyNucleonBasis.h>
#include <U3xU4MultiShellTensors.h>

namespace JT
{
  class TwoBodyOperator
  {
  private:
    const HO::Table_nljmjmt& HO_labels_;
    const U3xU4::Table_gelfand& gelfand_labels_;

    HO::qn_t JJ_, MMJ_, TT_, MMT_, 
             parity_, // (0=even,1=odd)
             hw_, Nmax_, valenceShell_, maxShell_;
    // Note that nuclear potentials mix isospin, thus we assign TT = -1

    std::string name_;
    bool on_the_fly, symmetric_;

    // only read input file during expansion (expand_new*, expand_exist*)
    std::optional<HO::TwBME_scalar> TBME_scalar; 
    std::optional<HO::TwBME_general> TBME_general;

    // List of available operators
    const std::unordered_set<std::string> known_scalar_ops = {
      "L.L", "L.S", "CoMNumberOp", "Kinetic", "Rrel"
    };

    const std::unordered_set<std::string> known_potentials = {
      "N2LO_EMN500_bare", "N3LO_EMN500_bare", "N4LO_EMN500_bare",
      "N3LO_EM500_bare",  "N2LO_opt_bare",    "N2LO_sat_bare", 
      "N2LO_EMN500_srg1.0", "N3LO_EMN500_srg1.0", "N4LO_EMN500_srg1.0",
      "N3LO_EM500_srg1.0",  "N2LO_opt_srg1.0",    "N2LO_sat_srg1.0",
      "N2LO_EMN500_srg2.0", "N3LO_EMN500_srg2.0", "N4LO_EMN500_srg2.0",
      "N3LO_EM500_srg2.0",  "N2LO_opt_srg2.0",    "N2LO_sat_srg2.0",
      "N2LO_EMN500_srg4.0", "N3LO_EMN500_srg4.0", "N4LO_EMN500_srg4.0",
      "N3LO_EM500_srg4.0",  "N2LO_opt_srg4.0",    "N2LO_sat_srg4.0"
    };

    const std::unordered_set<std::string> known_general_ops = {
      "GamowTellerMinus_2B_N3LO_EM500_bare"
    };

  public:
    // Constructor
    TwoBodyOperator(const std::string& name, int hw, int Nmax, 
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
                                    const TwoBodyOperator& x);

    // Name of the directory that contains data of this operator
    std::string dirname() const;

  private:
    // ME: / na lla jja mmja mmta | O | nc llc jjc mmjc mmtc \
    // 	   \ nb llb jjb mmjb mmtb |	  | nd lld jjb mmjd mmtd /
    double sphericalME(int na, int lla, int jja, int mmja, int mmta, int nb,
                       int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                       int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                       int mmjd, int mmtd) const;
    // Antisymmetrized matrix element
    double sphericalME_AS(int na, int lla, int jja, int mmja, int mmta, int nb,
                          int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                          int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                          int mmjd, int mmtd) const;

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
    std::tuple<std::vector<std::array<int, 4>>,
               std::vector<std::vector<std::vector<int>>>,
               std::vector<std::vector<int>>>
    GenerateAllCombinationsHOShells(int Nmax, int valenceShell,
                  int maxShell, int Nmax_old, int maxShell_old) const;
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
                           U3xU4::SSTDecomp_phys_db& db) const;

    // Set strengths for U3xU4::MultiShellTensor_phys
    void setTensorStrength(U3xU4::MultiShellTensor_phys& tensor,
                           const std::vector<std::vector<int>>& inverse,
                           const std::vector<int>& phase,
                           U3xU4::SSTDecomp_phys_db& db) const;

    // selection rules
    bool HOShellselection(int na, int nb, int nc, int nd) const;
    bool HOShellselection(int na, int nb, int nc, int nd,
                          int Nmax_old, int maxShell_old) const;
    bool LLselection(const int& LL) const;
    bool SSselection(const int& SS) const;
    bool U3selection(const u3::irrep& irr) const;
    bool U4selection(const u4::irrep& irr) const;
    // Combined kinematic selection for a candidate tensor, shared by all four
    // expand paths: U3/U4 selection and shift-quanta sign, plus four exact,
    // strength-independent selection rules that remove ~96% of the
    // "accepted-then-empty" tensors with zero false negatives. Two-tier gating:
    //   - antisymmetrization rules (per-pair Pauli; SU(4)-singlet => lam,mu even)
    //     apply to any JJ==0 nuclear interaction, scalar potentials AND NO2B;
    //   - two-body-structure rules (lam-mu == shift_quanta; creation/annihilation
    //     pair S+T-parity match) apply to recognized scalar potentials only, as
    //     normal-ordered three-body content breaks them.
    bool passes_selection(const U3xU4::MultiShellTensor_phys& tensor) const;

    // helper to scale strength of tensors for Kinetic and Rrel
    void scaleStrengths(U3xU4::MultiShellTensors_phys& tensors) const;

  public:
    // Expand into a sum of U3xU4::TwoBodyMultiShellTensors
    U3xU4::TwoBodyMultiShellTensors expand_old();

    // Expand into a sum of U3xU4::MultiShellTensor_phys with OpenMP
    U3xU4::MultiShellTensors_phys expand_new();
    void expand_exist(int Nmax_old, int maxShell_old,
      U3xU4::MultiShellTensors_phys& tensors);

    // Expand into a sum of U3xU4::MultiShellTensor_phys with hybrid MPI + OpenMP
    // and write the result to filename (each rank writes its part, rank 0
    // merges them). expand_exist_MPI extends an existing file, written up to
    // Nmax_old / maxShell_old, to this operator's maxShell_ by appending only
    // the new tensors. Both are the costly expansion step, called only from
    // the expansion program (app/tutorial/TwoBodyOperatorJT.cpp); both stop at once if
    // filename cannot be written (e.g. a read-only file from `make data`).
    U3xU4::MultiShellTensors_phys expand_new_MPI(const std::string& filename);
    void expand_exist_MPI(const std::string& filename,
                                                  int Nmax_old, int maxShell_old);

    // Load the expansion from the data directory; never expands.
    U3xU4::MultiShellTensors_phys expand();
    /* Note for expand():
    - The filename is <data dir>/TwoBodyOperatorJT/<name>/Expansion/<name>_hw<hw>.
      If the file exists, it is read with readBinary() or readBinary_MPI().
    - If the file is missing, or reaches a smaller maxShell than this operator
      needs, expand() stops with an error that says how to get the file:
      download it (`make data`) or run the expansion program separately.
      Expanding is very costly, so it is never started from here.
    - For operators whose matrix elements are computed on the fly:
      + the expansion is always done at hw_ = 20, thus filename will always
        end with hw20
      + when readBinary() or readBinary_MPI() is called, strength of tensors are
        scaled to the requested hw
    */

    // Write to binary file
    void writeBinary(const std::string& filename, 
                     const U3xU4::MultiShellTensors_phys& tensors) const;

    // Read from binary file; stops with an error if the file reaches a smaller
    // maxShell than this operator needs
    U3xU4::MultiShellTensors_phys readBinary(const std::string& filename);
    U3xU4::MultiShellTensors_phys readBinary_MPI(const std::string& filename);

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
