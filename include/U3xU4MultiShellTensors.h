#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <HOutilities.h>
#include <RME.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4irrep.h>
#include <U3xU4ManyNucleonBasis.h>
#include <U3xU4SingleShellTensors.h>

#include <qcx_Int.h>

namespace U3xU4
{
  /* Class for holding one-body multi-shell tensors
              (n13 n23 n33) (n14 n24 n34 n44)
      [Ta x Tb]
              kappa 2L 2S 2T 2MT 2J 2MJ
    where Ta and Tb can be either ad or at in such a way that
    the first tensor is always in a lower or equal shell.
    The reduction (n14 n24 n34 n44)->(S,T) is always multiplicity free.
  */
  class OneBodyMultiShellTensor
  {
  private:
    U3xU4::irrep a_, // first tensor
                 b_, // second tensor
                 t_; // coupled tensor

    uint8_t kappamax_;
    HO::qn_t LL_, SS_, TT_, MMT_, JJ_, MMJ_;

    // Store coefficients in the expansion of one-body JT-coupled operator.
    // The size of this array is equal to kappamax.
    std::vector<double> coeffs_;

  public:
    // Constructor
    OneBodyMultiShellTensor(
      const U3xU4::irrep& a, const U3xU4::irrep& b, const U3xU4::irrep& t, 
      qcx_utils::UInt8 kappamax, qcx_utils::Int8 LL, qcx_utils::Int8 SS, 
      qcx_utils::Int8 TT, qcx_utils::Int8 MMT, qcx_utils::Int8 JJ,
      qcx_utils::Int8 MMJ, const std::vector<double>& coeffs)
        : a_(a),
          b_(b),
          t_(t),
          kappamax_(kappamax),
          LL_(LL),
          SS_(SS),
          TT_(TT),
          MMT_(MMT),
          JJ_(JJ),
          MMJ_(MMJ),
          coeffs_(coeffs)
    { }

    // Getters
    const auto& a() const { return a_; }
    const auto& b() const { return b_; }
    const auto& t() const { return t_; }
    const auto& getLastIrrep() const { return t_; }
    int kappamax() const { return static_cast<int>(kappamax_); }
    int LL() const { return static_cast<int>(LL_); }
    int SS() const { return static_cast<int>(SS_); }
    int TT() const { return static_cast<int>(TT_); }
    int MMT() const { return static_cast<int>(MMT_); }
    int JJ() const { return static_cast<int>(JJ_); }
    int MMJ() const { return static_cast<int>(MMJ_); }
    const auto& coeffs() const { return coeffs_; }

    // Get specific coefficient
    double coeff(int k) const { return coeffs_[k]; }

    // Print
    friend std::ostream& operator<<(std::ostream& s,
                                    const OneBodyMultiShellTensor& x);

    // binary writing and reading
    void writeBinary(std::ostream& s) const;
    void readBinary(std::istream& s);
  };
  using OneBodyMultiShellTensors = std::vector<OneBodyMultiShellTensor>;

  /* Class for holding two-body multi-shell tensors
         rho(n13 n23 n33) eta(n14 n24 n34 n44)
        T
         kappa 2L zeta 2S 2T 2MT 2J 2MJ
    where T is a coupled tensor of single-shell tensors that are coupled
    in an increasing order of shell numbers.
    E.g. 1: no tensor in the same shell
            [[[T1 x T2] x T3] x T4]
    E.g. 2: two tensors T2 and T3 in the same shell
            [[T1 x [T2 x T3]] x T4]
    E.g. 3: three tensors T2, T3, T4 in the same shell
            [T1 x [[T2 x T3] x T4]]
    E.g. 4: four tensors in the same shell
            [[T1 x T2] x [T3 x T4]]
    Tensors of the same shell are always next to each other such that
    ad's are always on the left of at's.

    Note on the multiplicities:
    1. Couling with either ad or at is always free of multiplicities.
    2. Case 1: There is only one single-shell tensor of form [[ad x ad] x [at x at]],
               there is a multiplicity (rho,eta) for the final coupling. There is no
               intershell coupling irrep.
    3. For all other cases, single-shell tensors are free of multiplicity because
       none of them can reach two ad's and two at's.
    4. Case 2: two single-shell tensors (free of multiplicity), there is only one 
               intershell coupling, which may have multiplicity.
    5. Case 3: three single-shell tensors, there are three cases:
               1 tensor x 1 tensor x 2 tensors => multiplicity for the last intershell
                                                  coupling
               1 tensor x 2 tensors x 1 tensor => free of multiplicity
               2 tensors x 1 tensor x 1 tensor => free of multiplicity
    6. Case 4: four single-shell tensors, there is only one case:
               1 tensor x 1 tensor x 1 tensor x 1 tensor => free of multiplicity
  */
  class TwoBodyMultiShellTensor
  {
  private:
    const HO::Table_nljmjmt& labels_;

    // List of single-shell tensors, each corresponds to Gamma in the document.
    U3xU4::SingleShellTensors Gammas_;

    // The coupled irreps, each corresponds to Omega in the document.
    // This list has one less element than the single_shell_tensors list.
    U3xU4::irreps Omegas_;

    // The last element of coupled_irreps will dictate the following:
    uint8_t rhomax_, etamax_,  kappamax_, zetamax_; 
    HO::qn_t LL_, SS_, TT_, MMT_, JJ_, MMJ_;

    /* Store coefficients in the expansion of two-body JT-coupled operator.
       The size of this array is equal to rhomax * etamax * kappamax * zetamax.
       To access the elements:
       for (int r = 0; r < rhomax; r++)
         for (int e = 0; e < etamax; e++)
           for (int k = 0; k < kappa; k++)
             for (int z = 0; z < zeta; z++)
               index = ((r * etamax + e) * kappamax + k) * zetamax + z;
    */
    std::vector<double> coeffs_;

  public:
    // Constructor
    TwoBodyMultiShellTensor(const HO::Table_nljmjmt& labels)
        : labels_(labels), rhomax_(0), etamax_(0), kappamax_(0), 
          LL_(0), zetamax_(0), SS_(0), TT_(0), MMT_(0), JJ_(0), MMJ_(0)
    {}
    TwoBodyMultiShellTensor(
      const U3xU4::SingleShellTensors& Gammas, const U3xU4::irreps& Omegas, 
      qcx_utils::UInt8 rhomax, qcx_utils::UInt8 etamax,
      qcx_utils::UInt8 kappamax, qcx_utils::Int8 LL, 
      qcx_utils::UInt8 zetamax, qcx_utils::Int8 SS, qcx_utils::Int8 TT, 
      qcx_utils::Int8 MMT, qcx_utils::Int8 JJ, qcx_utils::Int8 MMJ, 
      const std::vector<double>& coeffs, const HO::Table_nljmjmt& labels)
        : Gammas_(Gammas),
          Omegas_(Omegas),
          rhomax_(rhomax),
          etamax_(etamax),
          kappamax_(kappamax),
          LL_(LL),
          zetamax_(zetamax),
          SS_(SS),
          TT_(TT),
          MMT_(MMT),
          JJ_(JJ),
          MMJ_(MMJ),
          coeffs_(coeffs),
          labels_(labels)
    {
      assert(Gammas_.size() > 0 && Gammas_.size() == Omegas_.size() + 1);
    }

    // Getters
    const auto& Gammas() const { return Gammas_; }
    const auto& Omegas() const { return Omegas_; }
    const U3xU4::irrep& getLastIrrep() const 
    { 
      if (Omegas_.empty())
        return Gammas_.back().u3u4();
      else
        return Omegas_.back();
    }
    int rhomax() const { return static_cast<int>(rhomax_); }
    int etamax() const { return static_cast<int>(etamax_); }
    int kappamax() const { return static_cast<int>(kappamax_); }
    int zetamax() const { return static_cast<int>(zetamax_); }
    int LL() const { return static_cast<int>(LL_); }
    int SS() const { return static_cast<int>(SS_); }
    int TT() const { return static_cast<int>(TT_); }
    int MMT() const { return static_cast<int>(MMT_); }
    int JJ() const { return static_cast<int>(JJ_); }
    int MMJ() const { return static_cast<int>(MMJ_); }
    const auto& coeffs() const { return coeffs_; }

    // Setter
    void setCoeffs(const std::vector<double>& coeffs)
    {
      coeffs_ = std::move(coeffs);
    }

    // Get specific coefficient
    double coeff(int r, int e, int k, int z)
    {
      return coeffs_[((r * etamax() + e) * kappamax() + k) * zetamax() + z];
    }

    // Print
    friend std::ostream& operator<<(std::ostream& s,
                                    const TwoBodyMultiShellTensor& x);

    // Write to binary file
    void writeBinary(std::ostream& s) const;

    // Read from binary file
    void readBinary(std::istream& s);

    // Expand into uncoupled products of creation operators (ad)
    // and annihilation operators (a)
    // coeffs have rhomax x etamax x kappamax x zetamax columns, index of each
    // column is
    // ((rho * etamax + eta) * kappamax + kappa) * zetamax + zeta
    std::pair<std::vector<Eigen::RowVectorXd>,  // coeffs
              std::vector<U3xU4::ads_ats_phys>> // list of ad's and a's
    expand() const;
  };
  using TwoBodyMultiShellTensors = std::vector<TwoBodyMultiShellTensor>;

  /* Class for holding a general U3xU4 multi-shell tensor.
     It has an array of single-shell tensors U3xU4::SingleShellTensor
     and an array of OmegaElement's (coupled irreps with their multiplicities).
     No tensor components are specified. This class primarily serves
     to compute multi-shell RMEs.
  */
  class MultiShellTensor
  {
  private:
    // List of single-shell tensors, each corresponds to Gamma in the document.
    U3xU4::SingleShellTensors Gammas_;

    // The coupled (irrep, rhomax, etamax)'s, each corresponds to Omega in the document.
    // This list has one less element than the single_shell_tensors list.
    U3xU4::OmegaElements Omegas_;

  public:
    // Constructor
    MultiShellTensor() = default;
    MultiShellTensor(const U3xU4::SingleShellTensors& Gammas,
                     const U3xU4::OmegaElements& Omegas)
        : Gammas_(Gammas), Omegas_(Omegas)
    { 
      assert(Gammas.size() == Omegas.size() + 1);
    }

    // Getters
    const auto& Gammas() const { return Gammas_; }
    const auto& Omegas() const { return Omegas_; }
    const U3xU4::irrep& getLastIrrep() const
    {
      if (Omegas_.empty())
        return Gammas_.back().u3u4();
      else
        return Omegas_.back();
    }

    // Get change of oscillator quanta
    int shift_quanta() const
    {
      int res = 0;
      for (const auto& G : Gammas_) res += G.shift_quanta();
      return res;
    }

    // comparison:
    bool operator==(const MultiShellTensor&) const = default;

    // Print
    friend std::ostream& operator<<(std::ostream& s,
                                    const MultiShellTensor& x);

    // Write to binary file
    void writeBinary(std::ostream& s) const;

    // Read from binary file
    void readBinary(std::istream& s);

    // This method assumes that everything is already unified, i.e., bra and ket
    // irreps are unified, and identities have been filled in already.
    bool RME_exist(const ManyNucleonBasis::Irrep& bra,
                   const ManyNucleonBasis::Irrep& ket) const;

    sam::RME RME(const ManyNucleonBasis::Irrep& bra,
                 const ManyNucleonBasis::Irrep& ket) const;

    sam::RME RME(const ManyNucleonBasis::Irrep& bra,
                 const ManyNucleonBasis::Irrep& ket,
                 const U3xU4::Table_gelfand& labels) const;

  private:
    // Phase during uncoupling of RME
    int8_t uncoupling_phase(const ManyNucleonBasis::Irrep& bra,
                            const ManyNucleonBasis::Irrep& ket) const;
  };
  using MultiShellTensors = std::vector<MultiShellTensor>;

  // A function to couple single-shell tensors into multi-shell tensors
  using CouplingPath = 
    std::pair<U3xU4::SingleShellTensors, // list of selected single-shell tensors
              U3xU4::OmegaElements>;     // list of all coupled (irrep,rhomax,etamax)

  void GenerateMultiShellTensors(
      const std::vector<U3xU4::SingleShellTensors>&
                                        Tadat_lists,// list of lists of Tadat
      size_t depth,                                 // recursion depth
      U3xU4::SingleShellTensors Tadat_choices,      // current choices
      U3xU4::irreps coupled_irreps,                 // current coupled irreps
      std::vector<int> rhomax_list,                 // rhomax values
      std::vector<int> etamax_list,                 // etamax values
      std::vector<CouplingPath>& all_results);      // final results

  /* Class for holding a general U3xU4 multi-shell tensor with physical labels.
     It has total spin (JJ,MMJ) and isospin (TT,MMT) quantum numbers, 
     a MultiShellTensor object and an array of subgroup labels, namely 
     (kappamax,LL,zetamax,SS). In addition, it also stores an array of tensor 
     strengths, which are coefficients in the expansion from JT-coupled operator. 
     The strengths are stored in a 2D array: 
     - The 1st dimension has the same size with the array of subgroup labels
     - The 2nd dimension has varying size, depending on the multiplicities of
       the tensor, including all multiplicities of SingleShellTensors, 
       intershell couplings, kappamax and zetamax. All of these can be extracted
       easily using qcx_utils::NDIndex object. 
  */
  class MultiShellTensor_phys : public U3xU4::MultiShellTensor
  {
  private:
    const HO::Table_nljmjmt& labels_;

    HO::qn_t MMT_, JJ_, MMJ_;

    using labels_type = std::tuple<uint8_t,              // kappamax
                                   HO::qn_t,             // LL
                                   uint8_t,              // zetamax
                                   HO::qn_t,             // SS
                                   HO::qn_t,             // TT
                                   std::vector<double>>; // strengths

    std::vector<labels_type> LLSSTTstrengths_;

  public:
    // Constructor
    MultiShellTensor_phys(const HO::Table_nljmjmt& labels)
      : labels_(labels), MMT_(0), JJ_(0), MMJ_(0)
    {}
    MultiShellTensor_phys(
      const U3xU4::SingleShellTensors& Gammas,
      const U3xU4::OmegaElements& Omegas,
      qcx_utils::Int8 MMT, qcx_utils::Int8 JJ, qcx_utils::Int8 MMJ,
      const HO::Table_nljmjmt& labels)
        : U3xU4::MultiShellTensor(Gammas, Omegas),
          MMT_(MMT), JJ_(JJ), MMJ_(MMJ), labels_(labels)
    { }

    // Getters
    const U3xU4::MultiShellTensor& T() const { return *this; }
    int MMT() const { return static_cast<int>(MMT_); }
    int JJ() const { return static_cast<int>(JJ_); }
    int MMJ() const { return static_cast<int>(MMJ_); }
    const auto& LLSSTTstrengths() const { return LLSSTTstrengths_; }

    // comparison:
    bool operator==(const MultiShellTensor_phys& x) const
    {
      return T() == x.T() && MMT_ == x.MMT_ && JJ_ == x.JJ_ && MMJ_ == x.MMJ_;
    }

    // Calculate multiplicities
    size_t alphamax() const
    {
      size_t res = 1;
      for (const auto& G : Gammas()) res *= G.rho0max() * G.eta0max();
      for (const auto& O : Omegas()) res *= O.rhomax() * O.etamax();
      return res;
    }

    // Print
    friend std::ostream& operator<<(std::ostream& s,
                                    const MultiShellTensor_phys& x);

    // Write to binary file
    void writeBinary(std::ostream& s) const;

    // Read from binary file
    void readBinary(std::istream& s);

    // Expand into uncoupled products of creation operators (ad)
    // and annihilation operators (a)
    std::pair<std::vector<Eigen::RowVectorXd>,  // coeffs
              std::vector<U3xU4::ads_ats_phys>> // list of ad's and a's
              expand(int LL, int SS, int TT) const;

    // Setter: done after expanding and computing strengths
    void setLLSSTTstrengths(const std::vector<labels_type>& LLSSTTstrengths)
    {
      LLSSTTstrengths_ = std::move(LLSSTTstrengths);
    }

    // Add to existing strengths
    void addLLSSTTstrengths(const std::vector<labels_type>& LLSSTTstrengths)
    {
      for (const auto& [k, LL, z, SS, TT, strengths] : LLSSTTstrengths)
      {
        for (auto& [k_, LL_, z_, SS_, TT_, strengths_] : LLSSTTstrengths_)
        {
          if (k == k_ && LL == LL_ && z == z_ && SS == SS_ && TT == TT_)
          {
            assert(strengths.size() == strengths_.size());
            for (size_t i = 0; i < strengths.size(); i++)
              strengths_[i] += strengths[i];
            break;
          }
          else
          {
            LLSSTTstrengths_.push_back({k, LL, z, SS, TT, strengths});
          }
        }
      }
    }

    // Multiply the strengths with a number
    void enhanceStrengths(double x)
    {
      for (auto& [k, LL, z, SS, TT, strengths] : LLSSTTstrengths_)
        for (auto& s : strengths) s *= x;
    }
  };
  using MultiShellTensors_phys = std::vector<MultiShellTensor_phys>;
  
  /**
   * Unification of a multi-shell tensor with a basis irrep.
   *
   * @param T Multi-shell tensor to be unified.
   * @param irrep Basis irrep.
   *
   * @return Unified multi-shell tensor, or, an empty optional if unification
   * is not possible.
   */
  std::optional<MultiShellTensor>
  unify_tensor(const MultiShellTensor& T, const ManyNucleonBasis::Irrep& irrep);

  /////////////////// COMPUTING MATRIX ELEMENTS ///////////////////

  // Symmetric matrix
  Eigen::SparseMatrix<double> 
    MatrixElements(const MultiShellTensors_phys& tensors, 
                   const U3xU4::ManyNucleonBasis& basis);

  Eigen::SparseMatrix<double> 
    MatrixElements(const MultiShellTensors_phys& tensors, 
                   const U3xU4::ManyNucleonBasis& basis,
                   const U3xU4::Table_gelfand& gelfand_labels);

  // Rectangular matrix
  Eigen::SparseMatrix<double> 
    MatrixElements(const MultiShellTensors_phys& tensors, 
                   const U3xU4::ManyNucleonBasis& bras,
                   const U3xU4::ManyNucleonBasis& kets);

  Eigen::SparseMatrix<double> 
    MatrixElements(const MultiShellTensors_phys& tensors, 
                   const U3xU4::ManyNucleonBasis& bras,
                   const U3xU4::ManyNucleonBasis& kets,
                   const U3xU4::Table_gelfand& gelfand_labels);

} // namespace U3xU4
