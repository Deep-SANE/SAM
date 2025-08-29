#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <Databases.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4irrep.h>
#include <U3xU4ManyNucleonBasis.h>
#include <U3xU4SingleShellTensors.h>

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
  public:
    U3xU4::irrep a, // first tensor
        b,          // second tensor
        t;          // coupled tensor

    int16_t kappamax, // multiplicity of LL
        LL,           // twice orbital angular momentum
        SS,           // twice spin
        TT, MMT,      // twice isospin and projection
        JJ, MMJ;      // twice total angular momentum and projection

    // Store coefficients in the expansion of one-body JT-coupled operator.
    // The size of this array is equal to kappamax.
    std::vector<double> coeffs;

    // Constructor
    OneBodyMultiShellTensor(const U3xU4::irrep& a_, const U3xU4::irrep& b_,
                            const U3xU4::irrep& t_, int kappa_, int LL_,
                            int SS_, int TT_, int MMT_, int JJ_, int MMJ_,
                            const std::vector<double>& coeffs_)
        : a(a_),
          b(b_),
          t(t_),
          kappamax(kappa_),
          LL(LL_),
          SS(SS_),
          TT(TT_),
          MMT(MMT_),
          JJ(JJ_),
          MMJ(MMJ_),
          coeffs(coeffs_)
    { }

    // Get specific coefficient
    double coeff(int k) const { return coeffs[k]; }

    // Print
    friend std::ostream& operator<<(std::ostream& s,
                                    const OneBodyMultiShellTensor& x);
  };

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
    const HO::Table_nljmjmt& labels;

  public:
    // List of single-shell tensors, each corresponds to Gamma in the document.
    std::vector<U3xU4::SingleShellTensor> single_shell_tensors;

    // The coupled irreps, each corresponds to Omega in the document.
    // This list has one less element than the single_shell_tensors list.
    std::vector<U3xU4::irrep> coupled_irreps;

    // The last element of coupled_irreps will dictate the following:
    int16_t rhomax, // U(3) multiplicity
        etamax,     // U(4) multiplicity
        kappamax,   // multiplicity of LL
        LL,         // twice orbital angular momentum
        zetamax,    // multiplicity of (SS,TT)
        SS,         // twice spin
        TT, MMT,    // twice isospin and projection
        JJ, MMJ;    // twice total angular momentum and projection

    /* Store coefficients in the expansion of two-body JT-coupled operator.
       The size of this array is equal to rhomax * etamax * kappamax * zetamax.
       To access the elements:
       for (int r = 0; r < rhomax; r++)
         for (int e = 0; e < etamax; e++)
           for (int k = 0; k < kappa; k++)
             for (int z = 0; z < zeta; z++)
               index = ((r * etamax + e) * kappamax + k) * zetamax + z;
    */
    std::vector<double> coeffs;

    // Constructor
    TwoBodyMultiShellTensor(const HO::Table_nljmjmt& labels_)
        : labels(labels_), rhomax(0), etamax(0), kappamax(0), LL(0),
          zetamax(0), SS(0), TT(0), MMT(0), JJ(0), MMJ(0)
    {}
    TwoBodyMultiShellTensor(
        const std::vector<U3xU4::SingleShellTensor>& single_shell_tensors_,
        const std::vector<U3xU4::irrep>& coupled_irreps_, int rho_, int eta_,
        int kappa_, int LL_, int zeta_, int SS_, int TT_, int MMT_, int JJ_,
        int MMJ_, const std::vector<double>& coeffs_,
        const HO::Table_nljmjmt& labels_)
        : single_shell_tensors(single_shell_tensors_),
          coupled_irreps(coupled_irreps_),
          rhomax(rho_),
          etamax(eta_),
          kappamax(kappa_),
          LL(LL_),
          zetamax(zeta_),
          SS(SS_),
          TT(TT_),
          MMT(MMT_),
          JJ(JJ_),
          MMJ(MMJ_),
          coeffs(coeffs_),
          labels(labels_)
    {
      assert(single_shell_tensors_.size() > 0);
      assert(single_shell_tensors_.size() == coupled_irreps_.size() + 1);
    }

    // Get specific coefficient
    double coeff(int r, int e, int k, int z)
    {
      return coeffs[((r * etamax + e) * kappamax + k) * zetamax + z];
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
    expand(sam::Databases& dbs) const;
  };

  /* Class for holding a general U3xU4 multi-shell tensor.
     It has an array of single-shell tensors U3xU4::SingleShellTensor
     and an array of OmegaElement's (coupled irreps with their multiplicities).
     No tensor components are specified. This class primarily serves
     to compute multi-shell RMEs.
  */
  class MultiShellTensor
  {
  public:
    // List of single-shell tensors, each corresponds to Gamma in the document.
    std::vector<U3xU4::SingleShellTensor> Gamma;

    // The coupled (irrep, rhomax, etamax)'s, each corresponds to Omega in the document.
    // This list has one less element than the single_shell_tensors list.
    std::vector<U3xU4::OmegaElement> Omega;

    // Constructor
    MultiShellTensor(
      const std::vector<U3xU4::SingleShellTensor>& Gamma_,
      const std::vector<U3xU4::OmegaElement>& Omega_)
        : Gamma(Gamma_), Omega(Omega_)
    { }

    // Print
    friend std::ostream& operator<<(std::ostream& s,
                                    const MultiShellTensor& x);

    // This method assumes that everything is already unified, i.e., bra and ket
    // irreps are unified, and identities have been filled in already.
    std::vector<double> RME(const ManyNucleonBasis::Irrep& ket,
                            const ManyNucleonBasis::Irrep& bra,
                            sam::Databases& dbs) const;
  };

  /* Class for holding a general U3xU4 multi-shell tensor with physical labels.
     It has total spin and isospin quantum numbers, a MultiShellTensor object 
     and an array of subgroup labels, namely (kappamax,LL,zetamax,SS,TT). 
     In addition, it also stores an array of tensor strengths, which are 
     coefficients in the expansion from JT-coupled operator. The strengths are 
     stored in a 2D array: 
     - The 1st dimension has the same size with the array of subgroup labels
     - The 2nd dimension has varying size, depending on the multiplicities of
       the tensor, including all multiplicities of SingleShellTensors, 
       intershell couplings, kappamax and zetamax. All of these can be extracted
       easily using qcx_utils::NDIndex object. 
  */
  class MultiShellTensor_phys
  {
  public:
    int16_t TT, MMT,    // twice isospin and projection
            JJ, MMJ;    // twice total angular momentum and projection

    // A multi-shell tensor without subgroup labels
    U3xU4::MultiShellTensor T;

    // An array of subgroup labels
    std::vector<std::array<uint16_t,5>> LST;

    // Array to store strengths
    std::vector<std::vector<double>> strengths;

    // Constructor
    MultiShellTensor_phys(
      const U3xU4::MultiShellTensor& T_,
      const std::vector<std::array<uint16_t,5>>& LST_,
      const std::vector<std::vector<double>>& strengths_)
        : T(T_), LST(LST_), strengths(strengths_)
    { 
      assert(LST_.size() == strengths.size());
    }

    // Print
    friend std::ostream& operator<<(std::ostream& s,
                                    const MultiShellTensor_phys& x);
  };

  Eigen::SparseMatrix<double> MatrixElements(
    std::vector<MultiShellTensor_phys>& tensors,
    U3xU4::ManyNucleonBasis& basis, sam::Databases& dbs);

  // A function to couple single-shell tensors into multi-shell tensors
  using CouplingPath = std::pair<
    std::vector<U3xU4::SingleShellTensor>, // list of selected single-shell tensors
    std::vector<U3xU4::OmegaElement>>;     // list of all coupled (irrep,rhomax,etamax)

  void CoupleSingShellTensors(
      const std::vector<std::vector<U3xU4::SingleShellTensor>>&
          Tadat_lists,                          // list of lists of Tadat
      std::size_t depth,                        // recursion depth
      std::vector<U3xU4::SingleShellTensor> Tadat_choices,  // current choices
      std::vector<U3xU4::irrep> coupled_irreps, // current coupled irreps
      std::vector<int> rhomax_list,             // rhomax values
      std::vector<int> etamax_list,             // etamax values
      std::vector<CouplingPath>& all_results);  // final results

  /**
   * Unification of a multi-shell tensor with a basis irrep.
   *
   * @param T Multi-shell tensor to be unified.
   * @param irrep Basis irrep.
   *
   * @return Unified multi-shell tensor.
   */
  MultiShellTensor unify_tensor(const MultiShellTensor& T,
                                const ManyNucleonBasis::Irrep& irrep);
} // namespace U3xU4
