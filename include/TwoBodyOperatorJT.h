#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

#include <U3basis.h>
#include <U3SO3xU3SO3.h>
#include <U3U2xU3U2.h>

#include <U4basis.h>
#include <U4SU2SU2xU4SU2SU2.h>
#include <U4U3xU4U3.h>

#include <HOutilities.h>
#include <U3xU4MultiShellTensors.h>

namespace JT
{
  class TwoBodyOperator
  {
  private:
    int8_t JJ, MMJ, // twice angular momentum and projection
        TT, MMT;    // twice isospin and projection

    bool parity; // parity, true = +1, false = -1

    std::string name;
    int8_t hw, Nmax, valenceShell, maxShell;
    bool on_the_fly = true;

    std::optional<HO::TBME_scalar> TBME; // only read input file during
                                         // expand() or writeBinary()
    const HO::Table_nljmjmt& labels;

  public:
    // Constructor
    TwoBodyOperator(const std::string& name_, const HO::Table_nljmjmt& labels_,
                    int8_t hw_, int8_t Nmax_, int8_t valenceShell_,
                    int8_t maxShell_);

    void is() const;
    void GetJJ() const
    {
      std::cout << "2J=" << JJ << ", 2MJ=" << MMJ << std::endl;
    }
    void GetTT() const
    {
      std::cout << "2T=" << TT << ", 2MT=" << MMT << std::endl;
    }
    void GetParity() const
    {
      std::cout << "P=" << ((parity == true) ? 1 : -1) << std::endl;
    }

    // ME: / na lla jja mmja mmta | O | nc llc jjc mmjc mmtc \
    // 	   \ nb llb jjb mmjb mmtb |	  | nd lld jjb mmjd mmtd /
    double sphericalME(int na, int lla, int jja, int mmja, int mmta, int nb,
                       int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                       int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                       int mmjd, int mmtd, qcx_utils::SO3_CGC& cgcs,
                       qcx_utils::SO3_Racah& racah,
                       qcx_utils::SO3_Racah9& racah9) const;
    // Antisymmetrized matrix element
    double sphericalME_AS(int na, int lla, int jja, int mmja, int mmta, int nb,
                          int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                          int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                          int mmjd, int mmtd, qcx_utils::SO3_CGC& cgcs,
                          qcx_utils::SO3_Racah& racah,
                          qcx_utils::SO3_Racah9& racah9) const;

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

    // Set coefficients for two-body multi-shell tensor.
    // It returns a flag, which indicate whether this tensor can be ignored.
    bool setTensorStrength(U3xU4::TwoBodyMultiShellTensor& tensor,
                           const std::vector<std::vector<int>>& inverse,
                           const std::vector<int>& phase,
                           qcx_utils::SO3_CGC& cgcs,
                           qcx_utils::SO3_Racah& racah,
                           qcx_utils::SO3_Racah9& racah9,
                           u3::WigU3SO3_db& wigU3SO3db,
                           u4::WigU4SU2SU2_db& wigU4SU2SU2db) const;

    // Expand into a sum of multi-shell two-body tensors
    std::vector<U3xU4::TwoBodyMultiShellTensor>
    expand(qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racah,
           qcx_utils::SO3_Racah9& racah9, u3::WigU3SO3_db& wigU3SO3db,
           u4::WigU4SU2SU2_db& wigU4SU2SU2db);

    // Write to binary file
    void writeBinary(qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racah,
      qcx_utils::SO3_Racah9& racah9, u3::WigU3SO3_db& wigU3SO3db,
      u4::WigU4SU2SU2_db& wigU4SU2SU2db);

    // Read from binary file
    std::vector<U3xU4::TwoBodyMultiShellTensor> readBinary();

  private:
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
  };

} // namespace JT
