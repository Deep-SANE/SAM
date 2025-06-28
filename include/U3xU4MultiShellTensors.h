#pragma once

#include <U3basis.h>

#include <U4basis.h>

namespace U3xU4
{
  /* Class for holding one-body multi-shell tensors
                          (lam mu) (P P' P'')
          [Ta x Tb]
                          kappa 2L 2S 2T 2MT 2J 2MJ
          where Ta and Tb can be either ad(na,0)(1,0,0) or at(0,nb)(0,0,1)
          in such a way that the first tensor is always in a lower or equal
     shell.
  */
  class OneBodyMultiShellTensor
  {
  public:
    // storing scheme: similar to lsu3shell
    // +(n+1) means ad[n,0,0][1,0,0,0]
    // -(n+1) means at[n,n,0][1,1,1,0]
    // this means n will start from 1 instead of 0
    int a3, // first tensor
        b3; // second tensor

    u3::irrep t3; // coupled tensor
    u4::irrep t4; // coupled tensor

    int kappa,   // = 1,...,max(kappa)
        LL,      // twice orbital angular momentum
        SS,      // twice spin
        TT, MMT, // twice isospin and projection
        JJ, MMJ; // twice total angular momentum and projection

    // Constructor
    // To be user-friendly, the constructor will take the exact shells
    // but it will store one shell higher internally
    OneBodyMultiShellTensor(int a3_, int b3_, const u3::irrep& t3_,
                            const u4::irrep& t4_, int kappa_, int LL_, int SS_,
                            int TT_, int MMT_, int JJ_, int MMJ_);

    void print() const;

    // Comparison operator
    bool operator==(const OneBodyMultiShellTensor& other) const
    {
      return (a3 == other.a3) && (b3 == other.b3) && (t3 == other.t3) &&
             (t4 == other.t4) && (kappa == other.kappa) && (LL == other.LL) &&
             (SS == other.SS) && (TT == other.TT) && (MMT == other.MMT) &&
             (JJ == other.JJ) && (MMJ == other.MMJ);
    }

    // RME ... TBD
  };

} // namespace U3xU4