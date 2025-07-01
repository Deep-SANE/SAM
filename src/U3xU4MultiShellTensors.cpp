#include <cassert>

#include <U3xU4MultiShellTensors.h>

namespace U3xU4
{
  OneBodyMultiShellTensor::OneBodyMultiShellTensor(
      int a3_, int b3_, const u3::irrep& t3_, const u4::irrep& t4_, int kappa_,
      int LL_, int SS_, int TT_, int MMT_, int JJ_, int MMJ_)
      : a3(a3_),
        b3(b3_),
        t3(t3_),
        t4(t4_),
        kappa(kappa_),
        LL(LL_),
        SS(SS_),
        TT(TT_),
        MMT(MMT_),
        JJ(JJ_),
        MMJ(MMJ_)
  { assert(std::abs(a3) <= std::abs(b3)); }

  void OneBodyMultiShellTensor::print() const
  {
    if (a3 > 0)
      std::cout << "[ ad(" << a3 - 1 << ",0)(1,0,0)";
    else
      std::cout << "[ at(" << -(a3 + 1) << ",0)(0,0,1)";

    if (b3 > 0)
      std::cout << " x ad(" << b3 - 1 << ",0)(1,0,0) ]";
    else
      std::cout << " x at(" << -(b3 + 1) << ",0)(0,0,1) ]";

    std::cout << "(" << t3.lam() << "," << t3.mu() << ")(" 
              << t4.P1() << "," << t4.P2() << "," << t4.P3() << ")" 
              << kappa << "," << LL << "," << SS << "," << TT << ","
              << MMT << "," << JJ << "," << MMJ << std::endl;
  }
} // namespace U3xU4
