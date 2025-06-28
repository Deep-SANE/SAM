#include <U3xU4MultiShellTensors.h>

namespace U3xU4
{
  OneBodyMultiShellTensor::OneBodyMultiShellTensor(
      int a3_, int b3_, const u3::irrep& t3_, const u4::irrep& t4_, int kappa_,
      int LL_, int SS_, int TT_, int MMT_, int JJ_, int MMJ_)
      : a3((a3_ > 0) ? a3_ + 1 : a3_ - 1),
        b3((b3_ > 0) ? b3_ + 1 : b3_ - 1),
        t3(t3_),
        t4(t4_),
        kappa(kappa_),
        LL(LL_),
        SS(SS_),
        TT(TT_),
        MMT(MMT_),
        JJ(JJ_),
        MMJ(MMJ_)
  { }

  void OneBodyMultiShellTensor::print() const
  {
    std::cout << "[ T";
    if (a3 > 0)
      std::cout << "[" << a3 - 1 << ",0,0][1,0,0,0]";
    else
      std::cout << "[" << -(a3 + 1) << "," << -(a3 + 1) << ",0][1,1,1,0]";

    std::cout << " x T";
    if (b3 > 0)
      std::cout << "[" << b3 - 1 << ",0,0][1,0,0,0] ]";
    else
      std::cout << "[" << -(b3 + 1) << "," << -(b3 + 1) << ",0][1,1,1,0] ]";

    std::cout << t3 << t4 << kappa << "," << LL << "," << SS << "," << TT << ","
              << MMT << "," << JJ << "," << MMJ << std::endl;
  }
} // namespace U3xU4
