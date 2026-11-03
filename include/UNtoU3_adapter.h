#ifndef SAM_UNTOU3_ADAPTER_H
#define SAM_UNTOU3_ADAPTER_H

// Adapter that lets the standalone untou3 library (Deep-SANE/untou3, main
// branch) operate directly on u3lib/u4lib irrep types. untou3 reaches irreps
// only through untou3::IrrepTraits, so specializing it for u3::irrep and
// u4::irrep keeps untou3 independent of our libraries while usable inside SAM.
// See the untou3 README, "Using other irrep types".

#include <array>

#include <U3basis.h>
#include <U4basis.h>
#include <UNtoU3.h>

template <>
struct untou3::IrrepTraits<u3::irrep>
{
  static std::array<int, 3> labels(const u3::irrep& i)
  {
    return {i.n13(), i.n23(), i.n33()};
  }

  static u3::irrep make(int e1, int e2, int e3) { return {e1, e2, e3}; }
};

template <>
struct untou3::IrrepTraits<u4::irrep>
{
  static std::array<int, 4> labels(const u4::irrep& i)
  {
    return {i.n14(), i.n24(), i.n34(), i.n44()};
  }
};

namespace U3xU4
{
  // U(N) -> U(3) reductions specialized to the u3lib/u4lib irrep types.
  template <int L>
  using UNtoU3 = untou3::UNtoU3<L, u3::irrep, u4::irrep>;
}

#endif // SAM_UNTOU3_ADAPTER_H
