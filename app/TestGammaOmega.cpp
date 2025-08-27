#include <qcx_print.h>
#include <qcx_reference_wrapper.h>

#include <U3xU4irrep.h>
#include <U3xU4GammaOmega.h>

using qcx_utils::print;

int main()
{
  U3xU4::GammaElement g_vacuum(0);

  print("{}\n", g_vacuum);

  U3xU4::GammaElement ge(2, 1, {4, 2, 0}, {3, 0, 0, 0});

  print("{}\n", ge);

  const U3xU4::irrep& ige = ge;

  print("{}\n", ige);

  U3xU4::OmegaElement oe(1, {4, 2, 0}, 1, {3, 0, 0, 0});

  print("{}\n", oe);
  
  const U3xU4::irrep& ioe = oe;

  print("{}\n", ioe);

  qcx_utils::reference_wrapper<const U3xU4::irrep> rw1 = ge;

  print("{}\n", rw1.get());

  qcx_utils::reference_wrapper<const U3xU4::irrep> rw2;
  rw2 = oe;

  print("{}\n", rw2.get());

  auto rw3 = qcx_utils::cref<U3xU4::irrep>(ge);

  print("{}\n", rw3.get());
  print("{}\n", rw3->su3());
  print("{}\n", rw3->u3());
  print("{}\n", rw3->u4());
}
