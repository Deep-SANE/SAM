#include <qcx_print.h>
#include <qcx_optional_reference.h>

#include <U3xU4irrep.h>
#include <U3xU4GammaOmega.h>

using qcx_utils::print;

int main()
{
  // basis vacuum gamma element in shell 0:
  U3xU4::GammaElement ge_basis_vacuum(0, 1, {}, {});
  print("{}\n", ge_basis_vacuum);

  // basis non-vacuum gamma element in shell 2:
  U3xU4::GammaElement ge_basis(2, 1, {4, 2, 0}, {3, 0, 0, 0});
  print("{}\n", ge_basis);

  // tensor non-vacuum element in shell 2:
  U3xU4::GammaElement ge_tensor(2, 1, {4, 2, 0}, 1, {1, 1, 1, 0});
  print("{}\n", ge_tensor);

  const U3xU4::irrep& ige = ge_tensor;

  print("{}\n", ige);

  // (basis/tensor) omega element:
  U3xU4::OmegaElement oe(1, {4, 2, 0}, 1, {3, 0, 0, 0});

  print("{}\n", oe);
  
  const U3xU4::irrep& ioe = oe;

  print("{}\n", ioe);

  qcx_utils::optional_reference<const U3xU4::irrep> rw1 = ge_tensor;

  print("{}\n", rw1.get());

  qcx_utils::optional_reference<const U3xU4::irrep> rw2;
  rw2 = oe;

  print("{}\n", rw2.get());

  auto rw3 = qcx_utils::cref<U3xU4::irrep>(ge_tensor);

  print("{}\n", rw3.get());
  print("{}\n", rw3->su3());
  print("{}\n", rw3->u3());
  print("{}\n", rw3->u4());
}
