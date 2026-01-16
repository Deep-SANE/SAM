#include <cstdint>
#include <iostream>

#include <qcx_print.h>
#include <qcx_utils.h>
#include <u3lib.h>
#include <u4lib.h>
#include <U3xU4ManyNucleonBasis.h>

int main(int argc, char* argv[])
{
  using qcx_utils::print;

  qcx_utils::initialize(&argc, &argv);
  u3::initialize();
  u4::initialize();

  int Z = 3, N = 3, Nmax = 2, JJ = 2;

  U3xU4::hw_subspace 
    hw0(0, {}, {{1,1,0,0}, {3,2,1,0}, {3,3,0,0}}, {}, {0}),
    hw2(2, {}, {{1,1,0,0}, {3,2,1,0}, {3,3,0,0}}, {}, {0});

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ, {});
  basis.construct();

  auto irreps = basis.irreps();
  auto labels = basis.decomposition();

  for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
  {
    auto [gamma, omega] = irreps[irrep_id];
    auto offset = irreps.offset(irrep_id);
    // auto labels = basis.decomposition(irreps[irrep_id]);

    print("Irrep ID {:5} : ", irrep_id);
    print("gamma = {}", gamma);
    print(" ; omega = {}", omega);
    print(" ; kappamax LL zetamax SS TT offset = {}", labels[irrep_id]);
    print(" ; offset = {}\n", offset);
  }

  print("\nBasis dimension: {}\n", basis.dim());

  u4::finalize();
  u3::finalize();
  qcx_utils::finalize();
}
