#include <cstdint>
#include <iostream>

#include <qcx_print.h>
#include <qcx_utils.h>
#include <U3xU4ManyNucleonBasis.h>

int main(int argc, char* argv[])
{
  using qcx_utils::print;

  qcx_utils::initialize(&argc, &argv);

  int Z, N, Nmax, MMT, JJ;

  std::cout << "Enter number of protons: ";
  std::cin >> Z;
  std::cout << "Enter number of neutrons: ";
  std::cin >> N;
  std::cout << "Enter Nmax: ";
  std::cin >> Nmax;
  std::cout << "Enter JJ: ";
  std::cin >> JJ;

  U3xU4::ManyNucleonBasis basis(Z, N, Nmax, JJ);
  basis.construct();

  auto irreps = basis.irreps();

  for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
  {
    auto [gamma, omega] = irreps[irrep_id];
    auto offset = irreps.offset(irrep_id);
    auto labels = basis.decomposition(irreps[irrep_id]);

    print("Irrep ID {:5} : ", irrep_id);
    print("gamma = {}", gamma);
    print(" ; omega = {}", omega);
    print(" ; kappamax LL zetamax SS TT offset = {}", labels);
    print(" ; offset = {}\n", offset);
  }

  print("\nBasis dimension: {}\n", basis.dim());

  qcx_utils::finalize();
}
