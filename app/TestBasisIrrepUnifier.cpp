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
  size_t bra_irrep_id, ket_irrep_id;
  std::cout << "Enter bra irrep id: ";
  std::cin >> bra_irrep_id;
  std::cout << "Enter ket irrep id: ";
  std::cin >> ket_irrep_id;

  U3xU4::ManyNucleonBasis bra(Z, N, Nmax, JJ);
  bra.construct();

  U3xU4::ManyNucleonBasis ket(Z, N, Nmax, JJ);
  ket.construct();

  auto bra_irreps = bra.irreps();
  auto ket_irreps = ket.irreps();

  auto bra_irrep = bra_irreps[bra_irrep_id];
  auto ket_irrep = ket_irreps[ket_irrep_id];

  print("Bra irrep: gamma = {}; omega = {}\n", bra_irrep.gamma,
        bra_irrep.omega);
  print("Ket irrep: gamma = {}; omega = {}\n", ket_irrep.gamma,
        ket_irrep.omega);

  U3xU4::IrrepUnifier iu;
  const auto& [bra_uirrep, ket_uirrep] = iu.unify(bra_irrep, ket_irrep);

  print("Bra unified irrep: gamma = {}; omega = {}\n", bra_uirrep.gamma,
        bra_uirrep.omega);

  print("Ket unified irrep: gamma = {}; omega = {}\n", ket_uirrep.gamma,
        ket_uirrep.omega);

  qcx_utils::finalize();
}
