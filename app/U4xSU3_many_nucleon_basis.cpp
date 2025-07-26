#include <cstdint>
#include <iostream>

#include <qcx_print.h>
#include <U4xSU3_many_nucleon_basis.h>

int main()
{
  using qcx_utils::print;

  int Z, N, Nmax, MMT, JJ;

  std::cout << "Enter number of protons: ";
  std::cin >> Z;
  std::cout << "Enter number of neutrons: ";
  std::cin >> N;
  std::cout << "Enter Nmax: ";
  std::cin >> Nmax;
  std::cout << "Enter JJ: ";
  std::cin >> JJ;

  int A = Z + N;
  MMT = abs(Z - N);
  int n0hw = Nmax % 2;

  U4xSU3::ManyNucleonBasis basis(A, MMT, Nmax, JJ);
  basis.construct();

  for (size_t irrep_id = 0; irrep_id < basis.n_irreps(); irrep_id++)
  {
    auto [dist, gamma, omega_u4, omega_su3, offset] = basis.irrep(irrep_id);

    print("Irrep ID {:5} : ", irrep_id);
    print("dist = {}", dist);
    print(" ; gamma = {}", gamma);
    print(" ; omega U(4) = {}", omega_u4);
    print(" ; omega SU(3) = {}", omega_su3);
    print(" ; offset = {}\n", offset);
  }

  print("\nBasis dimension: {}\n", basis.dim());
}
