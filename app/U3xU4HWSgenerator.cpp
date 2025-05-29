#include <iostream>

#include <U3basis.h>
#include <U4basis.h>

#include <U3xU4HWSgenerator.h>

int main()
{
  int n13, n23, n33;
  std::cout << "Enter U(3) irrep: ";
  std::cin >> n13 >> n23 >> n33;
  u3::irrep b(n13, n23, n33);
  if (!b.check())
    throw std::out_of_range("Invalid irrep");

  int n14, n24, n34, n44;
  std::cout << "Enter U(4) irrep: ";
  std::cin >> n14 >> n24 >> n34 >> n44;
  u4::irrep p(n14, n24, n34, n44);
  if (!p.check())
    throw std::out_of_range("Invalid irrep");

  auto [SDarr, coeffs] = U3xU4::HWS_to_SD(b, p);

  std::cout << "Slater Determinants are: " << std::endl;

  for (std::size_t i = 0; i < SDarr.size(); i++)
  {
    SDarr[i].print();
    std::cout << "--> " << coeffs.row(i) << std::endl;
  }

  return 0;
}