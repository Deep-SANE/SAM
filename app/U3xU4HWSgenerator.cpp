#include <chrono>
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

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto [SDarr, coeffs] = U3xU4::HWS_to_SD(b, p);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();  

  std::cout << "Outer multiplicity is " << coeffs.cols() << std::endl;

/*
  std::cout << "Slater Determinants are: " << std::endl;

  for (std::size_t i = 0; i < SDarr.size(); i++)
  {
    SDarr[i].print();
    std::cout << "--> " << coeffs.row(i) << std::endl;
  }
*/

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
              << duration.count() / 60.0 << " minutes" << std::endl;
  return 0;
}