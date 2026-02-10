#include <chrono>
#include <fstream>
#include <iostream>

#include <U3basis.h>
#include <U4basis.h>

#include <sam.h>

#include <U3xU4HWSgenerator.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  std::cout << "This program expands the HWS of a non-vacuum irrep"
            << " into Slater Determinants" << std::endl;

  int n;
  std::cout << "Enter shell number (<= 20): ";
  std::cin >> n;

  u3::irrep x(n, 0, 0);
  auto sps3 = u3::gelfandBasis(x);
  u4::irrep y(1, 0, 0, 0);
  auto sps4 = u4::gelfandBasis(y);

  int n13, n23, n33;
  std::cout << "Enter U(3) irrep n13 >= n23 >= n33: ";
  std::cin >> n13 >> n23 >> n33;
  u3::irrep b(n13, n23, n33);

  int n14, n24, n34, n44;
  std::cout << "Enter U(4) irrep n14 >= n24 >= n34 >= n44: ";
  std::cin >> n14 >> n24 >> n34 >> n44;
  u4::irrep p(n14, n24, n34, n44);

  U3xU4::GammaElement G(n, b, p);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  std::filesystem::path path = U3xU4::get_HWS_file(G, sps3, sps4);
  auto [SDarr, coeffs] = U3xU4::read_HWS_data(path);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Outer multiplicity is " << coeffs.cols() << std::endl;

  bool print;
  std::cout << "Want to print the results? (1 yes, 0 no) ";
  std::cin >> print;
  if (print == true)
  {
    std::cout << "Slater Determinants are: " << std::endl;

    for (size_t i = 0; i < SDarr.size(); i++)
    {
      // SDarr[i].print();
      std::cout << SDarr[i] << "--> " << coeffs.row(i) << std::endl;
    }
  }

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
