#include <chrono>
#include <fstream>
#include <iostream>

#include <U3basis.h>
#include <U4basis.h>

#include <U3xU4HWSgenerator.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

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

  int A = p.quanta(),        // total number of nucleons
      N_quanta = b.quanta(); // total number of quanta

  assert(N_quanta % A == 0);

  int n = N_quanta / A; // shell number

  u3::irrep x(n, 0, 0);
  std::vector<u3::gelfand> sps3 = u3::gelfandBasis(x);
  u4::irrep y(1, 0, 0, 0);
  std::vector<u4::gelfand> sps4 = u4::gelfandBasis(y);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  std::filesystem::path path = U3xU4::get_HWS_file(A, n, U3xU4::irrep(b,p), sps3, sps4);
  auto [SDarr, coeffs] = U3xU4::read_HWS_data(path);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Outer multiplicity is " << coeffs[0].cols() << std::endl;

  bool print;
  std::cout << "Want to print the results? (1 yes, 0 no) ";
  std::cin >> print;
  if (print == true)
  {
    std::cout << "Slater Determinants are: " << std::endl;

    for (std::size_t i = 0; i < SDarr.size(); i++)
    {
      //SDarr[i].print();
      std::cout << SDarr[i] << "--> " << coeffs[i] << std::endl;
    }
  }

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}