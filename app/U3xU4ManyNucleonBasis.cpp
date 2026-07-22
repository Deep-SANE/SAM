#include <chrono>
#include <cstdint>
#include <iostream>

#include <qcx_print.h>
#include <qcx_utils.h>

#include <sam.h>

#include <U3xU4ManyNucleonBasis.h>

int main(int argc, char* argv[])
{
  using qcx_utils::print;

  sam::initialize(&argc, &argv);

  int Z, N, Nmax, JJ;

  std::cout << "Enter number of protons: ";
  std::cin >> Z;
  std::cout << "Enter number of neutrons: ";
  std::cin >> N;
  std::cout << "Enter Nmax: ";
  std::cin >> Nmax;
  std::cout << "Enter JJ: ";
  std::cin >> JJ;

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  basis.construct();

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();
/*
  auto irreps = basis.irreps();
  const auto& labels = basis.decomposition();

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
*/
  print("\nBasis dimension: {}\n", basis.dim());

  // Calculate the duration
    std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  sam::finalize();
}
