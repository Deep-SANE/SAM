#include <chrono>
#include <iomanip>
#include <iostream>

#include <HOutilities.h>

int main()
{
  int hw = -1, Nmax = 12, valenceShell = -1, maxShell = 12; // default

  HO::TBME_scalar TBME("N2LO_opt_bare", hw, Nmax, valenceShell, maxShell);
  TBME.is();
  std::cout << "Loaded " << TBME.countMEs() << " matrix elements" << std::endl;

  std::cout << std::fixed << std::setprecision(10);

  auto t0 = std::chrono::high_resolution_clock::now();

  std::cout << TBME.getME(11, 11, 7, 1, 0) << "\n"
            << TBME.getME(11, 11, 1, 7, 0) << "\n"
            << TBME.getME(7, 1, 11, 11, 0) << "\n"
            << TBME.getME(1, 7, 11, 11, 0) << "\n"
            << std::endl;

  std::cout << TBME.getME(43, 21, 21, 1, 0) << "\n"
            << TBME.getME(43, 21, 1, 21, 0) << "\n"
            << TBME.getME(21, 43, 21, 1, 0) << "\n"
            << TBME.getME(21, 43, 1, 21, 0) << "\n"
            << TBME.getME(21, 1, 43, 21, 0) << "\n"
            << TBME.getME(21, 1, 21, 43, 0) << "\n"
            << TBME.getME(1, 21, 43, 21, 0) << "\n"
            << TBME.getME(1, 21, 21, 43, 0) << "\n"
            << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = t1 - t0;
  std::cout << "Hashing time: " << duration.count() << " seconds" << std::endl;

  return 0;
}
