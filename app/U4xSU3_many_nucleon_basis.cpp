#include <cstdint>
#include <iostream>

#include <sam_config.h>
#include <U4xSU3_many_nucleon_basis.h>

int main()
{
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
  bool n0hw = Nmax % 2;

  U4xSU3::ManyNucleonBasis basis;

  auto distributions = basis.generate_distributions(Nmax, A);

  uint64_t dim = 0;

  for (int nhw = n0hw; nhw <= Nmax; nhw += 2)
  {
    print("hw: {}\n", nhw);

    for (const auto& distribution : distributions[nhw])
    {
      print("\t[{}]\n", fmt::join(distribution, ","));

//    dim += dimDistribution(distribution, MMT, JJ);
    }
  }

  print("Basis dimension: {}\n", dim);
}
