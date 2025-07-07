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
  int n0hw = Nmax % 2;

  U4xSU3::ManyNucleonBasis basis(A, MMT, Nmax, JJ);
  basis.construct();

//uint64_t dim = 0;

  for (int nhw = n0hw; nhw <= Nmax; nhw += 2)
  {
    print("hw: {}\n", nhw);

    auto [dist_id_begin, dist_id_end] = basis.distributions_ids(nhw);

    for (std::size_t dist_id = dist_id_begin; dist_id < dist_id_end; dist_id++)
    {
      print("distribution: {}\n", basis.distribution(dist_id));

      auto [gamma_id_begin, n_gammas] = basis.dist_gammas(dist_id);

      for (int i_gamma = 0; i_gamma < n_gammas; i_gamma++)
      {
        auto gamma_id = gamma_id_begin + i_gamma;
        auto [gammas_u4, gammas_su3] = basis.gamma_irreps(gamma_id);
        auto n_irreps = gammas_u4.size();

        for (int i_irrep = 0; i_irrep < n_irreps; i_irrep++)
        {
          print("\tgamma[{}]:", i_irrep);

          auto& mu4 = gammas_u4[i_irrep];
          print(" {}[{} {} {} {}]", mu4.mult, mu4.u4.n14(), mu4.u4.n24(),
                mu4.u4.n34(), mu4.u4.n44());

          auto& msu3 = gammas_su3[i_irrep];
          print(" {}({} {})\n", msu3.mult, msu3.lm, msu3.mu);
        }

        print("\n");
      }
    }
  }

//print("Basis dimension: {}\n", dim);
}
