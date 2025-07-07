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

      auto [first_gamma_id, n_gammas] = basis.dist_gammas(dist_id);
      auto end_gamma_id = first_gamma_id + n_gammas;
      

      for (std::size_t i_gamma = first_gamma_id; i_gamma < end_gamma_id; i_gamma++)
      {
        auto [gammas_u4, gammas_su3] = basis.gamma_irreps(i_gamma);
        auto n_gammas = gammas_u4.size();

        for (int i_gamma = 0; i_gamma < n_gammas; i_gamma++)
        {
          print("\tgamma[{}]:", i_gamma);

          auto& mu4 = gammas_u4[i_gamma];
          print(" {}[{} {} {} {}]", mu4.mult, mu4.u4.n14(), mu4.u4.n24(),
                mu4.u4.n34(), mu4.u4.n44());

          auto& msu3 = gammas_su3[i_gamma];
          print(" {}({} {})\n", msu3.mult, msu3.lm, msu3.mu);
        }

        print("\n");
      }
    }
  }

//print("Basis dimension: {}\n", dim);
}
