// U3xU4ManyNucleonBasis: build the U(3)xU(4) symmetry-adapted basis of a
// nucleus and print its dimension, optionally with every basis irrep.
//
// Asks for Z, N, Nmax and JJ = 2J. A basis irrep is a product of single-shell
// irreps gamma (one per occupied HO shell: U(3)xU(4) labels and their
// multiplicity alpha) coupled shell by shell through the omega irreps; each
// irrep then splits into states with good L, S, T and J (kappa, zeta are the
// L and S multiplicities). This is the basis MatrixOperatorJT uses.
//
// Run it as shipped:
//   mpirun -np 1 bin/tutorial/U3xU4ManyNucleonBasis

#include <chrono>
#include <iostream>

#include <print.h>
#include <utils.h>

#include <sam.h>
#include <U3xU4ManyNucleonBasis.h>

int main(int argc, char* argv[])
{
  using utils::print;

  sam::initialize(&argc, &argv);

  int Z, N, Nmax, JJ, list;

  std::cout << "Enter number of protons Z: ";
  std::cin >> Z;
  std::cout << "Enter number of neutrons N: ";
  std::cin >> N;
  std::cout << "Enter Nmax: ";
  std::cin >> Nmax;
  std::cout << "Enter JJ = 2J: ";
  std::cin >> JJ;
  std::cout << "List every irrep (1 = yes, 0 = no): ";
  std::cin >> list;

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);

  auto t0 = std::chrono::steady_clock::now();
  basis.construct();
  auto t1 = std::chrono::steady_clock::now();

  if (list == 1)
  {
    auto irreps = basis.irreps();
    const auto& labels = basis.decomposition();

    for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
    {
      auto [gamma, omega] = irreps[irrep_id];

      print("irrep {:5}: gamma = {} ; omega = {}\n", irrep_id, gamma, omega);
      // (kappa max, LL, zeta max, SS, TT, offset) of each L, S, T block:
      print("             kappamax LL zetamax SS TT offset = {}\n",
            labels[irrep_id]);
      print("             first state = {}\n", irreps.offset(irrep_id));
    }
  }

  print("\nZ = {}, N = {}, Nmax = {}, J = {}/2: basis dimension {}\n", Z, N,
        Nmax, JJ, basis.dim());

  std::chrono::duration<double> t = t1 - t0;
  print("Time: {:.2f} s\n", t.count());

  sam::finalize();
}
