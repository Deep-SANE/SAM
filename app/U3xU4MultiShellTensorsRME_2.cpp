// clang-format off

#include <qcx_print.h>

#include <Databases.h>
#include <U3xU4irrep.h>
#include <U3xU4GammaOmega.h>
#include <U3xU4MultiShellTensors.h>
#include <U3xU4ManyNucleonBasis.h>

using qcx_utils::print;

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  // (1) bra irrep:

  U3xU4::GammaElement
    gamma0_bra(0,1,{0,0,0},{1,0,0,0}),
    gamma1_bra(1,1,{2,1,1},{3,1,0,0}),
    gamma2_bra(2,1,{2,0,0},{1,0,0,0}),
    gamma3_bra(3,1,{4,2,0},{1,1,0,0});

  U3xU4::OmegaElement
    omega01_bra  (1,{1,0,0},1,{3,2,0,0}),
    omega012_bra (1,{2,1,0},1,{3,2,1,0}),
    omega0123_bra(2,{4,2,0},1,{3,1,0,0});

  U3xU4::ManyNucleonBasis::Irrep 
    bra({gamma0_bra, gamma1_bra, gamma2_bra, gamma3_bra},
        {omega01_bra, omega012_bra, omega0123_bra});
/*
  U3xU4::GammaElement
    gamma0_bra(0,1,{0,0,0},{1,0,0,0}),
    gamma1_bra(1,1,{1,1,0},{2,0,0,0}),
    gamma2_bra(2,1,{2,0,0},{1,0,0,0});

  U3xU4::OmegaElement
    omega01_bra  (1,{1,1,0},1,{2,1,0,0}),
    omega012_bra (1,{1,0,0},1,{2,1,1,0});

  U3xU4::ManyNucleonBasis::Irrep 
    bra({gamma0_bra, gamma1_bra, gamma2_bra},
        {omega01_bra, omega012_bra});
*/
  print("Bra irrep: gamma = {}, omega = {}\n", bra.gamma, bra.omega);

  // (2) ket irrep:

  U3xU4::GammaElement
    gamma1_ket(1,1,{3,1,0},{2,1,1,0}),
    gamma2_ket(2,1,{3,1,0},{2,0,0,0}),
    gamma3_ket(3,1,{4,2,0},{1,1,0,0});

  U3xU4::OmegaElement
    omega12_ket (2,{3,2,0},1,{1,1,0,0}),
    omega123_ket(1,{2,0,0},1,{2,2,0,0});

  U3xU4::ManyNucleonBasis::Irrep 
    ket({gamma1_ket, gamma2_ket, gamma3_ket},
        {omega12_ket, omega123_ket});
/*
  U3xU4::ManyNucleonBasis::Irrep 
    ket({gamma0_bra, gamma1_bra, gamma2_bra},
        {omega01_bra, omega012_bra});
*/
  print("Ket irrep: gamma = {}, omega = {}\n\n", ket.gamma, ket.omega);

  // (3) irreps unification:

  U3xU4::IrrepUnifier iu;
  const auto& [ubra, uket] = iu.unify(bra, ket);

  print("Bra unified irrep: gamma = {}, omega = {}\n", ubra.gamma, ubra.omega);
  print("Ket unified irrep: gamma = {}, omega = {}\n", uket.gamma, uket.omega);

  // (4) tensor:

  U3xU4::SingleShellTensor
    Gamma0(0, 1, 0, {}, {}, {{0,0,0},{1,0,0,0}}, 1, 1),
    Gamma1(1, 1, 1, {}, {}, {{2,1,0},{2,1,1,0}}, 1, 1),
    Gamma2(2, 0, 1, {}, {}, {{2,2,0},{1,1,1,0}}, 1, 1),
    Gamma5(5, 1, 1, {}, {}, {{0,0,0},{0,0,0,0}}, 1, 1);

  U3xU4::OmegaElement
    Omega01  (1,{2,1,0},1,{3,1,1,0}),
    Omega012 (1,{2,2,0},1,{4,2,2,0}),
    Omega0125(1,{2,2,0},1,{4,2,2,0});

  U3xU4::MultiShellTensor 
    T({Gamma0, Gamma1, Gamma2/*, Gamma5*/}, {Omega01, Omega012/*, Omega0125*/});
/*
  U3xU4::SingleShellTensor
    Gamma0(1, 1, 1, {}, {}, {{2,1,0},{0,0,0,0}}, 1, 1),
    Gamma1(2, 1, 1, {}, {}, {{2,1,0},{0,0,0,0}}, 1, 1);

  U3xU4::OmegaElement
    Omega01  (1,{4,2,0},1,{0,0,0,0});

  U3xU4::MultiShellTensor 
    T({Gamma0, Gamma1}, {Omega01});
*/
  print("\nTensor: Gamma = {}, Omaga = {}\n", T.Gamma, T.Omega);

  // (5) unify tensor with irreps:

  auto uT = U3xU4::unify_tensor(T, ubra);
  if(!uT.has_value())
  {
    std::cout << "Tensor acting out of states" << std::endl;
    return 1;
  }

  print("\nUnified tensor: Gamma = {}, Omega = {} \n", uT->Gamma, uT->Omega);

  // (6) calculate RMEs:

  auto RMEs = uT->RME(uket, ubra, dbs);

  print("\nRMEs:\n");
  for (size_t i = 0; i < RMEs.size(); i++)
    print("{: .8f}\n", RMEs[i]);

#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif
}
