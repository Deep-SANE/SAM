// clang-format off

#include <qcx_print.h>

#include <Databases.h>
#include <U3xU4irrep.h>
#include <U3xU4MultiShellTensors.h>
#include <U3xU4SingleShellTensors.h>
#include <U4xSU3Irrep.h>
#include <U4xSU3ManyNucleonBasis.h>

using qcx_utils::print;

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  // (1) bra irrep:

  std::vector<U4xSU3::ManyNucleonBasis::DistributionElement>
    dist_bra = {1,4,1,2};

  U4xSU3::ManyNucleonBasis::GammaElement
    gamma0_bra({1,0,0,0},1,{0,0}),
    gamma1_bra({3,1,0,0},1,{1,0}),
    gamma2_bra({1,0,0,0},1,{2,0}),
    gamma3_bra({1,1,0,0},1,{2,2});

  U4xSU3::ManyNucleonBasis::OmegaElement
    omega01_bra  (1,{3,2,0,0},1,{1,0}),
    omega012_bra (1,{3,2,1,0},1,{1,1}),
    omega0123_bra(1,{3,1,0,0},2,{2,2});

  U4xSU3::Irrep bra(dist_bra,
                    {gamma0_bra, gamma1_bra, gamma2_bra, gamma3_bra},
                    {omega01_bra, omega012_bra, omega0123_bra});

  print("Bra irrep: dist = {}, gamma = {}, omega = {}\n", bra.dist, bra.gamma, bra.omega);

  // (2) ket irrep:

  std::vector<U4xSU3::ManyNucleonBasis::DistributionElement>
    dist_ket = {0,4,2,2};

  U4xSU3::ManyNucleonBasis::GammaElement
    gamma1_ket({2,1,1,0},1,{2,1}),
    gamma2_ket({2,0,0,0},1,{2,1}),
    gamma3_ket({1,1,0,0},1,{2,2});

  U4xSU3::ManyNucleonBasis::OmegaElement
    omega12_ket (1,{1,1,0,0},2,{1,2}),
    omega123_ket(1,{2,2,0,0},1,{2,0});

  U4xSU3::Irrep ket(dist_ket,
                    {gamma1_ket, gamma2_ket, gamma3_ket},
                    {omega12_ket, omega123_ket});

  print("Ket irrep: dist = {}, gamma = {}, omega = {}\n\n", ket.dist, ket.gamma, ket.omega);

  // (3) irreps unification:

  U4xSU3::IrrepUnifier iu;
  const auto& [ubra, uket] = iu.unify(bra, ket);

  print("Bra unified irrep: gamma = {}, omega = {}, shells = {}\n", ubra.gamma, ubra.omega, ubra.gamma_shells);
  print("Ket unified irrep: gamma = {}, omega = {}, shells = {}\n", uket.gamma, uket.omega, uket.gamma_shells);

  // (4) tensor:

  U3xU4::Tadat
    Gamma0(0, 1, 0, {}, {}, {{0,0,0},{1,0,0,0}}, 1, 1),
    Gamma1(1, 1, 1, {}, {}, {{2,1,0},{2,1,1,0}}, 1, 1),
    Gamma2(2, 0, 1, {}, {}, {{2,2,0},{1,1,1,0}}, 1, 1);

  U3xU4::irrep
    Omega01 ({2,1,0},{3,1,1,0}),
    Omega012({2,2,0},{4,2,2,0});

  U3xU4::MultiShellTensor T({Gamma0, Gamma1, Gamma2}, {{Omega01, 1, 1},{Omega012, 1, 1}});

  const auto& Gamma = T.single_shell_tensors;
  const auto& Omega = T.coupled_irreps;

  print("\nTensor: Gamma = {}, Omaga = {}\n", Gamma, Omega);

  // (5) unify tensor with irreps:

  auto uT = U3xU4::unify_tensor(T, ubra);

  const auto& uGamma = uT.single_shell_tensors;
  const auto& uOmega = uT.coupled_irreps;

  print("\nUnified tensor: Gamma = {}, Omega = {} \n", uGamma, uOmega);

  // (6) calculate RMEs:

  auto RMEs = uT.RME(uket, ubra, dbs);

  print("\nRMEs:\n");
  for (const auto& rme : RMEs)
    print("{: .8f}\n", rme);

#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif
}
