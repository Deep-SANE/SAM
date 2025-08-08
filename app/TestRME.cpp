#include <Databases.h>
#include <TwoBodyOperatorJT.h>

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  // example: two-body Hamiltonian (bare) with N3LO_EMN500:
  int hw = -1, valenceShell = 0, Nmax = 4, maxShell = 4;
  HO::Table_nljmjmt labels(maxShell);

  JT::TwoBodyOperator NN("N3LO_EMN500_bare", labels, hw, Nmax, valenceShell,
                         maxShell);

  auto tensors = NN.expand(dbs.cgcs, dbs.racah, dbs.racah9, dbs.wigU3SO3db,
                           dbs.wigU4SU2SU2db);

#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif
}
