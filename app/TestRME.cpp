#include <Databases.h>
#include <qcx_print.h>
#include <TwoBodyOperatorJT.h>

using qcx_utils::print;

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

  auto tensors = NN.expand(dbs.cgcdb, dbs.racahdb, dbs.racah9db, dbs.wigU3SO3db,
                           dbs.wigU4SU2SU2db);

  print("Number of tesors: {}\n", tensors.size());

#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif
}
