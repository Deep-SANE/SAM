#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <sam.h>

#include <HOutilities.h>

// Verify TwBME_general's PRODUCTION load path: construct by operator name, call
// read() (which resolves the filename from the name and reads it), and confirm
// the loaded matrix elements match the collaborator's text file. The reference
// list (expected_twbme.txt) is produced by ref_parse_me2j.py on that text file.
//
// read() resolves:
//   <data>/General_operators/GamowTellerMinus_2B_N3LO_EM500_bare_hw16_emax14_e2max28.me2j.gz
// (<data> = $SAM_DATA_DIR if set, else "data/"). The emax/e2max in the FILENAME
// is only a label -- the reader uses the file's own header -- and gzopen reads a
// plain (uncompressed) file transparently, so the collaborator's .me2j can sit
// at that path as-is.
//   argv[1] = expected reference list (optional)
int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  std::string expected = (argc > 1) ? argv[1] :
    "/Users/phongdang/Downloads/expected_twbme.txt";

  int hw = 16, Nmax = 4, valenceShell = 0, maxShell = 2;

  HO::TwBME_general TBME("GamowTellerMinus_2B_N3LO_EM500_bare",
                         hw, Nmax, valenceShell, maxShell);

  TBME.read();
  TBME.is();
  std::cout << "Loaded " << TBME.num_mes() << " matrix elements\n";

  std::ifstream in(expected);
  if (!in)
  {
    std::cerr << "ERROR: cannot open reference list: " << expected << "\n";
    sam::finalize();
    return 1;
  }

  std::cout << std::scientific << std::setprecision(6);

  int a, b, c, d, Jab, Jcd;
  double v;
  long n = 0, bad = 0;
  double maxdiff = 0.0;
  while (in >> a >> b >> c >> d >> Jab >> Jcd >> v)
  {
    double got = TBME.getME(static_cast<uint16_t>(a), static_cast<uint16_t>(b),
                            static_cast<uint16_t>(c), static_cast<uint16_t>(d),
                            static_cast<HO::qn_t>(Jab),
                            static_cast<HO::qn_t>(Jcd));
    double diff = std::fabs(got - v);
    if (diff > maxdiff) maxdiff = diff;
    ++n;
    if (diff > 1e-6)
    {
      if (bad < 25)
        std::cout << "MISMATCH (" << a << "," << b << "," << c << "," << d
                  << "; Jab=" << Jab << ",Jcd=" << Jcd << ")"
                  << "  expected=" << v << "  got=" << got
                  << "  diff=" << diff << "\n";
      ++bad;
    }
  }

  std::cout << "\nCompared " << n << " reference entries: " << (n - bad)
            << " match, " << bad << " mismatch.  max|diff| = " << maxdiff
            << "\n";
  std::cout << "Stored count " << TBME.num_mes()
            << (TBME.num_mes() == static_cast<size_t>(n)
                ? "  == reference count\n"
                : "  != reference count (different cut/dedup!)\n");

  sam::finalize();
  return (bad == 0 && TBME.num_mes() == static_cast<size_t>(n)) ? 0 : 2;
}
