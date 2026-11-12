// Spectra and wave functions of light nuclei with JT::Operator: the
// Hamiltonian "Hamiltonian with N2LO_opt_bare" (relative kinetic energy +
// NNLOopt NN interaction + Lawson term 100 N_cm) at hw = 20 MeV and Nmax = 4,
// for 4He (J = 0), 6Li (J = 1) and 6He (J = 0). For each nucleus:
//   - the five lowest eigenvalues (matrix_CSR + Spectra);
//   - the ground-state probabilities of the U(3) and U(4) irreps in each hw
//     subspace (sam::U3_wavefunction_analyzer, U4_wavefunction_analyzer),
//     and that each set adds up to 100%.
//
// Needs the released expansions of N2LO_opt_bare, Kinetic and CoMNumberOp:
//   make data OPS="N2LO_opt_bare Kinetic CoMNumberOp"
// They are linked from the configured data directory (SAM_TEST_DATA_DIR, set
// by the test Makefile) into the temporary one; without them the cases are
// skipped.
//
// Reference numbers were produced by this code (SAM main, Nov 2026) with the
// data-v1 release.

#include <cstdlib>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <sam.h>

#include <eigensolver.h>
#include <OperatorJT.h>
#include <WavefunctionAnalyzer.h>

#include "test_helpers.h"

namespace
{
  namespace fs = std::filesystem;
  constexpr int hw = 20, Nmax = 4, nev = 5;

  struct Probability
  {
    int hw;
    std::string irrep;
    double percent;
  };

  struct Nucleus
  {
    std::string name;
    int Z, N, JJ;
    std::vector<double> evals;
    std::vector<Probability> u3, u4;
  };

  // released expansions this test needs, relative to the data directory
  const std::vector<std::string> needed = {
    "OneBodyOperatorJT/Kinetic/Expansion/Kinetic_hw20",
    "OneBodyOperatorJT/CoMNumberOp/Expansion/CoMNumberOp_hw20",
    "TwoBodyOperatorJT/Kinetic/Expansion/Kinetic_hw20",
    "TwoBodyOperatorJT/CoMNumberOp/Expansion/CoMNumberOp_hw20",
    "TwoBodyOperatorJT/N2LO_opt_bare/Expansion/N2LO_opt_bare_hw20",
  };

  // link the released files into the temporary data directory; returns an
  // empty string, or why the data is not available
  std::string link_released_data()
  {
    const char* source = std::getenv("SAM_TEST_DATA_DIR");
    if (!source || !*source)
      return "SAM_TEST_DATA_DIR is not set (run through make test)";
    for (const auto& file : needed)
    {
      fs::path from = fs::path(source) / file;
      if (!fs::exists(from))
        return "missing " + from.string() +
               "; download with: make data OPS=\"N2LO_opt_bare Kinetic "
               "CoMNumberOp\"";
      fs::path to = fs::path(sam::data_dir()) / file;
      fs::create_directories(to.parent_path());
      fs::create_symlink(fs::absolute(from), to);
    }
    return "";
  }

  // ground-state probabilities from an analyzer, keyed by "hw:irrep"
  template <class Subspaces, class Irreps>
  std::map<std::string, double> probabilities(const Subspaces& subspaces,
                                              Irreps irreps)
  {
    std::map<std::string, double> p;
    for (const auto& s : subspaces)
    {
      const auto& irrs = irreps(s);
      for (size_t i = 0; i < irrs.size(); i++)
      {
        std::ostringstream key;
        key << int(s.hw) << ":" << irrs[i];
        p[key.str()] += s.probs[i][0];
      }
    }
    return p;
  }

  void compare(const std::map<std::string, double>& got,
               const std::vector<Probability>& expected,
               const std::string& group)
  {
    double total = 0;
    for (const auto& [key, value] : got) total += value;
    test::near(total, 100.0, group + " probabilities add up to 100%", 1e-8, 0);
    for (const auto& e : expected)
    {
      std::string key = std::to_string(e.hw) + ":" + e.irrep;
      auto it = got.find(key);
      test::require(it != got.end(), group + " irrep " + e.irrep + " at " +
                    std::to_string(e.hw) + "hw is missing");
      test::near(it->second, e.percent, group + " probability of " + e.irrep +
                 " at " + std::to_string(e.hw) + "hw", 1e-6, 1e-7);
    }
  }

  void check(const Nucleus& x)
  {
    U3xU4::ManyNucleonBasis basis(x.N, x.Z, Nmax, x.JJ);
    basis.construct();
    HO::Table_nljmjmt HO_labels(basis.maxShell());
    U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

    JT::Operator H("Hamiltonian with N2LO_opt_bare", x.N, x.Z, hw, Nmax,
                   HO_labels, gelfand_labels);
    auto mat = H.matrix_CSR(basis);
    auto [evals, evecs] = utils::eigensolver_symmetric_spectra(mat, nev, 1e-10);

    test::require(evals.size() == x.evals.size(), "number of eigenvalues");
    for (size_t k = 0; k < evals.size(); k++)
      test::near(evals[k], x.evals[k], "eigenvalue " + std::to_string(k),
                 1e-7, 1e-9);

    compare(probabilities(sam::U3_wavefunction_analyzer(basis, evecs, 1),
                          [](const auto& s) -> const auto& { return s.u3s; }),
            x.u3, "U(3)");
    compare(probabilities(sam::U4_wavefunction_analyzer(basis, evecs, 1),
                          [](const auto& s) -> const auto& { return s.u4s; }),
            x.u4, "U(4)");
  }
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  test::Suite suite;

  std::vector<Nucleus> nuclei = {
    {"4He, J = 0", 2, 2, 0,
     // lowest eigenvalues (MeV)
     {-22.361781042786436, 8.7465565683768123, 18.2138118498392,
      24.098758567718978, 28.24516203465188},
     // ground state: U(3) probabilities (%) above 0.1%, by hw excitation
     {{0, "[0,0,0]", 90.272058453994703}, {2, "[2,0,0]", 3.102429472739066},
      {4, "[4,0,0]", 6.5426903202115012}},
     // ground state: U(4) probabilities (%) above 0.1%
     {{0, "[0,0,0,0]", 90.272058453994703},
      {2, "[0,0,0,0]", 0.95681745385674655},
      {2, "[2,2,0,0]", 2.1432682448555118},
      {4, "[0,0,0,0]", 3.0884851638323099},
      {4, "[2,2,0,0]", 3.4948784033652451}}},
    {"6Li, J = 1", 3, 3, 2,
     // lowest eigenvalues (MeV)
     {-17.625013429109355, -9.3596896222563988, -5.6924989427292925,
      -3.4836144048127937, 14.049299436321759},
     // ground state: U(3) probabilities (%) above 0.1%, by hw excitation
     {{0, "[2,0,0]", 73.956082416035855}, {0, "[1,1,0]", 3.1107404092917634},
      {2, "[4,0,0]", 8.543518742172953}, {2, "[3,1,0]", 2.5681987480852562},
      {2, "[2,2,0]", 2.8742081047904007}, {4, "[6,0,0]", 4.4721849428795659},
      {4, "[5,1,0]", 1.3143424334077349}, {4, "[4,2,0]", 2.900842551589117}},
     // ground state: U(4) probabilities (%) above 0.1%
     {{0, "[1,1,0,0]", 73.956082416035855},
      {0, "[2,0,0,0]", 3.1107404092917634},
      {2, "[1,1,0,0]", 12.416367781239931},
      {2, "[2,0,0,0]", 0.19179780953882819},
      {2, "[3,2,1,0]", 1.2213852793378177},
      {2, "[3,3,0,0]", 0.20579654048107415},
      {4, "[1,1,0,0]", 5.7287059754429421},
      {4, "[2,0,0,0]", 0.23835117618075555},
      {4, "[3,2,1,0]", 1.7302358957282074},
      {4, "[3,3,0,0]", 1.0696282469018064}}},
    {"6He, J = 0", 2, 4, 0,
     // lowest eigenvalues (MeV)
     {-16.001963898055081, -2.2813294041066365, 14.394062554681947,
      16.694152281140642, 20.829620333346227},
     // ground state: U(3) probabilities (%) above 0.1%, by hw excitation
     {{0, "[2,0,0]", 71.791990250852322}, {0, "[1,1,0]", 7.362415628644797},
      {2, "[4,0,0]", 6.9682398396352561}, {2, "[3,1,0]", 2.5944217563953651},
      {2, "[2,2,0]", 2.9062182535472503}, {2, "[1,0,0]", 0.21062300447332993},
      {4, "[6,0,0]", 4.1187373762503787}, {4, "[5,1,0]", 1.4177894562928999},
      {4, "[4,2,0]", 2.3808911853427248}, {4, "[3,0,0]", 0.17249274932567288}},
     // ground state: U(4) probabilities (%) above 0.1%
     {{0, "[1,1,0,0]", 71.791990250852322}, {0, "[2,0,0,0]", 7.362415628644797},
      {2, "[1,1,0,0]", 9.390122399083273},
      {2, "[2,0,0,0]", 0.60637101743685917},
      {2, "[3,2,1,0]", 2.4319471627878309},
      {2, "[3,3,0,0]", 0.16406013851549586},
      {4, "[1,1,0,0]", 3.9617939771150503},
      {4, "[2,0,0,0]", 0.55110288105707039},
      {4, "[3,2,1,0]", 2.779593040824655},
      {4, "[3,3,0,0]", 0.59551252509084596},
      {4, "[4,2,0,0]", 0.14169288913740136}}}
  };

  std::string missing = link_released_data();
  for (const auto& x : nuclei)
  {
    std::string label = x.name + ", Nmax = 4, N2LO_opt_bare: spectrum and "
                        "U(3), U(4) probabilities";
    if (!missing.empty())
      suite.skip(label, missing);
    else
      suite.run(label, [&] { check(x); });
  }

  sam::finalize();
  return suite.result();
}
