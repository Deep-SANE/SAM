// Expansion of two-body operators into U(3)xU(4) tensors, for two cheap
// operators computed on the fly, Kinetic and CoMNumberOp, at hw = 20 MeV and
// Nmax = maxShell = 4 (the expansion step of app/tutorial/TwoBodyOperatorJT.cpp,
// without the nuclear potentials, which need interaction files).
//
// For each operator:
//   1. expand_new(): number of tensors and a signed checksum over all
//      strengths (expansion_checksum.h);
//   2. expand() on a missing file stops with the `make data` message; after
//      writeBinary() of the expansion, expand() (readBinary_MPI) and
//      readBinary() both read it back unchanged;
//   3. the maxShell-4 file read for maxShell 3 equals expand_new() at
//      maxShell 3 (readBinary keeps only the tensors that fit);
//   4. the same file read for maxShell 5 stops with the "only reaches" error;
//   5. read at hw = 25: Kinetic strengths scale by 25/20, CoMNumberOp's don't.
//
// expand_new() runs on OpenMP threads, so this also guards the per-thread
// setup of the u3lib/u4lib coefficient caches.
//
// The run uses a temporary data directory (set by the test Makefile).
//
// Reference numbers were produced by this code (SAM main, Nov 2026).

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <sam.h>

#include <TwoBodyOperatorJT.h>

#include "expansion_checksum.h"
#include "test_helpers.h"

namespace
{
  constexpr int hw = 20, maxShell = 4;

  struct Expected
  {
    std::string name;
    size_t tensors;
    double checksum;
  };

  // the operator with its label tables (tensors refer to the tables)
  struct Op
  {
    HO::Table_nljmjmt HO_labels;
    U3xU4::Table_gelfand gelfand_labels;
    JT::TwoBodyOperator op;

    Op(const std::string& name, int hw_, int maxShell_)
        : HO_labels(maxShell_), gelfand_labels(maxShell_),
          op(name, hw_, maxShell_, 0, maxShell_, HO_labels, gelfand_labels)
    { }
  };

  // run work(), which must throw a std::runtime_error whose message
  // contains `text`
  template <class F>
  void require_error(F work, const std::string& text)
  {
    try
    {
      work();
    }
    catch (const std::runtime_error& e)
    {
      test::require(std::string(e.what()).find(text) != std::string::npos,
                    "error message should contain \"" + text + "\", got: " +
                    e.what());
      return;
    }
    throw std::runtime_error("expected an error containing \"" + text + "\"");
  }
}

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);
  test::Suite suite;

  std::vector<Expected> operators = {
    {"Kinetic", 26, 135.09204921427738},
    {"CoMNumberOp", 16, -42.964167549885303},
  };

  for (const auto& x : operators)
  {
    Op op4(x.name, hw, maxShell);
    U3xU4::MultiShellTensors_phys tensors;
    std::string file = op4.op.dirname() + "/Expansion/" + x.name + "_hw20";

    suite.run(x.name + ": expand_new() at maxShell 4", [&]
    {
      tensors = op4.op.expand_new();
      test::require(tensors.size() == x.tensors, "number of tensors is " +
                    std::to_string(tensors.size()) + ", expected " +
                    std::to_string(x.tensors));
      test::near(test::checksum(tensors), x.checksum, "checksum", 1e-10, 1e-12);
    });

    suite.run(x.name + ": expand() needs the file; writeBinary + expand() "
              "and readBinary() read it back", [&]
    {
      require_error([&] { op4.op.expand(); }, "make data");
      std::filesystem::create_directories(op4.op.dirname() + "/Expansion");
      op4.op.writeBinary(file, tensors);
      for (const auto& loaded : {op4.op.expand(), op4.op.readBinary(file)})
      {
        test::require(loaded.size() == tensors.size(), "number of tensors");
        test::near(test::checksum(loaded), test::checksum(tensors), "checksum",
                   1e-12, 1e-14);
      }
    });

    suite.run(x.name + ": the maxShell-4 file serves maxShell 3", [&]
    {
      Op op3(x.name, hw, 3);
      auto expected = op3.op.expand_new();
      auto loaded = op3.op.readBinary(file);
      test::require(loaded.size() == expected.size(), "number of tensors is " +
                    std::to_string(loaded.size()) + ", expected " +
                    std::to_string(expected.size()));
      test::near(test::checksum(loaded), test::checksum(expected), "checksum",
                 1e-12, 1e-13);
    });

    suite.run(x.name + ": the maxShell-4 file cannot serve maxShell 5", [&]
    {
      Op op5(x.name, hw, 5);
      require_error([&] { op5.op.readBinary(file); },
                    "only reaches maxShell = 4");
    });

    suite.run(x.name + ": read at hw = 25", [&]
    {
      Op op25(x.name, 25, maxShell);
      auto loaded = op25.op.readBinary(file);
      double factor = (x.name == "Kinetic") ? 25.0 / 20.0 : 1.0;
      test::near(test::checksum(loaded), factor * test::checksum(tensors),
                 "checksum scaled by " + std::to_string(factor), 1e-12, 1e-13);
    });
  }

  sam::finalize();
  return suite.result();
}
