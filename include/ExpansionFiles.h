#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#include <unistd.h>

// checks on the operator expansion files kept under the data directory.
// expanding an operator is very costly, so SAM never starts or extends an
// expansion in the middle of a calculation; these helpers stop the run with
// a message saying what to do instead.
namespace sam
{
  // how to run an expansion separately with one of the expansion programs
  // in app/tutorial/ (app = "OneBodyOperatorJT" or "TwoBodyOperatorJT"),
  // copied into app/work/ so the shipped program stays unchanged:
  inline std::string expansion_steps(const std::string& app,
                                     int maxShell_needed)
  {
    return "  1. cp app/tutorial/" + app + ".cpp app/work/   (once)\n"
           "  2. in app/work/" + app + ".cpp set the operator name, hw and "
           "maxShell = " + std::to_string(maxShell_needed) + "\n"
           "  3. rebuild with 'make -j 8 apps'\n"
           "  4. run bin/work/" + app + " with mpirun";
  }

  // no expansion file on disk:
  [[noreturn]] inline void throw_missing_expansion(const std::string& filename,
                                                   const std::string& name,
                                                   int maxShell_needed,
                                                   const std::string& app)
  {
    throw std::runtime_error(
      "Sorry, there is no expansion file\n  " + filename +
      "\nExpanding " + name + " is very costly, so SAM does not do it "
      "automatically.\nIf it is released, download it with\n  make data OPS=" +
      name + "    (see 'make data-list')\nOtherwise expand it separately:\n" +
      expansion_steps(app, maxShell_needed));
  }

  // the file on disk does not reach the shells this calculation needs:
  [[noreturn]] inline void throw_expansion_too_small(
    const std::string& filename, int maxShell_file, int maxShell_needed,
    const std::string& app)
  {
    throw std::runtime_error(
      "Sorry, the expansion file\n  " + filename + "\nonly reaches maxShell = " +
      std::to_string(maxShell_file) + ", but this calculation needs maxShell = " +
      std::to_string(maxShell_needed) +
      ".\nExtending an expansion is very costly, so SAM does not do it "
      "automatically.\nCheck 'make data-list' for a released file that reaches "
      "it, or extend it separately:\n" + expansion_steps(app, maxShell_needed) +
      "\nReleased files are read-only; unlock this one first with\n  chmod u+w '" +
      filename + "'");
  }

  // fail early, before hours of computing, if the expansion output cannot be
  // written. files from `make data` are read-only on purpose, so a deliberate
  // re-expansion or extension has to unlock them first.
  inline void require_writable(const std::string& filename)
  {
    namespace fs = std::filesystem;
    const fs::path file(filename);
    const fs::path dir = file.has_parent_path() ? file.parent_path() : ".";

    // access() asks the operating system whether this process may write
    if (fs::exists(file) && access(filename.c_str(), W_OK) != 0)
      throw std::runtime_error(
        "Sorry, the expansion file\n  " + filename +
        "\nis read-only (released files are protected). To replace or extend "
        "it on purpose, run\n  chmod u+w '" + filename + "'\nand start again.");

    std::error_code ec;   // several ranks may race here; that is harmless
    fs::create_directories(dir, ec);
    if (access(dir.c_str(), W_OK) != 0)
      throw std::runtime_error(
        "Sorry, cannot write expansion files into " + dir.string());
  }
}
