#pragma once

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// small in-repo test framework, the same as in u3lib/u4lib:
//   require(cond, what)                    a check that must hold
//   near(actual, expected, what, atol, rtol)
//   Suite: runs named cases, prints [TEST]/[PASS]/[FAIL] and a summary, and
//          returns a nonzero exit code if any case failed.
namespace test
{
  inline void require(bool ok, const std::string& what)
  {
    if (!ok)
      throw std::runtime_error(what);
  }

  inline void near(double actual, double expected, const std::string& what,
                   double atol = 1e-12, double rtol = 1e-10)
  {
    if (!(std::isfinite(actual) && std::isfinite(expected) &&
          std::abs(actual - expected) <= atol + rtol * std::abs(expected)))
    {
      std::ostringstream s;
      s << std::setprecision(17) << what << ": got " << actual
        << ", expected " << expected;
      throw std::runtime_error(s.str());
    }
  }

  struct Suite
  {
    int passed = 0, failed = 0;

    template <class F>
    void run(const std::string& label, F work)
    {
      std::cout << "[TEST] " << label << std::endl;
      try
      {
        work();
        ++passed;
        std::cout << "[PASS] " << label << std::endl;
      }
      catch (const std::exception& e)
      {
        ++failed;
        std::cout << "[FAIL] " << label << ": " << e.what() << std::endl;
      }
    }

    int result() const
    {
      std::cout << "Summary: " << passed << " passed, " << failed
                << " failed\n";
      return failed ? 1 : 0;
    }
  };
} // namespace test
