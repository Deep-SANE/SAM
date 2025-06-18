#ifndef SAM_SAM_CONFIG_H
#define SAM_SAM_CONFIG_H

// C++23:
#if defined __cpp_lib_print && defined __cpp_lib_format && defined __cpp_lib_format_ranges

#include <print>

using std::print;
//using std::println;

// {fmt} library:
#elif defined(HAVE_FMT)

#include <utility>

#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <fmt/ranges.h>

using fmt::print;

/*
template <typename... Args>
void println(fmt::format_string<Args...> fmt, Args&&... args) {
  return fmt::print(stdout, fmt, std::forward<Args>(args)...);
  std::putc('\n', stdout);
}
*/

// none of the above options:
#else

template <typename... Ts>
void print(const Ts&...) { }

//template <typename... Ts>
//void println(const Ts&...) { }

#endif

#endif
