#pragma once

#include <cstdint>
#include <functional>
#include <tuple>

namespace detail
{
  template <typename T>
  inline void hash_combine(std::size_t& seed, const T& value)
  {
    std::hash<T> hasher;
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
} // namespace detail

namespace std
{
  template <typename... Ts>
  struct hash<std::tuple<Ts...>>
  {
    size_t operator()(const std::tuple<Ts...>& key) const noexcept
    {
      size_t seed = 0;
      std::apply([&](auto const&... args) 
      {
        (..., detail::hash_combine(seed, args));
      }, key);
      return seed;
    }
  };
}
