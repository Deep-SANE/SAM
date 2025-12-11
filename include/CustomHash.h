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

/*
namespace std
{
  template <>
  struct hash<std::tuple<int16_t, int16_t, int16_t, int16_t, int16_t>>
  {
    size_t operator()(const std::tuple<int16_t, int16_t, int16_t, int16_t,
                                       int16_t>& key) const noexcept
    {
      size_t seed = 0;
      detail::hash_combine(seed, std::get<0>(key));
      detail::hash_combine(seed, std::get<1>(key));
      detail::hash_combine(seed, std::get<2>(key));
      detail::hash_combine(seed, std::get<3>(key));
      detail::hash_combine(seed, std::get<4>(key));
      return seed;
    }
  };

  template <>
  struct hash<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t>>
  {
    size_t operator()(const std::tuple<uint16_t, uint16_t, uint16_t, uint16_t,
                                       uint16_t>& key) const noexcept
    {
      size_t seed = 0;
      detail::hash_combine(seed, std::get<0>(key));
      detail::hash_combine(seed, std::get<1>(key));
      detail::hash_combine(seed, std::get<2>(key));
      detail::hash_combine(seed, std::get<3>(key));
      detail::hash_combine(seed, std::get<4>(key));
      return seed;
    }
  };

  template <>
  struct hash<std::tuple<int8_t, int8_t, int8_t, int8_t, int8_t>>
  {
    size_t operator()(const std::tuple<int8_t, int8_t, int8_t, int8_t,
                                       int8_t>& key) const noexcept
    {
      size_t seed = 0;
      detail::hash_combine(seed, std::get<0>(key));
      detail::hash_combine(seed, std::get<1>(key));
      detail::hash_combine(seed, std::get<2>(key));
      detail::hash_combine(seed, std::get<3>(key));
      detail::hash_combine(seed, std::get<4>(key));
      return seed;
    }
  };

  template <>
  struct hash<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t, uint8_t>>
  {
    size_t operator()(const std::tuple<uint8_t, uint8_t, uint8_t, uint8_t,
                                       uint8_t>& key) const noexcept
    {
      size_t seed = 0;
      detail::hash_combine(seed, std::get<0>(key));
      detail::hash_combine(seed, std::get<1>(key));
      detail::hash_combine(seed, std::get<2>(key));
      detail::hash_combine(seed, std::get<3>(key));
      detail::hash_combine(seed, std::get<4>(key));
      return seed;
    }
  };

  template <>
  struct hash<std::tuple<uint, uint, uint, uint, uint>>
  {
    size_t operator()(
        const std::tuple<uint, uint, uint, uint, uint>& key) const noexcept
    {
      size_t seed = 0;
      detail::hash_combine(seed, std::get<0>(key));
      detail::hash_combine(seed, std::get<1>(key));
      detail::hash_combine(seed, std::get<2>(key));
      detail::hash_combine(seed, std::get<3>(key));
      detail::hash_combine(seed, std::get<4>(key));
      return seed;
    }
  };
} // namespace std
*/
