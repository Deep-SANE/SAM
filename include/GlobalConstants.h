#pragma once

#include <cmath>
#include <numbers>

///////////////// NUMERICAL CONSTANTS /////////////////

constexpr double SQRT2 = std::numbers::sqrt2;
constexpr double SQRT3 = std::numbers::sqrt3;
constexpr double PI = std::numbers::pi;

///////////////// PHYSICAL CONSTANTS ////////////////

// constexpr double mass_p = ; proton mass
// constexpr double mass_n = ; neutron mass

// g factors for proton and neutron quoted from
// K.S. Krane, Introductory Nuclear Physics (3rd edition), pg. 215
constexpr double gl_p = 1;
constexpr double gs_p = 5.5856912;
constexpr double gl_n = 0;
constexpr double gs_n = -3.8260837;
