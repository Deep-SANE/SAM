#pragma once

#include <cmath>
#include <numbers>

///////////////// NUMERICAL CONSTANTS /////////////////

constexpr double SQRT2 = std::numbers::sqrt2;
constexpr double SQRT3 = std::numbers::sqrt3;
constexpr double PI = std::numbers::pi;

///////////////// PHYSICAL CONSTANTS ////////////////

// proton and neutron masses quoted from NIST database
// https://physics.nist.gov/cuu/Constants/index.html
constexpr double mass_p = 938.272; // proton mass in MeV
constexpr double mass_n = 939.565; // neutron mass in MeV
constexpr double hbar = 197.327; // reduced Planck constant in MeV.fm

// g factors for proton and neutron quoted from
// K.S. Krane, Introductory Nuclear Physics (3rd edition), pg. 215
constexpr double gl_p = 1;
constexpr double gs_p = 5.5856912;
constexpr double gl_n = 0;
constexpr double gs_n = -3.8260837;

// the nucleon axial-vector coupling constant
constexpr double gAx = 1.27;
