#pragma once

#include <cstdint>

#include <qcx_convert.h>
#include <U3irrep.h>
#include <U3xU4irrep.h>
#include <U4irrep.h>

#ifdef HAVE_FMT
#include <fmt/format.h>
#endif

namespace U3xU4
{
  // class that represents a single gamma element:
  class GammaElement : public U3xU4::irrep
  {
    uint16_t shell_;    // shell number
    uint16_t alphamax_; // U(N)->U(3) reduction multiplicity

  public:
    GammaElement(int shell) : shell_(shell), alphamax_(1) { }

    GammaElement(int shell, int alphamax, const u3::irrep& u3,
                 const u4::irrep& u4)
        : U3xU4::irrep(u3, u4),
          shell_(shell),
          alphamax_(qcx_utils::itou16(alphamax))
    { }

    // getters:

    int shell() const { return shell_; }
    int alphamax() const { return alphamax_; }
  };

  // class that represents a single omega element:
  class OmegaElement : public U3xU4::irrep
  {
    uint16_t rhomax_; // U(3) coupling multiplicity
    uint16_t etamax_; // U(4) coupling multiplicity

  public:
    OmegaElement() : rhomax_(1), etamax_(1) { }

    OmegaElement(int rhomax, const u3::irrep& u3, int etamax,
                 const u4::irrep& u4)
        : U3xU4::irrep(u3, u4),
          rhomax_(qcx_utils::itou16(rhomax)),
          etamax_(qcx_utils::itou16(etamax))
    { }

    // getters:
    int rhomax() const { return rhomax_; }
    int etamax() const { return etamax_; }
  };

} // namespace U3xU4

#ifdef HAVE_FMT

// formatter for GammaElement:

template <>
class fmt::formatter<U3xU4::GammaElement>
{
public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename Context>
  constexpr auto format(const U3xU4::GammaElement& ge, Context& ctx) const
  {
    return format_to(ctx.out(), "{}{}{}", ge.u4(), ge.alphamax(), ge.su3());
  }
};

// formatter for OmegaElement:

template <>
class fmt::formatter<U3xU4::OmegaElement>
{

public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename Context>
  constexpr auto format(const U3xU4::OmegaElement& oe, Context& ctx) const
  {
    return format_to(ctx.out(), "{}{}{}{}", oe.etamax(), oe.u4(), oe.rhomax(),
                     oe.su3());
  }
};

#endif
