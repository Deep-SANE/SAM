#pragma once

#include <cstdint>
#include <ostream>
#include <stdexcept>

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

    union
    {
      // basis part:
      uint16_t alphamax_; // U(N)->U(3) reduction multiplicity

      // tensor part:
      struct
      {
        uint8_t rhomax; // rho(0)max
        uint8_t etamax; // eta(0)max
      } t_;
    };

  public:
    enum class State : uint8_t { Basis, Tensor, None };

  private:
    State state_;

  public:
    // constructs a generic vacuum gamma element:
    //GammaElement(int shell) : shell_(shell), state_(State::None) { }

    // constructs a basis gamma element:
    GammaElement(int shell, int alphamax, const u3::irrep& u3,
                 const u4::irrep& u4)
        : U3xU4::irrep(u3, u4),
          shell_(shell),
          alphamax_(qcx_utils::itou16(alphamax)),
          state_(State::Basis)
    { }

    // constructs a tensor gamma element:
    GammaElement(int shell, int rhomax, const u3::irrep& u3, int etamax,
                 const u4::irrep& u4)
        : U3xU4::irrep(u3, u4),
          shell_(shell),
          t_(qcx_utils::itou8(rhomax), qcx_utils::itou8(etamax)),
          state_(State::Tensor)
    { }

    // getters:

    int shell() const { return shell_; }

    int alphamax() const
    {
      assert(state_ == State::Basis);
      return alphamax_;
    }

    int rhomax() const
    {
      assert(state_ == State::Tensor);
      return t_.rhomax;
    }

    int etamax() const
    {
      assert(state_ == State::Tensor);
      return t_.etamax;
    }

    // state:
    State state() const { return state_; }
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

    // print
    friend std::ostream& operator<<(std::ostream& s, const OmegaElement& o)
    {
      return s << o.rhomax_ << o.u3() << o.etamax_ << o.u4();
    }
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
    /*
    if (ge.state() == U3xU4::GammaElement::State::None)
      return format_to(ctx.out(), "{}:{}{}", ge.shell(), ge.u4(), ge.su3());
    */
    if (ge.state() == U3xU4::GammaElement::State::Basis)
      return format_to(ctx.out(), "{}:{}{}{}", ge.shell(), ge.u4(),
                       ge.alphamax(), ge.u3());
    else if (ge.state() == U3xU4::GammaElement::State::Tensor)
      return format_to(ctx.out(), "{}:{}{}{}{}", ge.shell(), ge.etamax(),
                       ge.u4(), ge.rhomax(), ge.u3());
    else
      throw std::runtime_error("Invalid state of U3xU4::GammaElement!");
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
                     oe.u3());
  }
};

#endif
