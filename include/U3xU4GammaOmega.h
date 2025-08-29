#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

#include <qcx_convert.h>
#include <qcx_readwrite.h>

#include <UNtoU3.h>

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
    enum class State : uint8_t { Basis, Tensor };

  private:
    State state_;

  public:
    // constructs a basis gamma element:
    GammaElement(int shell, int alphamax, const u3::irrep& u3,
                 const u4::irrep& u4)
        : U3xU4::irrep(u3, u4),
          shell_(shell),
          alphamax_(qcx_utils::itou16(alphamax)),
          state_(State::Basis)
    { }

    // constructs a basis gamma element without alphamax:
    GammaElement(int shell, const u3::irrep& u3, const u4::irrep& u4)
        : U3xU4::irrep(u3, u4),
          shell_(shell),
          state_(State::Basis)
    { 
      int N = (shell + 1) * (shell + 2) / 2;
      untou3::UNtoU3<4> gen;
      gen.set_shell(shell);
      gen.reduceU4(u4, N);
      alphamax_ = gen.multiplicity(u3);
    }

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

    // write to binary file
    void writeBinary(std::ostream& s) const
    {
      U3xU4::irrep::writeBinary(s);
      qcx_utils::writeBinary(s, shell_);

      if (state_ == State::Basis)
        qcx_utils::writeBinary(s, alphamax_);
      else if (state_ == State::Tensor)
      {
        qcx_utils::writeBinary(s, t_.rhomax);
        qcx_utils::writeBinary(s, t_.etamax);
      }
      else
        throw std::runtime_error("Invalid state of U3xU4::GammaElement!");
    }

    // read from binary file
    void readBinary(std::istream& s)
    {
      U3xU4::irrep::readBinary(s);
      qcx_utils::readBinary(s, shell_);

      if (state_ == State::Basis)
        qcx_utils::readBinary(s, alphamax_);
      else if (state_ == State::Tensor)
      {
        qcx_utils::readBinary(s, t_.rhomax);
        qcx_utils::readBinary(s, t_.etamax);
      }
      else
        throw std::runtime_error("Invalid state of U3xU4::GammaElement!");
    }
  };

  inline std::ostream& operator<<(std::ostream& s, const GammaElement& ge)
  {
    if (ge.state() == U3xU4::GammaElement::State::Basis)
      return s << ge.alphamax() << ge.u3() << ge.u3();
    else if (ge.state() == U3xU4::GammaElement::State::Tensor)
      return s << ge.rhomax() << ge.u3() << ge.etamax() << ge.u4();
    else
      throw std::runtime_error("Invalid state of U3xU4::GammaElement!");
  }

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

  inline std::ostream& operator<<(std::ostream& s, const OmegaElement& oe)
  {
    return s << oe.rhomax() << oe.u3() << oe.etamax() << oe.u4();
  }

  // helper function to extract shells from a set of gamma elements:
  template <typename R> // R - a range of GammaElements
  inline std::vector<int> extract_shells_from_gamma(const R& gamma)
  {
    std::vector<int> shells;
    shells.reserve(gamma.size());

    for (const auto& g : gamma)
      shells.push_back(g.shell());

    return shells;
  }

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
    if (ge.state() == U3xU4::GammaElement::State::Basis)
      return format_to(ctx.out(), "{}:{}{}{}", ge.shell(), ge.alphamax(),
                       ge.u3(), ge.u4());
    else if (ge.state() == U3xU4::GammaElement::State::Tensor)
      return format_to(ctx.out(), "{}:{}{}{}{}", ge.shell(), ge.rhomax(),
                       ge.u3(), ge.etamax(), ge.u4());
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
    return format_to(ctx.out(), "{}{}{}{}", oe.rhomax(), oe.u3(), oe.etamax(),
                     oe.u4());
  }
};

#endif
