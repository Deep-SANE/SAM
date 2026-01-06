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
    uint8_t shell_;    // shell number

    union
    {
      // basis part:
      uint16_t alphamax_; // U(N)->U(3) reduction multiplicity

      // tensor part:
      struct
      {
        uint8_t rhomax_; // rho(0)max
        uint8_t etamax_; // eta(0)max
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
          shell_(qcx_utils::itou8(shell)),
          alphamax_(qcx_utils::itou16(alphamax)),
          state_(State::Basis)
    { }

    // constructs a basis gamma element without alphamax:
    GammaElement(int shell, const u3::irrep& u3, const u4::irrep& u4)
        : U3xU4::irrep(u3, u4),
          shell_(qcx_utils::itou8(shell)),
          state_(State::Basis)
    { 
      int N = (shell + 1) * (shell + 2) / 2;
      untou3::UNtoU3<4> gen;
      gen.set_shell(shell);
      gen.reduceU4(u4, N);
      alphamax_ = qcx_utils::itou16(gen.multiplicity(u3));
    }

    // constructs a tensor gamma element:
    GammaElement(int shell, int rhomax, const u3::irrep& u3, int etamax,
                 const u4::irrep& u4)
        : U3xU4::irrep(u3, u4),
          shell_(qcx_utils::itou8(shell)),
          t_(qcx_utils::itou8(rhomax), qcx_utils::itou8(etamax)),
          state_(State::Tensor)
    { }

    // getters:

    int shell() const { return static_cast<int>(shell_); }

    int alphamax() const
    {
      assert(state_ == State::Basis);
      return static_cast<int>(alphamax_);
    }

    // total quanta:
    int quanta() const 
    { 
      assert(state_ == State::Basis);
      return u3().quanta(); 
    }

    // total particles:
    int particles() const 
    { 
      assert(state_ == State::Basis);
      return u4().quanta(); 
    }

    int rhomax() const
    {
      assert(state_ == State::Tensor);
      return static_cast<int>(t_.rhomax_);
    }

    int etamax() const
    {
      assert(state_ == State::Tensor);
      return static_cast<int>(t_.etamax_);
    }

    const U3xU4::irrep& u3u4() const { return *this; }

    // state:
    State state() const { return state_; }

    // comparison:
    auto operator==(const GammaElement& x) const
    {
      bool flag = false;
      if (u3u4() == x.u3u4() && shell_ == x.shell_ && state_ == x.state_)
      {
        if (state_ == State::Basis && alphamax() == x.alphamax()) 
          flag = true;
        else if (state_ == State::Tensor && rhomax() == x.rhomax() &&
                 etamax() == x.etamax()) 
          flag = true;
      }
      return flag;
    }

    // checks for vacuum/identity:

    bool vacuum() const 
    {
      assert(state_ == State::Basis);
      return U3xU4::irrep::vacuum();
    }

    bool identity() const
    {
      assert(state_ == State::Tensor);
      return U3xU4::irrep::vacuum();
    }

    // write to binary file
    void writeBinary(std::ostream& s) const
    {
      U3xU4::irrep::writeBinary(s);
      qcx_utils::writeBinary(s, shell_);

      if (state_ == State::Basis)
        qcx_utils::writeBinary(s, alphamax_);
      else if (state_ == State::Tensor)
      {
        qcx_utils::writeBinary(s, t_.rhomax_);
        qcx_utils::writeBinary(s, t_.etamax_);
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
        qcx_utils::readBinary(s, t_.rhomax_);
        qcx_utils::readBinary(s, t_.etamax_);
      }
      else
        throw std::runtime_error("Invalid state of U3xU4::GammaElement!");
    }
  };
  using GammaElements = std::vector<U3xU4::GammaElement>;

  inline std::ostream& operator<<(std::ostream& s, const GammaElement& ge)
  {
    if (ge.state() == U3xU4::GammaElement::State::Basis)
      return s << ge.alphamax() << ge.u3() << ge.u4();
    else if (ge.state() == U3xU4::GammaElement::State::Tensor)
      return s << ge.rhomax() << ge.u3() << ge.etamax() << ge.u4();
    else
      throw std::runtime_error("Invalid state of U3xU4::GammaElement!");
  }

  // class that represents a single omega element:
  class OmegaElement : public U3xU4::irrep
  {
    uint8_t rhomax_; // U(3) coupling multiplicity
    uint8_t etamax_; // U(4) coupling multiplicity

  public:
    OmegaElement() : rhomax_(1), etamax_(1) { }

    OmegaElement(int rhomax, const u3::irrep& u3, int etamax,
                 const u4::irrep& u4)
        : U3xU4::irrep(u3, u4),
          rhomax_(qcx_utils::itou8(rhomax)),
          etamax_(qcx_utils::itou8(etamax))
    { }

    // comparison:
    auto operator<=>(const OmegaElement&) const = default;

    // getters:
    int rhomax() const { return static_cast<int>(rhomax_); }
    int etamax() const { return static_cast<int>(etamax_); }

    const U3xU4::irrep& u3u4() const { return *this; }

    // write to binary file
    void writeBinary(std::ostream& s) const
    {
      U3xU4::irrep::writeBinary(s);
      qcx_utils::writeBinary(s, rhomax_);
      qcx_utils::writeBinary(s, etamax_);
    }

    // read from binary file
    void readBinary(std::istream& s)
    {
      U3xU4::irrep::readBinary(s);
      qcx_utils::readBinary(s, rhomax_);
      qcx_utils::readBinary(s, etamax_);
    }
  };
  using OmegaElements = std::vector<U3xU4::OmegaElement>;

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
      return fmt::format_to(ctx.out(), "{}:{}{}{}", ge.shell(), ge.alphamax(),
                            ge.u3(), ge.u4());
    else if (ge.state() == U3xU4::GammaElement::State::Tensor)
      return fmt::format_to(ctx.out(), "{}:{}{}{}{}", ge.shell(), ge.rhomax(),
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
    return fmt::format_to(ctx.out(), "{}{}{}{}", oe.rhomax(), oe.u3(),
                          oe.etamax(), oe.u4());
  }
};

#endif
