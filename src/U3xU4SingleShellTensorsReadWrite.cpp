#include <iostream>

#include <readwrite.h>

#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  std::ostream& operator<<(std::ostream& s, const SingleShellTensor& x)
  {
    s << "[ (n=" << x.shell() << ") ";
    if (x.ads_ > 0)
    {
      s << static_cast<int>(x.ads_) << "ads ";
      for (const auto& ad : x.Omega_ads_) s << ad << " ";
      if (x.ats_ > 0) s << "x ";
    }
    if (x.ats_ > 0)
    {
      s << static_cast<int>(x.ats_) << "ats ";
      for (const auto& at : x.Omega_ats_) s << at << " ";
    }
    s << "] rho0max" << x.rho0max() << " eta0max" << x.eta0max() << x.T();
    return s;
  }

  void SingleShellTensor::writeBinary(std::ostream& s) const
  {
    U3xU4::GammaElement::writeBinary(s); // T, shell, rho0max, eta0max
    utils::writeBinary(s, ads_);
    utils::writeBinary(s, ats_);

    for (const auto& ir : Omega_ads_) ir.writeBinary(s);
    for (const auto& ir : Omega_ats_) ir.writeBinary(s);
  }

  void SingleShellTensor::readBinary(std::istream& s)
  {
    U3xU4::GammaElement::readBinary(s); // T, shell, rho0max, eta0max
    utils::readBinary(s, ads_);
    utils::readBinary(s, ats_);

    Omega_ads_.resize(std::max(0, static_cast<int>(ads_) - 1));
    for (auto& ir : Omega_ads_) ir.readBinary(s);

    Omega_ats_.resize(std::max(0, static_cast<int>(ats_) - 1));
    for (auto& ir : Omega_ats_) ir.readBinary(s);

    using std::to_string;

    std::string shell = "n" + to_string(U3xU4::GammaElement::shell());

    std::string ads = "_" + to_string(num_ads()) + "ads";
    for (const auto& Omega : Omega_ads_)
      ads += "_" + to_string(Omega.lam()) + "-" + to_string(Omega.mu()) + "-" +
               to_string(Omega.P1()) + "-" + to_string(Omega.P2()) + "-" +
               to_string(Omega.P3());

    std::string ats = "_" + to_string(num_ats()) + "ats";
    for (const auto& Omega : Omega_ats_)
      ats += "_" + to_string(Omega.lam()) + "-" + to_string(Omega.mu()) + "-" +
               to_string(Omega.P1()) + "-" + to_string(Omega.P2()) + "-" +
               to_string(Omega.P3());

    std::string irr = "_" + to_string(rho0max()) + "-" + to_string(lam()) +
                      "-" + to_string(mu()) + "-" + to_string(eta0max()) +
                      "-" + to_string(P1()) + "-" + to_string(P2()) + "-" +
                      to_string(P3());

    filename_ = shell + ads + ats + irr + ".bin";
  }

  std::ostream& operator<<(std::ostream& s, const SingleShellTensor_gelfand& x)
  {
    s << "[ (n=" << x.shell() << ") ";
    if (x.num_ads() > 0)
    {
      s << x.num_ads() << "ads ";
      for (const auto& ad : x.Omega_ads()) s << ad << " ";
      if (x.num_ats() > 0) s << "x ";
    }
    if (x.num_ats() > 0)
    {
      s << static_cast<int>(x.num_ats()) << "ats ";
      for (const auto& at : x.Omega_ats()) s << at << " ";
    }
    s << "] rho0max" << x.rho0max() << " eta0max" << x.eta0max()
      << x.T() << x.g3_ << " ; " << x.g4_;
    return s;
  }

  std::ostream& operator<<(std::ostream& s, const SingleShellTensor_phys& x)
  {
    s << "[ (n=" << x.shell() << ") ";
    if (x.num_ads() > 0)
    {
      s << x.num_ads() << "ads ";
      for (const auto& ad : x.Omega_ads()) s << ad << " ";
      if (x.num_ats() > 0) s << "x ";
    }
    if (x.num_ats() > 0)
    {
      s << x.num_ats() << "ats ";
      for (const auto& at : x.Omega_ats()) s << at << " ";
    }
    s << "] rho0max" << x.rho0max() << " eta0max" << x.eta0max()
      << x.T() << " kappamax" << x.kappamax() << " LL" << x.LL()
      << " zetamax" << x.zetamax() << " SS" << x.SS()
      << " TT" << x.TT() << " MMT" << x.MMT() 
      << " JJ" << x.JJ() << " MMJ" << x.MMJ();
    return s;
  }
} // namespace U3xU4
