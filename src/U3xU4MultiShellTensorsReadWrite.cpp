#include <iostream>

#include <readwrite.h>

#include <U3xU4MultiShellTensors.h>

namespace U3xU4
{
  std::ostream& operator<<(std::ostream& s, const OneBodyMultiShellTensor& x)
  {
    s << "[ T" << x.a_ << " x T" << x.b_ << " ]" << x.t_ << "kappamax"
      << x.kappamax() << " LL" << x.LL() << " SS" << x.SS() << " TT" << x.TT() 
      << " MMT" << x.MMT() << " JJ" << x.JJ() << " MMJ" << x.MMJ();

    for (auto coeff : x.coeffs_) s << "\t" << coeff;
    
    return s;
  }

  void OneBodyMultiShellTensor::writeBinary(std::ostream& s) const
  {
    a_.writeBinary(s);
    b_.writeBinary(s);
    t_.writeBinary(s);
    utils::writeBinary(s, kappamax_);
    utils::writeBinary(s, LL_);
    utils::writeBinary(s, SS_);
    utils::writeBinary(s, TT_);
    utils::writeBinary(s, MMT_);
    utils::writeBinary(s, JJ_);
    utils::writeBinary(s, MMJ_);
    utils::writeBinary(s, coeffs_);
  }

  void OneBodyMultiShellTensor::readBinary(std::istream& s)
  {
    a_.readBinary(s);
    b_.readBinary(s);
    t_.readBinary(s);
    utils::readBinary(s, kappamax_);
    utils::readBinary(s, LL_);
    utils::readBinary(s, SS_);
    utils::readBinary(s, TT_);
    utils::readBinary(s, MMT_);
    utils::readBinary(s, JJ_);
    utils::readBinary(s, MMJ_);
    coeffs_.resize(static_cast<size_t>(kappamax_));
    utils::readBinary(s, coeffs_);
  }

  std::ostream& operator<<(std::ostream& s, const TwoBodyMultiShellTensor& x)
  {
    s << x.Gammas_[0];
    if (x.Omegas_.size() > 0)
      for (int i = 0; i < x.Omegas_.size(); i++)
        s << " x " << x.Gammas_[i + 1] << " ] " << x.Omegas_[i];

    s << " rhomax" << x.rhomax() << " etamax" << x.etamax() << " kappaMax"
      << x.kappamax() << " LL" << x.LL() << " zetamax" << x.zetamax() 
      << " SS" << x.SS() << " TT" << x.TT() << " MMT" << x.MMT() << " JJ" 
      << x.JJ() << " MMJ" << x.MMJ();
    
    for (auto coeff : x.coeffs_) s << "\t" << coeff;
    
    return s;
  }

  void TwoBodyMultiShellTensor::writeBinary(std::ostream& s) const
  {
    size_t n_tensors = Gammas_.size();
    utils::writeBinary(s, n_tensors);

    for (const auto& t : Gammas_) t.writeBinary(s);

    for (const auto& ir : Omegas_) ir.writeBinary(s);

    utils::writeBinary(s, rhomax_);
    utils::writeBinary(s, etamax_);
    utils::writeBinary(s, kappamax_);
    utils::writeBinary(s, zetamax_);
    utils::writeBinary(s, LL_);
    utils::writeBinary(s, SS_);
    utils::writeBinary(s, TT_);
    utils::writeBinary(s, MMT_);
    utils::writeBinary(s, JJ_);
    utils::writeBinary(s, MMJ_);
    utils::writeBinary(s, coeffs_);
  }

  void TwoBodyMultiShellTensor::readBinary(std::istream& s)
  {
    size_t n_tensors;
    utils::readBinary(s, n_tensors);

    Gammas_.resize(n_tensors);
    for (auto& t : Gammas_) t.readBinary(s);

    assert(n_tensors > 0);
    Omegas_.resize(n_tensors - 1);
    for (auto& ir : Omegas_) ir.readBinary(s);

    utils::readBinary(s, rhomax_);
    utils::readBinary(s, etamax_);
    utils::readBinary(s, kappamax_);
    utils::readBinary(s, zetamax_);
    utils::readBinary(s, LL_);
    utils::readBinary(s, SS_);
    utils::readBinary(s, TT_);
    utils::readBinary(s, MMT_);
    utils::readBinary(s, JJ_);
    utils::readBinary(s, MMJ_);

    size_t n_coeffs = static_cast<size_t>(rhomax_ * etamax_ * kappamax_ * zetamax_);
    coeffs_.resize(n_coeffs);
    utils::readBinary(s, coeffs_);
  }

  std::ostream& operator<<(std::ostream& s, const MultiShellTensor& x)
  {
    s << x.Gammas_[0];
    if (!x.Omegas_.empty())
      for (size_t i = 0; i < x.Omegas_.size(); i++)
        s << " x " << x.Gammas_[i + 1] << " ] " << x.Omegas_[i];

    return s;
  }

  void MultiShellTensor::writeBinary(std::ostream& s) const
  {
    size_t n_Gammas = Gammas_.size();
    utils::writeBinary(s, n_Gammas);
    for (const auto& Gamma : Gammas_) Gamma.writeBinary(s);

    size_t n_Omegas = Omegas_.size();
    utils::writeBinary(s, n_Omegas);
    for (const auto& Omega : Omegas_) Omega.writeBinary(s);
  }

  void MultiShellTensor::readBinary(std::istream& s)
  {
    size_t n_Gammas;
    utils::readBinary(s, n_Gammas);
    Gammas_.resize(n_Gammas);
    for (auto& Gamma : Gammas_) Gamma.readBinary(s);

    size_t n_Omegas;
    utils::readBinary(s, n_Omegas);
    Omegas_.resize(n_Omegas);
    for (auto& Omega : Omegas_) Omega.readBinary(s);
  }

  std::ostream& operator<<(std::ostream& s, const MultiShellTensor_phys& x)
  {
    s << x.T();

    for (const auto& [kappamax, LL, zetamax, SS, TT, strengths] : x.LLSSTTstrengths_)
    {
      s << "\n kappamax:" << static_cast<int>(kappamax)
        << " LL:" << static_cast<int>(LL)
        << " zetamax:" << static_cast<int>(zetamax)
        << " SS:" << static_cast<int>(SS)
        << " TT:" << static_cast<int>(TT)
        << " strengths:";
      for (const auto& c : strengths) s << c << " ";
    }
    s << "\n MMT:" << x.MMT() << " JJ:" << x.JJ() << " MMJ:" << x.MMJ();

    return s;
  }

  void MultiShellTensor_phys::writeBinary(std::ostream& s) const
  {
    MultiShellTensor::writeBinary(s);
    utils::writeBinary(s, MMT_);
    utils::writeBinary(s, JJ_);
    utils::writeBinary(s, MMJ_);

    size_t cnt = LLSSTTstrengths_.size();
    utils::writeBinary(s, cnt);

    for (const auto& [kappamax, LL, zetamax, SS, TT, strengths] : LLSSTTstrengths_)
    {
      utils::writeBinary(s, kappamax);
      utils::writeBinary(s, LL);
      utils::writeBinary(s, zetamax);
      utils::writeBinary(s, SS);
      utils::writeBinary(s, TT);
      size_t cnt_strength = strengths.size();
      utils::writeBinary(s, cnt_strength);
      for (const auto& c : strengths) utils::writeBinary(s, c);
    }
  }

  void MultiShellTensor_phys::readBinary(std::istream& s)
  {
    MultiShellTensor::readBinary(s);
    utils::readBinary(s, MMT_);
    utils::readBinary(s, JJ_);
    utils::readBinary(s, MMJ_);

    size_t cnt;
    utils::readBinary(s, cnt);
    LLSSTTstrengths_.resize(cnt);
    for (auto& [kappamax, LL, zetamax, SS, TT, strengths] : LLSSTTstrengths_)
    {
      utils::readBinary(s, kappamax);
      utils::readBinary(s, LL);
      utils::readBinary(s, zetamax);
      utils::readBinary(s, SS);
      utils::readBinary(s, TT);
      size_t cnt_strength;
      utils::readBinary(s, cnt_strength);
      strengths.resize(cnt_strength);
      for (auto& c : strengths) utils::readBinary(s, c);
    }
  }

  std::ostream& operator<<(std::ostream& s, const MultiShellTensor_ShareGammas_phys& x)
  {
    for (const auto& G : x.Gammas_) s << G << "\t";
    
    for (const auto& [Omegas, LLSSTTstrengths] : x.Omegas_LLSSTTstrengths_)
    {
      s << "\n";
      for (const auto& O : Omegas) s << O << "\t";

      for (const auto& [kmax, LL, zmax, SS, TT, strengths] : LLSSTTstrengths)
      {
        s << "\n kappamax:" << static_cast<int>(kmax)
          << " LL:" << static_cast<int>(LL)
          << " zetamax:" << static_cast<int>(zmax)
          << " SS:" << static_cast<int>(SS)
          << " TT:" << static_cast<int>(TT)
          << " strengths:";
        for (const auto& c : *strengths) s << c << " ";
      }
    }
    s << "\n MMT:" << x.MMT() << " JJ:" << x.JJ() << " MMJ:" << x.MMJ();

    return s;
  }
} // namespace U3xU4
