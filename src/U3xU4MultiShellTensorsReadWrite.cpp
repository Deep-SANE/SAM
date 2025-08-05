#include <iostream>

#include <U3xU4MultiShellTensors.h>

namespace U3xU4
{
  std::ostream& operator<<(std::ostream& s, const OneBodyMultiShellTensor& x)
  {
    s << "[ T" << x.a << " x T" << x.b << " ]" << x.t << "kappaMax"
      << x.kappamax << "LL" << x.LL << "SS" << x.SS << "TT" << x.TT << "MMT"
      << x.MMT << "JJ" << x.JJ << "MMJ" << x.MMJ;
    for (auto coeff : x.coeffs)
      s << "\t" << coeff;
    return s;
  }

  std::ostream& operator<<(std::ostream& s, const TwoBodyMultiShellTensor& x)
  {
    s << x.single_shell_tensors[0];
    if (x.coupled_irreps.size() > 0)
      for (int i = 0; i < x.coupled_irreps.size(); i++)
        s << " x " << x.single_shell_tensors[i + 1] << " ] "
          << x.coupled_irreps[i];

    s << "rhoMax" << x.rhomax << "etaMax" << x.etamax << "kappaMax"
      << x.kappamax << "LL" << x.LL << "zetaMax" << x.zetamax << "SS" << x.SS
      << "TT" << x.TT << "MMT" << x.MMT << "JJ" << x.JJ << "MMJ" << x.MMJ;
    for (auto coeff : x.coeffs)
      s << "\t" << coeff;
    return s;
  }

  void TwoBodyMultiShellTensor::writeBinary(std::ostream& s) const
  {
    uint32_t n_tensors = single_shell_tensors.size();
    s.write(reinterpret_cast<const char*>(&n_tensors), sizeof(n_tensors));
    for (const auto& t : single_shell_tensors)
      t.writeBinary(s);

    for (const auto& ir : coupled_irreps)
      ir.writeBinary(s);

    s.write(reinterpret_cast<const char*>(&rhomax), sizeof(rhomax));
    s.write(reinterpret_cast<const char*>(&etamax), sizeof(etamax));
    s.write(reinterpret_cast<const char*>(&kappamax), sizeof(kappamax));
    s.write(reinterpret_cast<const char*>(&LL), sizeof(LL));
    s.write(reinterpret_cast<const char*>(&zetamax), sizeof(zetamax));
    s.write(reinterpret_cast<const char*>(&SS), sizeof(SS));
    s.write(reinterpret_cast<const char*>(&TT), sizeof(TT));
    s.write(reinterpret_cast<const char*>(&MMT), sizeof(MMT));
    s.write(reinterpret_cast<const char*>(&JJ), sizeof(JJ));
    s.write(reinterpret_cast<const char*>(&MMJ), sizeof(MMJ));

    s.write(reinterpret_cast<const char*>(coeffs.data()),
            coeffs.size() * sizeof(double));
  }

  void TwoBodyMultiShellTensor::readBinary(std::istream& s)
  {
    uint32_t n_tensors;
    s.read(reinterpret_cast<char*>(&n_tensors), sizeof(n_tensors));

    single_shell_tensors.resize(n_tensors);
    for (auto& t : single_shell_tensors)
      t.readBinary(s);

    uint32_t n_irreps = std::max(uint32_t(0), n_tensors - 1);
    coupled_irreps.resize(n_irreps);
    for (auto& ir : coupled_irreps)
      ir.readBinary(s);

    s.read(reinterpret_cast<char*>(&rhomax), sizeof(rhomax));
    s.read(reinterpret_cast<char*>(&etamax), sizeof(etamax));
    s.read(reinterpret_cast<char*>(&kappamax), sizeof(kappamax));
    s.read(reinterpret_cast<char*>(&LL), sizeof(LL));
    s.read(reinterpret_cast<char*>(&zetamax), sizeof(zetamax));
    s.read(reinterpret_cast<char*>(&SS), sizeof(SS));
    s.read(reinterpret_cast<char*>(&TT), sizeof(TT));
    s.read(reinterpret_cast<char*>(&MMT), sizeof(MMT));
    s.read(reinterpret_cast<char*>(&JJ), sizeof(JJ));
    s.read(reinterpret_cast<char*>(&MMJ), sizeof(MMJ));

    size_t n_coeffs = static_cast<size_t>(rhomax * etamax * kappamax * zetamax);
    coeffs.resize(n_coeffs);
    s.read(reinterpret_cast<char*>(coeffs.data()), n_coeffs * sizeof(double));
  }
} // namespace U3xU4
