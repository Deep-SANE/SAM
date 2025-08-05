#include <iostream>

#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  std::ostream& operator<<(std::ostream& s, const ad_gelfand& x)
  {
    return s << "ad[" << x.sps3[x.c] << "][" << x.sps4[x.r] << "]";
  }

  std::ostream& operator<<(std::ostream& s, const ad_phys& x)
  {
    auto [n, ll, jj, mmj, mmt] = x.getLabels();
    return s << "ad(" << n << "," << ll << "," << jj << "," << mmj << "," << mmt
             << ")";
  }

  std::ostream& operator<<(std::ostream& s, const at_gelfand& x)
  {
    if (x.tensor == true)
      return s << "at[" << x.sps3_conj[x.c] << "][" << x.sps4_conj[x.r] << "]";
    else
      return s << "a[" << x.sps3[x.c] << "][" << x.sps4[x.r] << "]";
  }

  std::ostream& operator<<(std::ostream& s, const at_phys& x)
  {
    auto [n, ll, jj, mmj, mmt] = x.getLabels();
    if (x.tensor)
      return s << "at(" << n << "," << ll << "," << jj << "," << mmj << ","
               << mmt << ")";
    else
      return s << "a(" << n << "," << ll << "," << jj << "," << mmj << ","
               << mmt << ")";
  }

  std::ostream& operator<<(std::ostream& s, const Tadat& x)
  {
    s << "[ (n=" << x.n << ") ";
    if (x.ads > 0)
    {
      s << static_cast<int>(x.ads) << "ads ";
      for (auto ad : x.prod_ads)
        s << ad << " ";
      if (x.ats > 0)
        s << "x ";
    }
    if (x.ats > 0)
    {
      s << static_cast<int>(x.ats) << "ats ";
      for (auto at : x.prod_ats)
        s << at << " ";
    }
    s << "] rho0max" << static_cast<int>(x.rho0max) << " eta0max"
      << static_cast<int>(x.eta0max) << x.T;
    return s;
  }

  void Tadat::writeBinary(std::ostream& s) const
  {
    s.write(reinterpret_cast<const char*>(&n), sizeof(n));
    s.write(reinterpret_cast<const char*>(&ads), sizeof(ads));
    s.write(reinterpret_cast<const char*>(&ats), sizeof(ats));
    s.write(reinterpret_cast<const char*>(&rho0max), sizeof(rho0max));
    s.write(reinterpret_cast<const char*>(&eta0max), sizeof(eta0max));

    for (const auto& ir : prod_ads)
      ir.writeBinary(s);

    for (const auto& ir : prod_ats)
      ir.writeBinary(s);

    T.writeBinary(s);
  }

  void Tadat::readBinary(std::istream& s)
  {
    s.read(reinterpret_cast<char*>(&n), sizeof(n));
    s.read(reinterpret_cast<char*>(&ads), sizeof(ads));
    s.read(reinterpret_cast<char*>(&ats), sizeof(ats));
    s.read(reinterpret_cast<char*>(&rho0max), sizeof(rho0max));
    s.read(reinterpret_cast<char*>(&eta0max), sizeof(eta0max));

    prod_ads.resize(std::max(0, static_cast<int>(ads) - 1));
    for (auto& ir : prod_ads)
      ir.readBinary(s);

    prod_ats.resize(std::max(0, static_cast<int>(ats) - 1));
    for (auto& ir : prod_ats)
      ir.readBinary(s);

    T.readBinary(s);
  }
} // namespace U3xU4
