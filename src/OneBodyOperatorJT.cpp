#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>

#include <qcx_convert.h>
#include <qcx_readwrite.h>
#include <so3Coeffs/SO3_CGC.h>

#include <Databases.h>
#include <GlobalConstants.h>
#include <HOutilities.h>
#include <OneBodyOperatorJT.h>

namespace JT
{
  OneBodyOperator::OneBodyOperator(const std::string& name, int hw, int Nmax, 
            int valenceShell, int maxShell, const HO::Table_nljmjmt& labels)
    : labels_(labels)
  {
    if (maxShell > 14)
      throw std::out_of_range(
          "Sorry, we only have maxShell up to 14 at the moment");

    hw_ = (hw >= 0) ? qcx_utils::ito8(hw) : 20;
    Nmax_ = (Nmax >= 0) ? qcx_utils::ito8(Nmax) : 14;
    valenceShell_ = (valenceShell > 0) ? qcx_utils::ito8(valenceShell) : 0;
    maxShell_ = (maxShell >= 0) ? qcx_utils::ito8(maxShell) : 14;

    name_ = name;
    JJ_ = 0;
    MMJ_ = 0;
    TT_ = 0;
    MMT_ = 0;
    parity_ = 1;

    if (known_scalar_ops.count(name_)) { }
    else if (name_ == "E2") { JJ_ = 4; }
    else if (name_ == "M1") { JJ_ = 2; }
    else if (name_ == "FermiPlus")
    {
      TT_ = 2;
      MMT_ = 2;
    }
    else if (name_ == "FermiMinus")
    {
      TT_ = 2;
      MMT_ = -2;
    }
    else if (name_ == "GamowTellerPlus")
    {
      JJ_ = 2;
      TT_ = 2;
      MMT_ = 2;
    }
    else if (name_ == "GamowTellerMinus")
    {
      JJ_ = 2;
      TT_ = 2;
      MMT_ = -2;
    }
    else
    {
      throw std::invalid_argument(
        "Name is incorrect or operator is not implemented");
    }
  }

  // Check the type of this operator
  void OneBodyOperator::is() const
  {
    if (name_ == "NumberOp")
      std::cout << "Number operator N" << std::endl;
    else if (name_ == "L.L")
      std::cout << "One-body part of L.L" << std::endl;
    else if (name_ == "L.S")
      std::cout << "One-body part of L.S" << std::endl;
    else if (name_ == "CoMNumberOp")
      std::cout << "One-body part of the CoM number operator without 1/A"
                << std::endl;
    else if (name_ == "Kinetic")
      std::cout
          << "One-body part of the relative kinetic energy without (A-1)/A"
          << std::endl;
    else if (name_ == "R.R")
      std::cout << "One-body part of the intrinsic mass radius squared without "
                   "(A-1)/A"
                << std::endl;
    else if (name_ == "E2")
      std::cout << "E2 transition operator without (HO strength)^2"
                << std::endl;
    else if (name_ == "M1")
      std::cout << "M1 transition operator without sqrt(3/4pi)" << std::endl;
    else if (name_ == "FermiPlus")
      std::cout << "Fermi for beta+" << std::endl;
    else if (name_ == "FermiMinus")
      std::cout << "Fermi for beta-" << std::endl;
    else if (name_ == "GamowTellerPlus")
      std::cout << "Gamow-Teller for beta+" << std::endl;
    else if (name_ == "GamowTellerMinus")
      std::cout << "Gamow-Teller for beta-" << std::endl;
    else
      throw std::invalid_argument("This operator has not been implemented");
  }

  std::ostream& operator<<(std::ostream& s, const OneBodyOperator& x)
  {
    if (x.name_ == "NumberOp")
      s << "Number operator N";
    else if (x.name_ == "L.L")
      s << "One-body part of L.L";
    else if (x.name_ == "L.S")
      s << "One-body part of L.S";
    else if (x.name_ == "CoMNumberOp")
      s << "One-body part of the CoM number operator without 1/A";
    else if (x.name_ == "Kinetic")
      s << "One-body part of the relative kinetic energy without (A-1)/A";
    else if (x.name_ == "R.R")
      s << "One-body part of the intrinsic mass radius squared without (A-1)/A";
    else if (x.name_ == "E2")
      s << "E2 transition operator without (HO strength)^2";
    else if (x.name_ == "M1")
      s << "M1 transition operator without sqrt(3/4pi)";
    else if (x.name_ == "FermiPlus")
      s << "Fermi for beta+";
    else if (x.name_ == "FermiMinus")
      s << "Fermi for beta-";
    else if (x.name_ == "GamowTellerPlus")
      s << "Gamow-Teller for beta+";
    else if (x.name_ == "GamowTellerMinus")
      s << "Gamow-Teller for beta-";
    else
      throw std::invalid_argument("This operator has not been implemented");

    return s;
  }

  // ME: < na lla jja mmja mmta | O | nb llb jjb mmjb mmtb >
  double OneBodyOperator::sphericalME(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb,
                                      sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& racah = dbs.racah;

    double x = 0.0;

    if (name_ == "NumberOp") // N operator
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = nb;
    }
    else if (name_ == "L.L") // One-body part of LL
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = llb * (llb + 2.0) / 4.0;
    }
    else if (name_ == "L.S") // One-body-part of LS
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb && mmta == mmtb)
        x = (jjb * (jjb + 2.0) - llb * (llb + 2.0) - 3.0) / 8.0;
    }
    else if (name_ == "CoMNumberOp") // One-body part of CoM number operator
    {
      if (na == nb && jja == jjb && mmja == mmjb && mmta == mmtb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));

            x += phase * factor *
                 HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                 HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
      }
    }
    else if (name_ == "Kinetic") // One-body part of the relative kinetic energy
                                // without (A-1)/A
    {
      if (jja == jjb && mmja == mmjb && mmta == mmtb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
            else if (na == nb - 2)
              x -= phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
            else if (na == nb + 2)
              x -= phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
      }
      x = (static_cast<int>(hw_) * x / 4.0);
    }
    else if (name_ == "R.R") // One-body part of the intrinsic mass radius
                            // squared without (A-1)/A
    {
      if (jja == jjb && mmja == mmjb && mmta == mmtb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
            else if (na == nb - 2)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
            else if (na == nb + 2)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
      }
    }
    else if (name_ == "E2") // E2 transition operator without (HO strength)^2
    {
      if (mmta == mmtb && mmtb == -1)
      {
        double phase = ((lla - llb + jjb - jja) / 2 % 2 == 0) ? 1.0 : -1.0,
               factor = cgcs.get({jjb, 4, jja, mmjb, 0, mmja}) *
                        racah.get({llb, 1, 4, jja, jjb, lla});
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          double factor = racah.get({llb, 2, lla, 2, llc, 4});
          if (na == nb)
            x += factor * HO::bosonCreatorRME(na, lla, nc, llc) *
                 HO::bosonAnnihilatorRME(nc, llc, nb, llb);
          else if (na == nb - 2)
            x += factor * HO::bosonAnnihilatorRME(na, lla, nc, llc) *
                 HO::bosonAnnihilatorRME(nc, llc, nb, llb);
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          double factor = racah.get({llb, 2, lla, 2, llc, 4});
          if (na == nb)
            x += factor * HO::bosonAnnihilatorRME(na, lla, nc, llc) *
                 HO::bosonCreatorRME(nc, llc, nb, llb);
          else if (na == nb + 2)
            x += factor * HO::bosonCreatorRME(na, lla, nc, llc) *
                 HO::bosonCreatorRME(nc, llc, nb, llb);
        }
      }
      x *= std::sqrt(1.5);
    }
    else if (name_ == "M1") // M1 transition operator without sqrt(3/4pi)
    {
      if (na == nb && lla == llb && mmta == mmtb)
      {
        double gl = (mmta == 1) ? gl_n : gl_p; 
        double gs = (mmta == 1) ? gs_n : gs_p;

        double phase = ((jjb - jja) / 2 % 2 == 0) ? 1.0 : -1.0;
        x = cgcs.get({jjb, 2, jja, mmjb, 0, mmja}) *
            (gl * phase * racah.get({llb, 1, 2, jja, jjb, llb}) *
                 std::sqrt(llb * (llb + 2.0) / 4.0) +
             gs * racah.get({1, llb, 2, jja, jjb, 1}) * std::sqrt(0.75));
      }
    }
    else if (name_ == "FermiPlus") // Fermi for beta+
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb &&
          mmta == mmtb + 2)
        x = std::sqrt(0.75 - mmtb * (mmtb - 2.0) / 4.0);
    }
    else if (name_ == "FermiMinus") // Fermi for beta-
    {
      if (na == nb && lla == llb && jja == jjb && mmja == mmjb &&
          mmta == mmtb - 2)
        x = std::sqrt(0.75 - mmtb * (mmtb + 2.0) / 4.0);
    }
    else
      throw std::invalid_argument("ME has not been implemented");

    return x;
  }

  // RME: < na lla jja || O || nb llb jjb >
  // = ME / ( < jb mjb ; J MJ | ja mja > * < 1/2 mtb ; T MT | 1/2 mta > )
  double OneBodyOperator::sphericalRME(int na, int lla, int jja, int nb,
                                       int llb, int jjb,
                                       sam::Databases& dbs) const
  { 
    double x = 0.0;

    if (name_ == "NumberOp") // N operator
    {
      if (na == nb && lla == llb && jja == jjb)
        x = nb;
    }
    else if (name_ == "L.L") // One-body part of LL
    {
      if (na == nb && lla == llb && jja == jjb)
        x = llb * (llb + 2.0) / 4.0;
    }
    else if (name_ == "L.S") // One-body-part of LS
    {
      if (na == nb && lla == llb && jja == jjb)
        x = (jjb * (jjb + 2.0) - llb * (llb + 2.0) - 3.0) / 8.0;
    }
    else if (name_ == "CoMNumberOp") // One-body part of CoM number operator
    {
      if (na == nb && jja == jjb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));

            x += phase * factor *
                 HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                 HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
      }
    }
    else if (name_ == "Kinetic") // One-body part of the relative kinetic energy
                                // without (A-1)/A
    {
      if (jja == jjb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
            else if (na == nb - 2)
              x -= phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
            else if (na == nb + 2)
              x -= phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
      }
      x = (static_cast<int>(hw_) * x / 4.0);
    }
    else if (name_ == "R.R") // One-body part of the intrinsic mass radius
                            // squared without (A-1)/A
    {
      if (jja == jjb)
      {
        int nc = nb - 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
            else if (na == nb - 2)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonAnnihilatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
        nc = nb + 1, llc_min = 2 * (nc % 2);
        for (int llc = 2 * nc; llc >= llc_min; llc -= 4)
        {
          for (int jjc = llc + 1; jjc >= abs(llc - 1); jjc -= 2)
          {
            double phase = ((jjc - jjb) / 2 % 2 == 0) ? 1.0 : -1.0,
                   factor = std::sqrt((jjc + 1.0) / (jjb + 1.0));
            if (na == nb)
              x += phase * factor *
                   HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
            else if (na == nb + 2)
              x += phase * factor *
                   HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, dbs) *
                   HO::bosonCreatorRME(nc, llc, jjc, nb, llb, jjb, dbs);
          }
        }
      }
    }
    else
      throw std::invalid_argument("RME has not been implemented");

    return x;
  }

  std::pair<bool, std::vector<double>>
  OneBodyOperator::getTensorStrength_ME(int lam0, int mu0, int P10, int P20,
                                     int P30, int kappa0, int LL0, int SS0,
                                     int na, int nb, sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    std::vector<double> x(kappa0, 0.0);

    // loop over la
    for (int lla = 2 * na; lla >= 2 * (na % 2); lla -= 4)
    {
      // loop over jja
      for (int jja = lla + 1; jja >= std::fabs(lla - 1); jja -= 2)
      {
        // loop over lb
        for (int llb = 2 * nb; llb >= 2 * (nb % 2); llb -= 4)
        {
          // check if lla x llb -> LL0 coupling is allowed
          if (LL0 < std::fabs(lla - llb) || LL0 > lla + llb)
            continue;

          // loop over jjb
          for (int jjb = llb + 1; jjb >= std::fabs(llb - 1); jjb -= 2)
          {
            // check if jja x jjb -> JJ coupling is allowed
            if (JJ() < std::fabs(jja - jjb) || JJ() > jja + jjb)
              continue;

            double rme;

            double U9j;
            if (na <= nb)
              U9j = racah9.get({lla, 1, jja, llb, 1, jjb, LL0, SS0, JJ()});
            else // na > nb
              U9j = racah9.get({llb, 1, jjb, lla, 1, jja, LL0, SS0, JJ()});

            Eigen::MatrixXd Coeffs3, Coeffs4;
            if (na <= nb)
            {
              Coeffs3 =
                  wigU3SO3db
                      .get(na, 0, lla / 2, 0, nb, llb / 2, lam0, mu0, LL0 / 2)
                      .second;
              Coeffs4 =
                  wigU4SU2SU2db
                      .get(1, 0, 0, 1, 1, 0, 0, 1, 1, 1, P10, P20, P30, SS0, TT())
                      .second;
            }
            else // na > nb
            {
              Coeffs3 =
                  wigU3SO3db
                      .get(0, nb, llb / 2, na, 0, lla / 2, lam0, mu0, LL0 / 2)
                      .second;
              Coeffs4 =
                  wigU4SU2SU2db
                      .get(0, 0, 1, 1, 1, 1, 0, 0, 1, 1, P10, P20, P30, SS0, TT())
                      .second;
            }

            // loop over projections
            for (int mmja = jja; mmja >= -jja; mmja -= 2)
            {
              int mmjb = mmja - MMJ(); // selection rule of sphericalME

              for (int mmta = 1; mmta >= -1; mmta -= 2)
              {
                int mmtb = mmta - MMT(); // selection rule of sphericalME

                double me = sphericalME(na, lla, jja, mmja, mmta, nb, llb,
                                          jjb, mmjb, mmtb, dbs);
                if (me == 0.0) continue;

                // phase = (-)^{(nb + lb + 1)/2 - jb + mjb + mtb}
#ifdef ADD_SU3LIB_PHASE_U3SO3
                double phase =
                    ((2 * nb + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0)
                          ? 1.0
                          : -1.0;
#else
                double phase =
                    ((nb + llb / 2 + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0)
                          ? 1.0
                          : -1.0;
#endif
                if (na <= nb)
                {
                  double cgcJ = cgcs.get({jja, jjb, JJ(), mmja, -mmjb, MMJ()}),
                         cgcT = cgcs.get({1, 1, TT(), mmta, -mmtb, MMT()});

                  for (int k = 0; k < kappa0; k++)
                  {
                    double y = phase * cgcJ * cgcT * U9j * Coeffs3(k, 0) *
                                 Coeffs4(0, 0) * me;
                    // std::cout << y << std::endl;
                    x[k] += y;
                  }
                }
                else // na > nb
                {
                  double cgcJ = cgcs.get({jjb, jja, JJ(), -mmjb, mmja, MMJ()}),
                         cgcT = cgcs.get({1, 1, TT(), -mmtb, mmta, MMT()});

                  for (int k = 0; k < kappa0; k++)
                  {
                    double y = phase * cgcJ * cgcT * U9j * Coeffs3(k, 0) *
                                 Coeffs4(0, 0) * me;
                    // std::cout << y << std::endl;
                    x[k] -= y;
                  }
                }
              }
            }
          }
        }
      }
    }
    // std::cout << "Finish getTensorStrength\n";
    double sum = 0.0;
    for (const auto& X : x) sum += std::fabs(X);
    
    return std::make_pair((std::fabs(sum) < 1e-8), x);
  }

  // Using sphericalRME to avoid loops over projection mj and mt
  std::pair<bool, std::vector<double>>
  OneBodyOperator::getTensorStrength_RME(int lam0, int mu0, int P10, int P20,
                                     int P30, int kappa0, int LL0, int SS0,
                                     int na, int nb, sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    std::vector<double> x(kappa0, 0.0);

    // loop over la
    for (int lla = 2 * na; lla >= 2 * (na % 2); lla -= 4)
    {
      // loop over jja
      for (int jja = lla + 1; jja >= std::fabs(lla - 1); jja -= 2)
      {
        // loop over lb
        for (int llb = 2 * nb; llb >= 2 * (nb % 2); llb -= 4)
        {
          // check if lla x llb -> LL0 coupling is allowed
          if (LL0 < std::fabs(lla - llb) || LL0 > lla + llb)
            continue;

          // loop over jjb
          for (int jjb = llb + 1; jjb >= std::fabs(llb - 1); jjb -= 2)
          {
            // check if jja x jjb -> JJ coupling is allowed
            if (JJ() < std::fabs(jja - jjb) || JJ() > jja + jjb)
              continue;

            double rme = sphericalRME(na, lla, jja, nb, llb, jjb, dbs);
            if (rme == 0.0) continue;

            double U9j;
            if (na <= nb)
              U9j = racah9.get({lla, 1, jja, llb, 1, jjb, LL0, SS0, JJ()});
            else // na > nb
              U9j = racah9.get({llb, 1, jjb, lla, 1, jja, LL0, SS0, JJ()});

            Eigen::MatrixXd Coeffs3, Coeffs4;
            if (na <= nb)
            {
              Coeffs3 =
                  wigU3SO3db
                      .get(na, 0, lla / 2, 0, nb, llb / 2, lam0, mu0, LL0 / 2)
                      .second;
              Coeffs4 =
                  wigU4SU2SU2db
                      .get(1, 0, 0, 1, 1, 0, 0, 1, 1, 1, P10, P20, P30, SS0, TT())
                      .second;

#ifdef ADD_SU3LIB_PHASE_U3SO3
              double phase = ((nb + 1) % 2 == 0) ? 1.0 : -1.0;
#else
              double phase = ((nb + llb / 2 + 2) / 2 % 2 == 0) ? 1.0 : -1.0;
#endif
              double factor = std::sqrt(2.0 * (jja + 1.0) / (JJ() + 1.0) /
                                          (TT() + 1.0));
              for (int k = 0; k < kappa0; k++)
              {
                double y = phase * U9j * Coeffs3(k, 0) * Coeffs4(0, 0) * rme *
                           factor;
                // std::cout << y << std::endl;
                x[k] += y;
              }
            }
            else // na > nb
            {
              Coeffs3 =
                  wigU3SO3db
                      .get(0, nb, llb / 2, na, 0, lla / 2, lam0, mu0, LL0 / 2)
                      .second;
              Coeffs4 =
                  wigU4SU2SU2db
                      .get(0, 0, 1, 1, 1, 1, 0, 0, 1, 1, P10, P20, P30, SS0, TT())
                      .second;

#ifdef ADD_SU3LIB_PHASE_U3SO3
              double phase = ((2* nb + jja + jjb - JJ() - TT()) / 2 % 2 == 0)
                               ? 1.0
                               : -1.0;
#else
              double phase =
                           ((nb + llb / 2 + jja + jjb - JJ() - TT()) / 2 % 2 == 0)
                               ? 1.0
                               : -1.0;
#endif
              double factor = std::sqrt(2.0 * (jja + 1.0) / (JJ() + 1.0) /
                                          (TT() + 1.0));
              for (int k = 0; k < kappa0; k++)
              {
                double y = phase * U9j * Coeffs3(k, 0) * Coeffs4(0, 0) * rme *
                           factor;
                // std::cout << y << std::endl;
                x[k] -= y;
              }
            }
          }
        }
      }
    }
    // std::cout << "Finish getTensorStrength\n";
    double sum = 0.0;
    for (const auto& X : x)
      sum += std::fabs(X);
    
    return std::make_pair((std::fabs(sum) < 1e-8), x);
  }

  U3xU4::OneBodyMultiShellTensors
    OneBodyOperator::expand_old(sam::Databases& dbs) const
  {
    U3xU4::OneBodyMultiShellTensors tensors;

    // loop for ad
    for (int na = 0; na <= maxShell_; na++)
    {
      u3::irrep u3a(na, 0, 0);
      u4::irrep u4a(1, 0, 0, 0);

      // loop for at
      for (int nb = 0; nb <= maxShell_; nb++)
      {
        u3::irrep u3b(nb, nb, 0);
        u4::irrep u4b(1, 1, 1, 0);

        // coupled irreps
        std::vector<u3::irrep> u3couple;
        std::vector<int> u3multi;

        std::vector<u4::irrep> u4couple;
        std::vector<int> u4multi;

        if (na <= nb)
        {
          std::tie(u3couple, u3multi) = u3::decomposition(u3a, u3b);
          std::tie(u4couple, u4multi) = u4::decomposition(u4a, u4b);
        }
        else // na > nb => switch the order
        {
          std::tie(u3couple, u3multi) = u3::decomposition(u3b, u3a);
          std::tie(u4couple, u4multi) = u4::decomposition(u4b, u4a);
        }

        for (int i = 0; i < u3couple.size(); i++)
        {
          assert(u3multi[i] == 1);

          // decompose into L0
          std::vector<u3::physicalvector> pv3s = u3::physicalBasis(u3couple[i]);

          for (int j = 0; j < u4couple.size(); j++)
          {
            assert(u4multi[j] == 1);

            // decompose into 2S0 and 2T0
            std::vector<u4::physicalvector> pv4s =
                u4::physicalBasis(u4couple[j]);

            for (int pv3 = 0; pv3 < pv3s.size(); pv3++)
            {
              int lam0 = pv3s[pv3].lam(), mu0 = pv3s[pv3].mu(),
                  kappa0 = pv3s[pv3].kappa(), LL0 = 2 * pv3s[pv3].L();

              for (int pv4 = 0; pv4 < pv4s.size(); pv4++)
              {
                int P10 = pv4s[pv4].P1(), P20 = pv4s[pv4].P2(),
                    P30 = pv4s[pv4].P3(), SS0 = pv4s[pv4].SS(),
                    TT0 = pv4s[pv4].TT();

                // couple LL0 and SS0 to JJ0
                // the sum over CGCs will kill JJ0 != JJ and TT0 != TT
                if (JJ() < std::fabs(LL0 - SS0) || JJ() > (LL0 + SS0) || TT0 != TT())
                  continue;

                // the selection rule of CGCs also dictates that
                // MMJ0 == MMJ and MMT0 == MMT

                // compute the expansion coefficients
                auto [ignore, coeff] = getTensorStrength_ME(
                    lam0, mu0, P10, P20, P30, kappa0, LL0, SS0, na,
                    nb, // pv3as, pv3bs, pv4as, pv4bs,
                    dbs);

                if (ignore == false)
                {
                  if (na <= nb)
                  {
                    U3xU4::OneBodyMultiShellTensor tensor(
                        U3xU4::irrep(u3a, u4a), U3xU4::irrep(u3b, u4b),
                        U3xU4::irrep(u3couple[i], u4couple[j]), kappa0, LL0,
                        SS0, TT_, MMT_, JJ_, MMJ_, coeff);
                    tensors.push_back(tensor);
                  }
                  else // na > nb
                  {
                    U3xU4::OneBodyMultiShellTensor tensor(
                        U3xU4::irrep(u3b, u4b), U3xU4::irrep(u3a, u4a),
                        U3xU4::irrep(u3couple[i], u4couple[j]), kappa0, LL0,
                        SS0, TT_, MMT_, JJ_, MMJ_, coeff);
                    tensors.push_back(tensor);
                  }
                }
              }
            }
          }
        }
      }
    }

    return tensors;
  }

  // Selection rules
  bool OneBodyOperator::LLselection(const int& LL) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (LL != 0) flag = false;
    }
    
    return flag;
  }
  
  bool OneBodyOperator::SSselection(const int& SS) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (SS != 0) flag = false;
    }
    
    return flag;
  }

  bool OneBodyOperator::U3selection(const u3::irrep& irr) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" || name_ == "CoMNumberOp")
    {
      if (irr != u3::irrep(0, 0, 0)) flag = false;
    }
    else if (name_ == "Kinetic")
    {
      if (irr != u3::irrep(0, 0, 0) &&
          irr != u3::irrep(2, 0, 0) &&
          irr != u3::irrep(2, 2, 0)) 
        flag = false;
    }
    
    return flag;
  }

  bool OneBodyOperator::U4selection(const u4::irrep& irr) const
  {
    bool flag = true;

    if (name_ == "L.L" || name_ == "L.S" ||
        name_ == "CoMNumberOp" || name_ == "Kinetic")
    {
      if (irr != u4::irrep(0, 0, 0, 0)) flag = false;
    }
    
    return flag;
  }

  void OneBodyOperator::setTensorStrength(U3xU4::MultiShellTensor_phys& tensor,
                                    int na, int nb, sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& racah9 = dbs.racah9;
    auto& wigU3SO3db = dbs.wigU3SO3db;
    auto& wigU4SU2SU2db = dbs.wigU4SU2SU2db;

    std::vector<std::tuple<uint8_t,              // kappamax
                           HO::qn_t,             // LL
                           uint8_t,              // zetamax
                           HO::qn_t,             // SS
                           HO::qn_t,             // TT
                           std::vector<double>>> LLSSTTstrengths;

    // decompose into L
    u3::physicalvectors pv3s = u3::physicalBasis(tensor.getLastIrrep().u3());

    // decompose into 2S and 2T
    u4::physicalvectors pv4s = u4::physicalBasis(tensor.getLastIrrep().u4());

    for (const auto& pv3 : pv3s)
    {
      int lam0 = pv3.lam(), mu0 = pv3.mu(),
          kappa0 = pv3.kappa(), LL0 = 2 * pv3.L();

      if (LLselection(LL0) == false) continue;

      for (const auto& pv4 : pv4s)
      {
        assert(pv4.zeta() == 1);

        int P10 = pv4.P1(), P20 = pv4.P2(), P30 = pv4.P3(),
            SS0 = pv4.SS(), TT0 = pv4.TT();

        if (SSselection(SS0) == false) continue;

        // if (JJ() < abs(LL0 - SS0) || JJ() > (LL0 + SS0) || TT0 != TT())
        if (u2::mult(LL0, SS0, JJ()) == 0 || TT0 != TT())
          continue;

        auto [allowU4, cgcU4] = (na <= nb)
          ? wigU4SU2SU2db.get(
              1, 0, 0, 1, 1, 0, 0, 1, 1, 1, P10, P20, P30, SS0, TT())
          : wigU4SU2SU2db.get(
              0, 0, 1, 1, 1, 1, 0, 0, 1, 1, P10, P20, P30, SS0, TT());

        if (allowU4 == false) continue;

        std::vector<double> strengths(kappa0, 0.0);

        // loop over la
        for (int lla = 2 * na; lla >= 2 * (na % 2); lla -= 4)
        {
          // loop over lb
          for (int llb = 2 * nb; llb >= 2 * (nb % 2); llb -= 4)
          {
            auto [allowU3, cgcU3] = (na <= nb) 
              ? wigU3SO3db.get(
                  na, 0, lla / 2, 0, nb, llb / 2, lam0, mu0, LL0 / 2)
              : wigU3SO3db.get(
                  0, nb, llb / 2, na, 0, lla / 2, lam0, mu0, LL0 / 2);

            if (allowU3 == false) continue;

            // loop over jja
            for (int jja = lla + 1; jja >= abs(lla - 1); jja -= 2)
            {
              // loop over jjb
              for (int jjb = llb + 1; jjb >= std::fabs(llb - 1); jjb -= 2)
              {
                // check if jja x jjb -> JJ coupling is allowed
                if (JJ() < abs(jja - jjb) || JJ() > jja + jjb) continue;

                double U9j = (na <= nb)
                  ? racah9.get({lla, 1, jja, llb, 1, jjb, LL0, SS0, JJ()})
                  : racah9.get({llb, 1, jjb, lla, 1, jja, LL0, SS0, JJ()});

                // loop over projections
                for (int mmja = jja; mmja >= -jja; mmja -= 2)
                {
                  int mmjb = mmja - MMJ(); // selection rule of sphericalME
                  if (abs(mmjb) > jjb) continue;

                  double cgcJ = (na <= nb)
                    ? cgcs.get({jja, jjb, JJ(), mmja, -mmjb, MMJ()})
                    : cgcs.get({jjb, jja, JJ(), -mmjb, mmja, MMJ()});

                  for (int mmta = 1; mmta >= -1; mmta -= 2)
                  {
                    int mmtb = mmta - MMT(); // selection rule of sphericalME
                    if (abs(mmtb) > 1) continue;

                    double cgcT = (na <= nb)
                      ? cgcs.get({1, 1, TT(), mmta, -mmtb, MMT()})
                      : cgcs.get({1, 1, TT(), -mmtb, mmta, MMT()});

                    double me = sphericalME(na, lla, jja, mmja, mmta, nb, llb,
                                            jjb, mmjb, mmtb, dbs);
                    if (me == 0.0) continue;

                    // phase = (-)^{(nb + lb + 1)/2 - jb + mjb + mtb}
#ifdef ADD_SU3LIB_PHASE_U3SO3
                    double phase = 
                      ((2 * nb + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0) 
                      ? 1.0 : -1.0;
#else
                    double phase = 
                      ((nb + llb / 2 + 1 - jjb + mmjb + mmtb) / 2 % 2 == 0)
                      ? 1.0 : -1.0;
#endif
                    double cgcU2 = phase * U9j * cgcJ * cgcT * cgcU4(0, 0) * me;
                    if (na > nb) cgcU2 *= -1.0;

                    for (int k = 0; k < kappa0; k++)
                    {
                      double y =  cgcU2 * cgcU3(k, 0);
                      // std::cout << y << std::endl;
                      strengths[k] += y;
                    }
                  }
                }
              }
            }
          }
        }

        double sum = 0.0;
        for (const auto& s : strengths) sum += std::fabs(s);

        if (std::fabs(sum) > 1e-8)
          LLSSTTstrengths.emplace_back(kappa0, LL0, 1, SS0, TT0, strengths);
      }
    }

    tensor.setLLSSTTstrengths(LLSSTTstrengths);
  }

  U3xU4::MultiShellTensors_phys 
    OneBodyOperator::expand_new(sam::Databases& dbs) const
  {
    U3xU4::MultiShellTensors_phys tensors;

    u4::irrep u4a(1, 0, 0, 0), u4b(1, 1, 1, 0);
    auto [u4couple, u4multi] = u4::decomposition(u4a, u4b); 
    // same as u4::decomposition(u4b, u4a)

    // loop for ad
    for (int na = 0; na <= maxShell_; na++)
    {
      if ((na - valenceShell_) > Nmax_) continue;

      u3::irrep u3a(na, 0, 0);
      U3xU4::SingleShellTensor 
        a(na, 1, 0, {}, {}, U3xU4::irrep(u3a, u4a), 1, 1);

      // loop for at
      for (int nb = 0; nb <= maxShell_; nb++)
      {
        if ((nb - valenceShell_) > Nmax_) continue;

        u3::irrep u3b(nb, nb, 0);
        U3xU4::SingleShellTensor 
          b(nb, 0, 1, {}, {}, U3xU4::irrep(u3b, u4b), 1, 1);

        // coupled irreps
        std::vector<u3::irrep> u3couple;
        std::vector<int> u3multi;

        if (na <= nb)
          std::tie(u3couple, u3multi) = u3::decomposition(u3a, u3b);
        else // na > nb => switch the order
          std::tie(u3couple, u3multi) = u3::decomposition(u3b, u3a);
        
        for (int i = 0; i < u3couple.size(); i++)
        {
          assert(u3multi[i] == 1);

          if (U3selection(u3couple[i]) == false) continue;

          for (int j = 0; j < u4couple.size(); j++)
          {
            assert(u4multi[j] == 1); 

            if (U4selection(u4couple[j]) == false) continue;

            U3xU4::OmegaElement t(1, u3couple[i], 1, u4couple[j]);

            if (na == nb)
            {
              U3xU4::SingleShellTensor ab(na, 1, 1, {}, {}, 
                            U3xU4::irrep(u3couple[i], u4couple[j]), 1, 1);

              U3xU4::MultiShellTensor_phys tensor({ab}, {}, MMT_, JJ_, MMJ_, labels_);

              setTensorStrength(tensor, na, nb, dbs);

              if (!tensor.LLSSTTstrengths().empty())
                tensors.push_back(std::move(tensor));
            }
            else
            {
              U3xU4::MultiShellTensor_phys tensor(
                (na < nb) ? std::vector({a, b}) : std::vector({b, a}), 
                {t}, MMT_, JJ_, MMJ_, labels_);

              setTensorStrength(tensor, na, nb, dbs);

              if (!tensor.LLSSTTstrengths().empty())
                tensors.push_back(std::move(tensor));
            }
          }
        }
      }
    }

    return tensors;
  }

  U3xU4::MultiShellTensors_phys 
    OneBodyOperator::expand(sam::Databases& dbs)
  {
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");

    std::string base_dir = std::string(sam_dir) + "/data/OneBodyOperatorExpansion";
    std::filesystem::create_directories(base_dir);

    std::string filename = base_dir + "/" + name_ 
                         + "_hw" + std::to_string(static_cast<int>(hw_));

    // if this file exists, returns immediately
    if (std::filesystem::exists(filename))
    {
      return readBinary(filename, dbs);
    }
    else
    {
      auto tensors = expand_new(dbs);
      writeBinary(filename, tensors);
      return tensors;
    }
  }

  // Write to binary file
  void OneBodyOperator::writeBinary(const std::string& filename, 
                     const U3xU4::MultiShellTensors_phys& tensors) const
  {
    std::cout << "Start writing " << name_ << " to binary file" << std::endl;
    
    if (valenceShell_ > 0)
      throw std::runtime_error("Set valenceShell to 0 for reusability");

    if (Nmax_ != maxShell_)
      throw std::runtime_error("Set Nmax and maxShell equal");

    std::ofstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for writing: " + filename);

    qcx_utils::writeBinary(s, maxShell_);

    // Number of tensors
    size_t n_tensors = tensors.size();
    qcx_utils::writeBinary(s, n_tensors);

    for (const auto& tensor : tensors) tensor.writeBinary(s);

    std::cout << "Finish writing to binary file" << std::endl;
  }

  // Read from binary file
  U3xU4::MultiShellTensors_phys OneBodyOperator::readBinary(
    const std::string& filename, sam::Databases& dbs)
  {
    std::ifstream s(filename, std::ios::binary);
    if (!s)
      throw std::runtime_error("Failed to open file for reading: " + filename);

    HO::qn_t maxShell;
    qcx_utils::readBinary(s, maxShell);

    std::cout << "Start reading " << name_ 
              << " from binary file with capacity maxShell = " 
              << static_cast<int>(maxShell) << std::endl;
    
    if (maxShell_ - valenceShell_ > maxShell /* Nmax */)
    {
      std::cout << "File doesn't reach your Nmax, so it will be expanded now"
                << std::endl;
      valenceShell_ = 0;
      Nmax_ = maxShell_;

      auto tensors = expand_new(dbs);
      writeBinary(filename, tensors);
      return tensors;
    }

    // Read number of tensors
    size_t n_tensors;
    qcx_utils::readBinary(s, n_tensors);

    U3xU4::MultiShellTensors_phys tensors;
    tensors.reserve(n_tensors);

    for (size_t i = 0; i < n_tensors; ++i)
    { 
      U3xU4::MultiShellTensor_phys tensor(labels_);
      tensor.readBinary(s);

      if (tensor.Gammas().back().shell() - valenceShell_ > Nmax_)
        continue;

      tensors.push_back(std::move(tensor));
    }
    std::cout << "Finish reading from binary file" << std::endl;
    return tensors;
  }

  // Matrix representation
  Eigen::SparseMatrix<double> OneBodyOperator::matrix(
    const U3xU4::ManyNucleonBasis& basis, sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return U3xU4::MatrixElements(tensors,basis,dbs);
  }
  Eigen::SparseMatrix<double> OneBodyOperator::matrix(
    const U3xU4::ManyNucleonBasis& bras,
    const U3xU4::ManyNucleonBasis& kets, sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return U3xU4::MatrixElements(tensors, bras, kets, dbs);
  }

  // Tensor Analyzers
  static std::string analysis_dir()
  {
    const char* sam_dir = std::getenv("SAM_DIR");
    if (!sam_dir)
      throw std::runtime_error("Environment variable SAM_DIR not set");

    std::string base_dir = std::string(sam_dir) 
                         + "/data/OneBodyOperatorExpansion/Analysis";
    std::filesystem::create_directories(base_dir);
    return base_dir;
  }

  std::vector<std::pair<u3::irrep, sam::strength_stats>>
    OneBodyOperator::U3_tensor_analyzer_write(sam::Databases& dbs)
  {
    std::string filename = analysis_dir() + "/" + name_ 
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_U3.dat";

    auto tensors = expand(dbs);
    return sam::U3_tensor_analyzer(tensors, filename);
  }
  
  std::vector<std::pair<u3::irrep, sam::strength_stats>>
    OneBodyOperator::U3_tensor_analyzer(sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return sam::U3_tensor_analyzer(tensors);
  }

  std::vector<std::pair<u4::irrep, sam::strength_stats>>
    OneBodyOperator::U4_tensor_analyzer_write(sam::Databases& dbs)
  {
    std::string filename = analysis_dir() + "/" + name_ 
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_U4.dat";

    auto tensors = expand(dbs);
    return sam::U4_tensor_analyzer(tensors, filename);
  }
  
  std::vector<std::pair<u4::irrep, sam::strength_stats>>
    OneBodyOperator::U4_tensor_analyzer(sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return sam::U4_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    OneBodyOperator::LL_tensor_analyzer_write(sam::Databases& dbs)
  {
    std::string filename = analysis_dir() + "/" + name_ 
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_LL.dat";

    auto tensors = expand(dbs);
    return sam::LL_tensor_analyzer(tensors, filename);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    OneBodyOperator::LL_tensor_analyzer(sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return sam::LL_tensor_analyzer(tensors);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    OneBodyOperator::SS_tensor_analyzer_write(sam::Databases& dbs)
  {
    std::string filename = analysis_dir() + "/" + name_ 
                         + "_hw" + std::to_string(static_cast<int>(hw_))
                         + "_Nmax" + std::to_string(static_cast<int>(Nmax_))
                         + "_SS.dat";

    auto tensors = expand(dbs);
    return sam::SS_tensor_analyzer(tensors, filename);
  }

  std::vector<std::pair<int, sam::strength_stats>>
    OneBodyOperator::SS_tensor_analyzer(sam::Databases& dbs)
  {
    auto tensors = expand(dbs);
    return sam::SS_tensor_analyzer(tensors);
  }

} // namespace JT
