#include <cassert>
#include <stdexcept>
#include <utility>

#include <UNtoU3.h>

#include <Databases.h>
#include <U3xU4HWSgenerator.h>
#include <U3xU4SingleShellTensors.h>
#include <RME.h>

namespace U3xU4
{
  sam::RME SingleShellTensor::RME(
    //std::size_t& ai_max, const U3xU4::irrep& i,
    const U3xU4::GammaElement& i,
    //std::size_t& af_max, const U3xU4::irrep& f,
    const U3xU4::GammaElement& f,
    std::size_t& rhomax, std::size_t& etamax,
    sam::Databases& dbs) const
  {
    auto& cgcs = dbs.cgcs;
    auto& wigU3U2db = dbs.wigU3U2db;
    auto& wigU4U3db = dbs.wigU4U3db;

    int A_i = i.particles(), N_quanta_i = i.quanta(), // initial irreps
        A_f = f.particles(), N_quanta_f = f.quanta(); // final irreps

    if ((A_i - ats + ads) != A_f) // number of particles is not conserved
      return {};

    auto ai_max = i.alphamax();
    auto af_max = f.alphamax();

    // If this is an identity tensor
    if (ads == 0 && ats == 0)
    {
      assert(prod_ads.empty());
      assert(prod_ats.empty());
      assert(T() == U3xU4::irrep(0,0,0, 0,0,0,0));
      assert(rho0max() == 1);
      assert(eta0max() == 1);

      if (i == f)
      {
        rhomax = 1;
        etamax = 1;
/*
        // If ai_max or af_max  has not been calculated
        int N = (shell() + 1) * (shell() + 2) / 2;
        untou3::UNtoU3<4> gen;
        gen.set_shell(shell());
        if (ai_max <= 0)
        {
          gen.reduceU4(i.u4(), N);
          ai_max = gen.multiplicity(i.u3());
        }
        if (af_max <= 0)
        {
          gen.reduceU4(f.u4(), N);
          af_max = gen.multiplicity(f.u3());
        }
*/
        // Now calculate RMEs
        std::vector<double> rmes;
        for (std::size_t af = 0; af < af_max; af++)
          for (std::size_t ai = 0; ai < ai_max; ai++)
            rmes.push_back((ai == af) ? 1.0 : 0.0);

        return {std::move(rmes),
                static_cast<size_t>(af_max),
                static_cast<size_t>(ai_max),
                1,
                1,
                1};
      }
      else
        return {};
    }

    // Vacuum on both side can only be connected by an identity
    if (A_i == 0 && A_f == 0) 
      return {};

    if (A_i > 0 && shell() != N_quanta_i / A_i)
      return {};

    if (A_f > 0 && shell() != N_quanta_f / A_f)
      return {};

    // Get outer multiplicities due to Wigner-Eckart theorem
    rhomax = u3::Rho(i.u3(), T().u3(), f.u3());
    etamax = u4::Eta(i.u4(), T().u4(), f.u4());
    if (rhomax == 0 || etamax == 0) // i x t does not produce f
      return {};

    // Building single-particle states
    u3::irrep x(shell(), 0, 0), x_conj(shell(), shell(), 0);
    std::vector<u3::gelfand> sps3 = u3::gelfandBasis(x),
                             sps3_conj = u3::gelfandBasis(x_conj);

    u4::irrep y(1, 0, 0, 0), y_conj(1, 1, 1, 0);
    std::vector<u4::gelfand> sps4 = u4::gelfandBasis(y),
                             sps4_conj = u4::gelfandBasis(y_conj);

    // Generate HWS for the ket irrep
    u3::gelfand i3hws(i.u3().n13(), i.u3().n23(), i.u3().n33(), i.u3().n13(),
                      i.u3().n23(), i.u3().n13());
    u4::gelfand i4hws(i.u4().n14(), i.u4().n24(), i.u4().n34(), i.u4().n44(),
                      i.u4().n14(), i.u4().n24(), i.u4().n34(), i.u4().n14(),
                      i.u4().n24(), i.u4().n14());
    std::vector<UNxU4::SlaterDeterminant> SDarr_i;
    std::vector<Eigen::RowVectorXd> coeffs_i;
    if (A_i == 0) // vacuum
    {
      const boost::dynamic_bitset<> bits(2 * (shell() + 1) * (shell() + 2), 0);
      UNxU4::SlaterDeterminant SDi(bits);
      SDarr_i.push_back(SDi);
      Eigen::RowVectorXd c = Eigen::RowVectorXd::Constant(1, 1.0);
      coeffs_i.push_back(c);
    }
    else
    {
      std::filesystem::path path_i =
          U3xU4::get_HWS_file(A_i, shell(), i, sps3, sps4);
      std::tie(SDarr_i, coeffs_i) = U3xU4::read_HWS_data(path_i);
    }
    // Check pre-calculated multiplicity is correct
    assert(ai_max == coeffs_i[0].cols());
    // if (ai_max > 0 && ai_max != coeffs_i[0].cols())
    //   throw std::runtime_error("max(alpha_i) disagree with nullity");
    // ai_max = coeffs_i[0].cols();
    // std::cout << "coeffs_i:" << std::endl;
    // for (int i = 0; i < coeffs_i.size(); i++)
    //     std::cout << coeffs_i[i] << std::endl;

    // Generate HWS for the bra irrep
    u3::gelfand f3hws(f.u3().n13(), f.u3().n23(), f.u3().n33(), f.u3().n13(),
                      f.u3().n23(), f.u3().n13());
    u4::gelfand f4hws(f.u4().n14(), f.u4().n24(), f.u4().n34(), f.u4().n44(),
                      f.u4().n14(), f.u4().n24(), f.u4().n34(), f.u4().n14(),
                      f.u4().n24(), f.u4().n14());
    std::vector<UNxU4::SlaterDeterminant> SDarr_f;
    std::vector<Eigen::RowVectorXd> coeffs_f;
    if (A_f == 0) // vacuum
    {
      const boost::dynamic_bitset<> bits(2 * (shell() + 1) * (shell() + 2), 0);
      UNxU4::SlaterDeterminant SDf(bits);
      SDarr_f.push_back(SDf);
      Eigen::RowVectorXd c = Eigen::RowVectorXd::Constant(1, 1.0);
      coeffs_f.push_back(c);
    }
    else
    {
      std::filesystem::path path_f =
          U3xU4::get_HWS_file(A_f, shell(), f, sps3, sps4);
      std::tie(SDarr_f, coeffs_f) = U3xU4::read_HWS_data(path_f);
    }
    // Check pre-calculated multiplicity is correct
    assert(af_max == coeffs_f[0].cols());
    // if (af_max > 0 && af_max != coeffs_f[0].cols())
    //   throw std::runtime_error("max(alpha_f) disagree with nullity");
    // af_max = coeffs_f[0].cols();
    // std::cout << "coeffs_f:" << std::endl;
    // for (int i = 0; i < coeffs_f.size(); i++)
    //     std::cout << coeffs_f[i] << std::endl;

    // Array of matricies containing full matrix elements:
    // index = alpha_i * max(alpha_f) + alpha_f
    std::vector<Eigen::MatrixXd> FMEs;
    for (std::size_t ai = 0; ai < ai_max; ai++)
    {
      for (std::size_t af = 0; af < af_max; af++)
      {
        Eigen::MatrixXd fme(rhomax * etamax, rho0max() * eta0max());
        fme.setZero();
        FMEs.push_back(fme);
      }
    }

    // A matrix containing coupling coefficients
    Eigen::MatrixXd CGCs(rhomax * etamax, rhomax * etamax);
    CGCs.setZero();

    // Generate tensor components of the tensor
    std::vector<u3::gelfand> g3Arr = u3::gelfandBasis(T().u3());
    std::vector<std::size_t> g3Arr0;
    int missing3 = (i.u3().n13() + i.u3().n23() + i.u3().n33() +
                    T().u3().n13() + T().u3().n23() + T().u3().n33() -
                    f.u3().n13() - f.u3().n23() - f.u3().n33()) /
                   3;

    for (std::size_t i = 0; i < g3Arr.size(); i++)
      if ((i3hws.e22() + g3Arr[i].e22()) == (f3hws.e22() + missing3) &&
          (i3hws.e11() + g3Arr[i].e11()) == (f3hws.e11() + missing3))
        g3Arr0.push_back(i);

    std::vector<u4::gelfand> g4Arr = u4::gelfandBasis(T().u4());
    std::vector<std::size_t> g4Arr0;
    int missing4 =
        (i.u4().n14() + i.u4().n24() + i.u4().n34() + i.u4().n44() +
         T().u4().n14() + T().u4().n24() + T().u4().n34() + T().u4().n44() -
         f.u4().n14() - f.u4().n24() - f.u4().n34() - f.u4().n44()) /
        4;

    for (std::size_t i = 0; i < g4Arr.size(); i++)
      if ((i4hws.e33() + g4Arr[i].e33()) == (f4hws.e33() + missing4) &&
          (i4hws.e22() + g4Arr[i].e22()) == (f4hws.e22() + missing4) &&
          (i4hws.e11() + g4Arr[i].e11()) == (f4hws.e11() + missing4))
        g4Arr0.push_back(i);

    // Get angular momenta
    int jj_i3 = i3hws.Lambdax2(), mm_i3 = i3hws.MLambdax2(),
        jj_f3 = f3hws.Lambdax2(), mm_f3 = f3hws.MLambdax2(),
        jj_i4 = i4hws.Lambdax2(), mm_i4 = i4hws.MLambdax2(),
        jj_f4 = f4hws.Lambdax2(), mm_f4 = f4hws.MLambdax2();

    // std::cout << "Count tensor components: " << g3Arr0.size()
    //		  <<" x " << g4Arr0.size() << std::endl;

    // Now run through tensor components
    int row = 0;
    bool done = false;
    for (std::size_t g3 = 0; g3 < g3Arr0.size() && !done; g3++)
    {
      u3::gelfand gt3 = g3Arr[g3Arr0[g3]];
      int jj_t3 = gt3.Lambdax2(), mm_t3 = gt3.MLambdax2();

      auto [allowU3, accU3] = wigU3U2db.get(
          i3hws.n13(), i3hws.n23(), i3hws.n33(), i3hws.n12(), i3hws.n22(),
          gt3.n13(), gt3.n23(), gt3.n33(), gt3.n12(), gt3.n22(), f3hws.n13(),
          f3hws.n23(), f3hws.n33(), f3hws.n12(), f3hws.n22());

      if (allowU3 == false)
        continue;

      double cgc3 = cgcs.compute(jj_i3, jj_t3, jj_f3, mm_i3, mm_t3, mm_f3);

      auto cgcU3 = accU3 * cgc3;

      // std::cout << "cgcU3 " << cgcU3 << std::endl;

      for (std::size_t g4 = 0; g4 < g4Arr0.size() && !done; g4++)
      {

        u4::gelfand gt4 = g4Arr[g4Arr0[g4]];
        int jj_t4 = gt4.Lambdax2(), mm_t4 = gt4.MLambdax2();

        auto [allowU4, accU4] =
            wigU4U3db.get(i4hws.n14(), i4hws.n24(), i4hws.n34(), i4hws.n44(),
                          i4hws.n13(), i4hws.n23(), i4hws.n33(), gt4.n14(),
                          gt4.n24(), gt4.n34(), gt4.n44(), gt4.n13(), gt4.n23(),
                          gt4.n33(), f4hws.n14(), f4hws.n24(), f4hws.n34(),
                          f4hws.n44(), f4hws.n13(), f4hws.n23(), f4hws.n33());

        if (allowU4 == false)
          continue;

        auto [allowU3_4, accU3_4] = wigU3U2db.get(
            i4hws.n13(), i4hws.n23(), i4hws.n33(), i4hws.n12(), i4hws.n22(),
            gt4.n13(), gt4.n23(), gt4.n33(), gt4.n12(), gt4.n22(), f4hws.n13(),
            f4hws.n23(), f4hws.n33(), f4hws.n12(), f4hws.n22());

        if (allowU3_4 == false)
          continue;

        double cgc4 = cgcs.compute(jj_i4, jj_t4, jj_f4, mm_i4, mm_t4, mm_f4);

        Eigen::RowVectorXd cgcU4(etamax);
        cgcU4.setZero();

        for (std::size_t e = 0; e < etamax; e++)
          for (std::size_t r = 0; r < accU4.rows(); r++)
            cgcU4(e) += accU4(r, e) * accU3_4(r) * cgc4;

        // std::cout << "cgcU4 " << cgcU4 << std::endl;

        // Elements of CGCs
        for (std::size_t r = 0; r < rhomax; r++)
          for (std::size_t e = 0; e < etamax; e++)
            CGCs(row, r * etamax + e) = cgcU3(r) * cgcU4(e);

        // Expand tensor into ad's and a's
        U3xU4::SingleShellTensor_gelfand Tadat(
            shell(), sps3, sps3_conj, sps4, sps4_conj, ads, ats, prod_ads,
            prod_ats, T(), rho0max(), eta0max(), gt3, gt4);

        auto [coeffs, adArr, aArr] = Tadat.expand(dbs);
        /*
        for (std::size_t t = 0; t < coeffs.size(); t++)
        {
          std::cout << coeffs[t] << "\t";

          if (!adArr.empty())
            for (std::size_t j = 0; j < adArr[t].size(); j++)
              std::cout << adArr[t][j] << "\t";

          if (!aArr.empty())
            for (std::size_t j = 0; j < aArr[t].size(); j++)
              std::cout << aArr[t][j] << "\t";

          std::cout << std::endl;
        }
        */

        // Compute full matrix elements
        for (std::size_t t = 0; t < coeffs.size(); t++)
        {
          // Determine how ad's and a's act on SDi's
          for (std::size_t i = 0; i < SDarr_i.size(); i++)
          {
            auto SDi = SDarr_i[i];
            bool annihilated = false;

            double phase = 1;

            // Action of a's from right to left
            for (int a = ats - 1; a >= 0 && !annihilated; a--)
            {
              auto [p, b] = aArr[t][a].getTensorCharacters();
              if (SDi.is_occupied(p, b))
              {
                // std::cout << "a: " << p << " " << b << std::endl;
                phase *= (SDi.preceding_occupation(p, b) % 2 == 0) ? 1 : -1;
                SDi.vacate(p, b);
              }
              else
                annihilated = true;
            }

            // Action of ad's from right to left
            for (int ad = ads - 1; ad >= 0 && !annihilated; ad--)
            {
              auto [p, b] = adArr[t][ad].getTensorCharacters();
              if (!SDi.is_occupied(p, b))
              {
                // std::cout << "ad: " << p << " " << b << std::endl;
                phase *= (SDi.preceding_occupation(p, b) % 2 == 0) ? 1 : -1;
                SDi.occupy(p, b);
              }
              else
                annihilated = true;
            }

            // Check which SDf matches the new SDi
            if (!annihilated)
            {
              for (std::size_t f = 0; f < SDarr_f.size(); f++)
              {
                auto SDf = SDarr_f[f];
                if (SDi == SDf)
                {
                  for (std::size_t ai = 0; ai < ai_max; ai++)
                  {
                    for (std::size_t af = 0; af < af_max; af++)
                    {
                      FMEs[ai * af_max + af].row(row) +=
                        phase * coeffs_i[i](ai) * coeffs[t] * coeffs_f[f](af);
                    }
                  }
                }
              }
            }
          }
        }

        row++;
        if (row == CGCs.rows())
          done = true;
      }
    }

    // std::cout << "CGCs = " << CGCs << std::endl;

    // Solve for RMEs
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factorize;
    factorize.compute(CGCs);

    std::vector<Eigen::MatrixXd> RMEs(FMEs.size());
    for (std::size_t i = 0; i < FMEs.size(); i++)
    {
      // std::cout << "FME = " << FMEs[i] << std::endl;
      RMEs[i] = factorize.solve(FMEs[i]);
    }

    std::vector<double> rmes;
    for (std::size_t af = 0; af < af_max; af++)
      for (std::size_t ai = 0; ai < ai_max; ai++)
        for (int rho0 = 0; rho0 < rho0max(); rho0++)
          for (int eta0 = 0; eta0 < eta0max(); eta0++)
            for (int rho = 0; rho < rhomax; rho++)
              for (int eta = 0; eta < etamax; eta++)
              {
                double rme = RMEs[ai * af_max + af](rho * etamax + eta,
                                                    rho0 * eta0max() + eta0);
                rmes.push_back((std::fabs(rme) < 1e-10) ? 0.0 : rme);
              }

    size_t aT_max = rho0max() * eta0max();

    return {std::move(rmes),
            static_cast<size_t>(af_max),
            static_cast<size_t>(ai_max),
            aT_max,
            rhomax,
            etamax};
  }
} // namespace U3xU4
