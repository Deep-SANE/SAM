#include <cassert>
#include <stdexcept>
#include <utility>

// #define TIME_SINGLE_SHELL_TENSOR_RME
#ifdef TIME_SINGLE_SHELL_TENSOR_RME
#include <chrono>
#endif

#include <UNtoU3.h>

#include <U3U2xU3U2.h>
#include <U4U3xU4U3.h>

#include <U3xU4HWSgenerator.h>
#include <U3xU4SingleShellTensors.h>
#include <RME.h>

namespace U3xU4
{
  bool SingleShellTensor::RME_exist(const U3xU4::GammaElement& f, 
                                    const U3xU4::GammaElement& i) const
  {
    int A_i = i.particles(), N_quanta_i = i.quanta(), 
        A_f = f.particles(), N_quanta_f = f.quanta(),
        n = shell();

    // std::cout << "A_i:" << A_i << " A_f:" << A_f
    //           << " ads:" << num_ads() << " ats:" << num_ats() << std::endl;

    if (A_i > 0) assert(n == N_quanta_i / A_i);
    if (A_f > 0) assert(n == N_quanta_f / A_f);

    // Number of particles is not conserved
    if ((A_i - num_ats() + num_ads()) != A_f) return false;

    // Destroying more particles available is not allowed
    if (A_i < num_ats()) return false;

    // Vacuum on both side can only be connected by an identity
    // if (A_i == 0 && A_f == 0 && !identity()) return false;
    // This is already taken care of by the previous two conditions:
    // Starting with A_i == 0 && A_f == 0, 
    // case 1: ats = 0 => first condition imposes ads equal to Af,
    //                    which is 0, thus the tensor is an identity
    // case 2: ats > 0 => second condition

    auto rhomax = u3::Rho(i.u3(), T().u3(), f.u3());
    if (rhomax == 0) return false;

    auto etamax = u4::Eta(i.u4(), T().u4(), f.u4());
    if (etamax == 0) return false;

    // std::cout << "Single-shell RME exists" << std::endl;
    return true;
  }

  sam::RME SingleShellTensor::RME(const U3xU4::GammaElement& f,
                                  const U3xU4::GammaElement& i) const
  {
    auto& cgcs = utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();
    auto& wigU4U3db = u4::wigU4U3db();

    // (1) Check if this RME exists
    if (RME_exist(f, i) == false) return {};

    auto ai_max = i.alphamax();
    auto af_max = f.alphamax();

    // (2) If this is an identity tensor
    if (identity())
    {
      assert(Omega_ads_.empty());
      assert(Omega_ats_.empty());
      assert(T() == U3xU4::irrep(0,0,0, 0,0,0,0));
      assert(rho0max() == 1);
      assert(eta0max() == 1);

      if (i == f)
      {
        // Now calculate RMEs
        std::vector<double> rmes;
        for (int af = 0; af < af_max; af++)
          for (int ai = 0; ai < ai_max; ai++)
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

    // (3) If this is not an identity tensor

    int A_i = i.particles(), N_quanta_i = i.quanta(), 
        A_f = f.particles(), N_quanta_f = f.quanta(),
        n = shell();
/*
    // (3.1) Special case: <1||a+||0> ===> TAKE MORE TIME

    if (A_f == 1 && A_i == 0)
    {
      assert(num_ads() == 1);
      assert(num_ats() == 0);
      assert(af_max == 1);
      assert(ai_max == 1);

      return {{1.0}, 1, 1, 1, 1, 1};
    }

    // (3.2) Special case: <0||a||1> ===> TAKE MORE TIME

    if (A_f == 0 && A_i == 1)
    {
      assert(num_ads() == 0);
      assert(num_ats() == 1);
      assert(af_max == 1);
      assert(ai_max == 1);

      // U(3) -> U(2) -> U(1) coupling
      auto [allowU3, accU3] = wigU3U2db.get(n, 0, 0, n, 0,
                                            n, n, 0, n, 0,
                                            0, 0, 0, 0, 0);
      assert(allowU3);
      double cgcU3 = cgcs.get({n, n, 0, n, -n, 0}) * accU3(0,0);

      // U(4) -> U(3) -> U(2) -> U(1) coupling
      auto [allowU4, accU4] = wigU4U3db.get(1, 0, 0, 0, 1, 0, 0,
                                            1, 1, 1, 0, 1, 1, 0,
                                            0, 0, 0, 0, 0, 0, 0);
      assert(allowU4);

      auto [allowU3_4, accU3_4] = wigU3U2db.get(1, 0, 0, 1, 0,
                                                1, 1, 0, 1, 0,
                                                0, 0, 0, 0, 0);
      assert(allowU3_4);

      double cgcU4 = cgcs.get({1, 1, 0, 1, -1, 0}) * accU4(0,0) * accU3_4(0,0);

      return {{1.0 / (cgcU3 * cgcU4)}, 1, 1, 1, 1, 1};
    }
*/
    // (3.3) General case

    // Get outer multiplicities due to Wigner-Eckart theorem
    auto [rhomax, etamax] = U3xU4::RhoEta(i, T(), f);

    // Construct single-particle states
    u3::irrep x(n, 0, 0), x_conj(n, n, 0);
    auto sps3 = u3::gelfandBasis(x);
    auto sps3_conj = u3::gelfandBasis(x_conj);

    u4::irrep y(1, 0, 0, 0), y_conj(1, 1, 1, 0);
    auto sps4 = u4::gelfandBasis(y);
    auto sps4_conj = u4::gelfandBasis(y_conj);

    // Generate HWS for the ket irrep
    u3::gelfand i3hws(i.n13(), i.n23(), i.n33(), i.n13(), i.n23(), i.n13());
    u4::gelfand i4hws(i.n14(), i.n24(), i.n34(), i.n44(),
                      i.n14(), i.n24(), i.n34(), i.n14(), i.n24(), i.n14());
    UNxU4::SlaterDeterminants SDarr_i;
    Eigen::MatrixXd coeffs_i(1, 1);
    if (A_i == 0) // vacuum
    {
      UNxU4::SlaterDeterminant SDi(n);
      SDarr_i.push_back(std::move(SDi));
      coeffs_i(0, 0) = 1.0;
    }
    else
    {
      auto path_i = U3xU4::get_HWS_file(i, sps3, sps4);
      std::tie(SDarr_i, coeffs_i) = U3xU4::read_HWS_data(path_i);
    }
    // Check pre-calculated multiplicity is correct
    assert(ai_max == coeffs_i.cols());
    // std::cout << "coeffs_i:" << coeffs_i << std::endl;

    // Generate HWS for the bra irrep
    u3::gelfand f3hws(f.n13(), f.n23(), f.n33(), f.n13(), f.n23(), f.n13());
    u4::gelfand f4hws(f.n14(), f.n24(), f.n34(), f.n44(),
                      f.n14(), f.n24(), f.n34(), f.n14(), f.n24(), f.n14());
    UNxU4::SlaterDeterminants SDarr_f;
    Eigen::MatrixXd coeffs_f(1, 1);
    if (A_f == 0) // vacuum
    {
      UNxU4::SlaterDeterminant SDf(n);
      SDarr_f.push_back(std::move(SDf));
      coeffs_f(0, 0) = 1.0;
    }
    else
    {
      auto path_f = U3xU4::get_HWS_file(f, sps3, sps4);
      std::tie(SDarr_f, coeffs_f) = U3xU4::read_HWS_data(path_f);
    }
    // Check pre-calculated multiplicity is correct
    assert(af_max == coeffs_f.cols());
    // std::cout << "coeffs_f:" << coeffs_f << std::endl;

    // Generate tensor components of the tensor
    u3::gelfands g3Arr = u3::gelfandBasis(T().u3());
    std::vector<size_t> g3Arr0;
    int missing3 = (N_quanta_i + T().u3().quanta() - N_quanta_f) / 3;
    for (size_t i = 0; i < g3Arr.size(); i++)
      if ((i3hws.e22() + g3Arr[i].e22()) == (f3hws.e22() + missing3) &&
          (i3hws.e11() + g3Arr[i].e11()) == (f3hws.e11() + missing3))
        g3Arr0.push_back(i);

    u4::gelfands g4Arr = u4::gelfandBasis(T().u4());
    std::vector<size_t> g4Arr0;
    int missing4 = (A_i + T().u4().quanta() - A_f) / 4;
    for (size_t i = 0; i < g4Arr.size(); i++)
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
    //      <<" x " << g4Arr0.size() << std::endl;

    // Array of matricies containing full matrix elements:
    // index = alpha_i * max(alpha_f) + alpha_f
    std::vector<Eigen::MatrixXd> FMEs;
    for (int ai = 0; ai < ai_max; ai++)
    {
      for (int af = 0; af < af_max; af++)
      {
        Eigen::MatrixXd fme(g3Arr0.size() * g4Arr0.size(), rho0max() * eta0max());
        fme.setZero();
        FMEs.push_back(std::move(fme));
      }
    }

    // A matrix containing coupling coefficients
    Eigen::MatrixXd CGCs(g3Arr0.size() * g4Arr0.size(), rhomax * etamax);
    CGCs.setZero();

    // Now run through tensor components
    int row = 0;
    for (const auto& g3 : g3Arr0)
    {
      const auto& gt3 = g3Arr[g3];
      // std::cout << gt3 << std::endl;
      int jj_t3 = gt3.Lambdax2(), mm_t3 = gt3.MLambdax2();

      const auto accU3 = wigU3U2db.get(
          i3hws.n13(), i3hws.n23(), i3hws.n33(), i3hws.n12(), i3hws.n22(),
          gt3.n13(), gt3.n23(), gt3.n33(), gt3.n12(), gt3.n22(), f3hws.n13(),
          f3hws.n23(), f3hws.n33(), f3hws.n12(), f3hws.n22());
      if (accU3.size() == 0) continue;

      double cgc3 = cgcs.get({jj_i3, jj_t3, jj_f3, mm_i3, mm_t3, mm_f3});

      auto cgcU3 = accU3 * cgc3;

      // std::cout << "cgcU3 " << cgcU3 << std::endl;

      for (const auto& g4 : g4Arr0)
      {

        const auto& gt4 = g4Arr[g4];
        // std::cout << gt4 << std::endl;
        int jj_t4 = gt4.Lambdax2(), mm_t4 = gt4.MLambdax2();

        const auto accU4 =
            wigU4U3db.get(i4hws.n14(), i4hws.n24(), i4hws.n34(), i4hws.n44(),
                          i4hws.n13(), i4hws.n23(), i4hws.n33(), gt4.n14(),
                          gt4.n24(), gt4.n34(), gt4.n44(), gt4.n13(), gt4.n23(),
                          gt4.n33(), f4hws.n14(), f4hws.n24(), f4hws.n34(),
                          f4hws.n44(), f4hws.n13(), f4hws.n23(), f4hws.n33());
        if (accU4.size() == 0) continue;

        const auto accU3_4 = wigU3U2db.get(
            i4hws.n13(), i4hws.n23(), i4hws.n33(), i4hws.n12(), i4hws.n22(),
            gt4.n13(), gt4.n23(), gt4.n33(), gt4.n12(), gt4.n22(), f4hws.n13(),
            f4hws.n23(), f4hws.n33(), f4hws.n12(), f4hws.n22());
        if (accU3_4.size() == 0) continue;

        double cgc4 = cgcs.get({jj_i4, jj_t4, jj_f4, mm_i4, mm_t4, mm_f4});

        Eigen::RowVectorXd cgcU4(etamax);
        cgcU4.setZero();

        for (int e = 0; e < etamax; e++)
          for (int r = 0; r < accU4.rows(); r++)
            cgcU4(e) += accU4(r, e) * accU3_4(r) * cgc4;

        // std::cout << "cgcU4 " << cgcU4 << std::endl;

        // Elements of CGCs
        for (int r = 0; r < rhomax; r++)
          for (int e = 0; e < etamax; e++)
            CGCs(row, r * etamax + e) = cgcU3(r) * cgcU4(e);

        // Expand tensor into ad's and a's
        U3xU4::SingleShellTensor_gelfand Tadat(
            n, sps3, sps3_conj, sps4, sps4_conj, ads_, ats_, 
            Omega_ads_, Omega_ats_, T(), rho0max(), eta0max(), gt3, gt4);

        auto [coeffs, adArr, aArr] = Tadat.expand();
        /*
        for (size_t t = 0; t < coeffs.size(); t++)
        {
          std::cout << coeffs[t] << "\t";

          if (!adArr.empty())
            for (size_t j = 0; j < adArr[t].size(); j++)
              std::cout << adArr[t][j] << "\t";

          if (!aArr.empty())
            for (size_t j = 0; j < aArr[t].size(); j++)
              std::cout << aArr[t][j] << "\t";

          std::cout << std::endl;
        }
        */

        // Compute full matrix elements
        for (size_t t = 0; t < coeffs.size(); t++)
        {
          // Determine how ad's and a's act on SDi's
          for (size_t i = 0; i < SDarr_i.size(); i++)
          {
            auto SDi = SDarr_i[i];
            bool annihilated = false;

            double phase = 1;

            // Action of a's from right to left
            for (int a = num_ats() - 1; a >= 0 && !annihilated; a--)
            {
              auto [p, b] = aArr[t][a].getSDLabels();
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
            for (int ad = num_ads() - 1; ad >= 0 && !annihilated; ad--)
            {
              auto [p, b] = adArr[t][ad].getSDLabels();
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
              for (size_t f = 0; f < SDarr_f.size(); f++)
              {
                auto SDf = SDarr_f[f];
                if (SDi == SDf)
                {
                  for (int ai = 0; ai < ai_max; ai++)
                  {
                    for (int af = 0; af < af_max; af++)
                    {
                      FMEs[ai * af_max + af].row(row) += phase 
                        * coeffs_i(i, ai) * coeffs[t] * coeffs_f(f, af);
                    }
                  }
                }
              }
            }
          }
        }
        row++;
      }
    }

    // std::cout << "CGCs = " << CGCs << std::endl;

    // Solve for RMEs
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factorize;
    factorize.compute(CGCs);

    std::vector<Eigen::MatrixXd> RMEs(FMEs.size());
    for (size_t i = 0; i < FMEs.size(); i++)
    {
      // std::cout << "FME = " << FMEs[i] << std::endl;
      RMEs[i] = factorize.solve(FMEs[i]);
    }

    std::vector<double> rmes;
    for (int af = 0; af < af_max; af++)
      for (int ai = 0; ai < ai_max; ai++)
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
            static_cast<size_t>(rhomax),
            static_cast<size_t>(etamax)};
  }

  sam::RME SingleShellTensor::RME(const U3xU4::GammaElement& f,
                                  const U3xU4::GammaElement& i,
                                  const U3xU4::Table_gelfand& labels) const
  {
#ifdef TIME_SINGLE_SHELL_TENSOR_RME
    static std::chrono::duration<double> total_time(0);
    static double last_print = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
#endif

    auto& cgcs = utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();
    auto& wigU4U3db = u4::wigU4U3db();

    // (1) Check if this RME exists
    if (RME_exist(f, i) == false) return {};

    auto ai_max = i.alphamax();
    auto af_max = f.alphamax();

    // (2) If this is an identity tensor
    if (identity())
    {
      assert(Omega_ads_.empty());
      assert(Omega_ats_.empty());
      assert(T() == U3xU4::irrep(0,0,0, 0,0,0,0));
      assert(rho0max() == 1);
      assert(eta0max() == 1);

      if (i == f)
      {
        // Now calculate RMEs
        std::vector<double> rmes;
        for (int af = 0; af < af_max; af++)
          for (int ai = 0; ai < ai_max; ai++)
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

    // (3) If this is not an identity tensor

    int A_i = i.particles(), N_quanta_i = i.quanta(), 
        A_f = f.particles(), N_quanta_f = f.quanta(),
        n = shell();
/*
    // (3.1) Special case: <1||a+||0> ===> IMPLEMENTED IN DATABASE

    if (A_f == 1 && A_i == 0)
    {
      assert(num_ads() == 1);
      assert(num_ats() == 0);
      assert(af_max == 1);
      assert(ai_max == 1);

      return {{1.0}, 1, 1, 1, 1, 1};
    }

    // (3.2) Special case: <0||a||1> ===> IMPLEMENTED IN DATABASE

    if (A_f == 0 && A_i == 1)
    {
      assert(num_ads() == 0);
      assert(num_ats() == 1);
      assert(af_max == 1);
      assert(ai_max == 1);

      return {{std::sqrt(2.0 * (n + 1.0) * (n + 2.0))}, 1, 1, 1, 1, 1};
    }

    // (3.3) Special case: <2||a+ a+||0> ===> IMPLEMENTED IN DATABASE

    if (A_f == 2 && A_i == 0)
    {
      assert(num_ads() == 2);
      assert(num_ats() == 0);
      assert(af_max == 1);
      assert(ai_max == 1);

      return {{SQRT2}, 1, 1, 1, 1, 1};
    }

    // (3.4) Special case: <0||a a||2> ===> IMPLEMENTED IN DATABASE

    if (A_f == 0 && A_i == 2)
    {
      assert(num_ads() == 0);
      assert(num_ats() == 2);
      assert(af_max == 1);
      assert(ai_max == 1);

      return {{- SQRT2 * std::sqrt(i.dim())}, 1, 1, 1, 1, 1};
    }
*/
    // (3.5) General case

    // (3.5.1) Get outer multiplicities due to Wigner-Eckart theorem
    auto [rhomax, etamax] = U3xU4::RhoEta(i, T(), f);

    // (3.5.2) Extract single-particle states
    const auto& sps3 = labels.sps3(n);
    const auto& sps3_conj = labels.sps3_conj(n);
    const auto& sps4 = labels.sps4();
    const auto& sps4_conj = labels.sps4_conj();

    // (3.5.3) Generate HWS for the ket irrep
    u3::gelfand i3hws(i.n13(), i.n23(), i.n33(), i.n13(), i.n23(), i.n13());
    u4::gelfand i4hws(i.n14(), i.n24(), i.n34(), i.n44(),
                      i.n14(), i.n24(), i.n34(), i.n14(), i.n24(), i.n14());
    UNxU4::SlaterDeterminants SDarr_i;
    Eigen::MatrixXd coeffs_i(1, 1);
    if (A_i == 0) // vacuum
    {
      UNxU4::SlaterDeterminant SDi(n);
      SDarr_i.push_back(std::move(SDi));
      coeffs_i(0, 0) = 1.0;
    }
    else
    {
      auto path_i = U3xU4::get_HWS_file(i, sps3, sps4);
      std::tie(SDarr_i, coeffs_i) = U3xU4::read_HWS_data(path_i);
    }
    // Check pre-calculated multiplicity is correct
    assert(ai_max == coeffs_i.cols());
    // std::cout << "coeffs_i:" << coeffs_i << std::endl;

    // (3.5.4) Generate HWS for the bra irrep
    u3::gelfand f3hws(f.n13(), f.n23(), f.n33(), f.n13(), f.n23(), f.n13());
    u4::gelfand f4hws(f.n14(), f.n24(), f.n34(), f.n44(),
                      f.n14(), f.n24(), f.n34(), f.n14(), f.n24(), f.n14());
    UNxU4::SlaterDeterminants SDarr_f;
    Eigen::MatrixXd coeffs_f(1, 1);
    if (A_f == 0) // vacuum
    {
      UNxU4::SlaterDeterminant SDf(n);
      SDarr_f.push_back(std::move(SDf));
      coeffs_f(0, 0) = 1.0;
    }
    else
    {
      auto path_f = U3xU4::get_HWS_file(f, sps3, sps4);
      std::tie(SDarr_f, coeffs_f) = U3xU4::read_HWS_data(path_f);
    }
    // Check pre-calculated multiplicity is correct
    assert(af_max == coeffs_f.cols());
    // std::cout << "coeffs_f:" << coeffs_f << std::endl;

    // (3.5.5) Generate tensor components of the tensor
    u3::gelfands g3Arr = u3::gelfandBasis(T().u3());
    std::vector<int> g3Arr0;
    int missing3 = (N_quanta_i + T().u3().quanta() - N_quanta_f) / 3;
    for (int i = 0; i < g3Arr.size(); i++)
      if ((i3hws.e22() + g3Arr[i].e22()) == (f3hws.e22() + missing3) &&
          (i3hws.e11() + g3Arr[i].e11()) == (f3hws.e11() + missing3))
        g3Arr0.push_back(i);

    u4::gelfands g4Arr = u4::gelfandBasis(T().u4());
    std::vector<int> g4Arr0;
    int missing4 = (A_i + T().u4().quanta() - A_f) / 4;
    for (int i = 0; i < g4Arr.size(); i++)
      if ((i4hws.e33() + g4Arr[i].e33()) == (f4hws.e33() + missing4) &&
          (i4hws.e22() + g4Arr[i].e22()) == (f4hws.e22() + missing4) &&
          (i4hws.e11() + g4Arr[i].e11()) == (f4hws.e11() + missing4))
        g4Arr0.push_back(i);

    // std::cout << "Count tensor components: " << g3Arr0.size()
    //      <<" x " << g4Arr0.size() << std::endl;

    // Array of matricies containing full matrix elements:
    // index = alpha_i * max(alpha_f) + alpha_f
    std::vector<Eigen::MatrixXd> FMEs;
    for (int ai = 0; ai < ai_max; ai++)
    {
      for (int af = 0; af < af_max; af++)
      {
        Eigen::MatrixXd fme(g3Arr0.size() * g4Arr0.size(), rho0max() * eta0max());
        fme.setZero();
        FMEs.push_back(std::move(fme));
      }
    }

    // A matrix containing coupling coefficients
    Eigen::MatrixXd CGCs(g3Arr0.size() * g4Arr0.size(), rhomax * etamax);
    CGCs.setZero();

    // Get angular momenta
    int jj_i3 = i3hws.Lambdax2(), mm_i3 = i3hws.MLambdax2(),
        jj_f3 = f3hws.Lambdax2(), mm_f3 = f3hws.MLambdax2(),
        jj_i4 = i4hws.Lambdax2(), mm_i4 = i4hws.MLambdax2(),
        jj_f4 = f4hws.Lambdax2(), mm_f4 = f4hws.MLambdax2();

    // (3.5.6) Now run through tensor components
    int row = 0;
    for (const auto& g3 : g3Arr0)
    {
      const auto& gt3 = g3Arr[g3];
      // std::cout << gt3 << std::endl;
      int jj_t3 = gt3.Lambdax2(), mm_t3 = gt3.MLambdax2();

      const auto accU3 = wigU3U2db.get(
          i3hws.n13(), i3hws.n23(), i3hws.n33(), i3hws.n12(), i3hws.n22(),
          gt3.n13(), gt3.n23(), gt3.n33(), gt3.n12(), gt3.n22(), f3hws.n13(),
          f3hws.n23(), f3hws.n33(), f3hws.n12(), f3hws.n22());
      if (accU3.size() == 0) continue;

      double cgc3 = cgcs.get({jj_i3, jj_t3, jj_f3, mm_i3, mm_t3, mm_f3});

      auto cgcU3 = accU3 * cgc3;

      // std::cout << "cgcU3 " << cgcU3 << std::endl;

      for (const auto& g4 : g4Arr0)
      {
        const auto& gt4 = g4Arr[g4];
        // std::cout << gt4 << std::endl;
        int jj_t4 = gt4.Lambdax2(), mm_t4 = gt4.MLambdax2();

        const auto accU4 =
            wigU4U3db.get(i4hws.n14(), i4hws.n24(), i4hws.n34(), i4hws.n44(),
                          i4hws.n13(), i4hws.n23(), i4hws.n33(), gt4.n14(),
                          gt4.n24(), gt4.n34(), gt4.n44(), gt4.n13(), gt4.n23(),
                          gt4.n33(), f4hws.n14(), f4hws.n24(), f4hws.n34(),
                          f4hws.n44(), f4hws.n13(), f4hws.n23(), f4hws.n33());
        if (accU4.size() == 0) continue;

        const auto accU3_4 = wigU3U2db.get(
            i4hws.n13(), i4hws.n23(), i4hws.n33(), i4hws.n12(), i4hws.n22(),
            gt4.n13(), gt4.n23(), gt4.n33(), gt4.n12(), gt4.n22(), f4hws.n13(),
            f4hws.n23(), f4hws.n33(), f4hws.n12(), f4hws.n22());
        if (accU3_4.size() == 0) continue;

        double cgc4 = cgcs.get({jj_i4, jj_t4, jj_f4, mm_i4, mm_t4, mm_f4});

        Eigen::RowVectorXd cgcU4(etamax);
        cgcU4.setZero();

        for (int e = 0; e < etamax; e++)
          for (int r = 0; r < accU4.rows(); r++)
            cgcU4(e) += accU4(r, e) * accU3_4(r) * cgc4;

        // std::cout << "cgcU4 " << cgcU4 << std::endl;

        // Elements of CGCs
        for (int r = 0; r < rhomax; r++)
          for (int e = 0; e < etamax; e++)
            CGCs(row, r * etamax + e) = cgcU3(r) * cgcU4(e);

        // Expand tensor into ad's and a's
        U3xU4::SingleShellTensor_gelfand Tadat(
            n, sps3, sps3_conj, sps4, sps4_conj, ads_, ats_, 
            Omega_ads_, Omega_ats_, T(), rho0max(), eta0max(), gt3, gt4);

        auto [coeffs, adArr, aArr] = Tadat.expand();
        /*
        for (size_t t = 0; t < coeffs.size(); t++)
        {
          std::cout << coeffs[t] << "\t";

          if (!adArr.empty())
            for (size_t j = 0; j < adArr[t].size(); j++)
              std::cout << adArr[t][j] << "\t";

          if (!aArr.empty())
            for (size_t j = 0; j < aArr[t].size(); j++)
              std::cout << aArr[t][j] << "\t";

          std::cout << std::endl;
        }
        */

        // Compute full matrix elements
        for (size_t t = 0; t < coeffs.size(); t++)
        {
          // Determine how ad's and a's act on SDi's
          for (size_t i = 0; i < SDarr_i.size(); i++)
          {
            auto SDi = SDarr_i[i]; // make a copy to act on with ad's and a's
            bool annihilated = false;

            double phase = 1;

            // Action of a's from right to left
            for (int a = num_ats() - 1; a >= 0 && !annihilated; a--)
            {
              auto [p, b] = aArr[t][a].getSDLabels();
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
            for (int ad = num_ads() - 1; ad >= 0 && !annihilated; ad--)
            {
              auto [p, b] = adArr[t][ad].getSDLabels();
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
              for (size_t f = 0; f < SDarr_f.size(); f++)
              {
                if (SDi == SDarr_f[f])
                {
                  for (int ai = 0; ai < ai_max; ai++)
                  {
                    for (int af = 0; af < af_max; af++)
                    {
                      FMEs[ai * af_max + af].row(row) += phase 
                        * coeffs_i(i, ai) * coeffs[t] * coeffs_f(f, af);
                    }
                  }
                }
              }
            }
          }
        }
        row++;
      }
    }

    // std::cout << "CGCs = " << CGCs << std::endl;

    // (3.5.7) Solve for RMEs
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factorize;
    factorize.compute(CGCs);

    std::vector<Eigen::MatrixXd> RMEs(FMEs.size());
    for (int i = 0; i < FMEs.size(); i++)
    {
      // std::cout << "FME = " << FMEs[i] << std::endl;
      RMEs[i] = factorize.solve(FMEs[i]);
    }

    std::vector<double> rmes;
    for (int af = 0; af < af_max; af++)
      for (int ai = 0; ai < ai_max; ai++)
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

#ifdef TIME_SINGLE_SHELL_TENSOR_RME
    auto end = std::chrono::high_resolution_clock::now();
    total_time += end - start;  // duration in seconds

    // Print occasionally 
    if (total_time.count() - last_print >= 10.0) 
    {
        std::cout << "Accumulated time: " << total_time.count() << " s\n";
        last_print = total_time.count();
    }
#endif

    return {std::move(rmes),
            static_cast<size_t>(af_max),
            static_cast<size_t>(ai_max),
            aT_max,
            static_cast<size_t>(rhomax),
            static_cast<size_t>(etamax)};
  }
} // namespace U3xU4
