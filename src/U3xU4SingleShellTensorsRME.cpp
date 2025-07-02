#include <cassert>
#include <stdexcept>

#include <U3xU4HWSgenerator.h>
#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  std::vector<Eigen::MatrixXd>
  Tadat::RME(std::size_t& ai_max, const U3xU4::irrep& i,
             std::size_t& af_max, const U3xU4::irrep& f,
             std::size_t& rhomax, std::size_t& etamax,
             u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
             u4::WigU4U3_db& wigU4U3db)
  {
    int A_i = i.particles(), N_quanta_i = i.quanta(), // initial irreps
        A_f = f.particles(), N_quanta_f = f.quanta(); // final irreps

    if ((A_i - ats + ads) != A_f)
      throw std::invalid_argument("number of particles is not conserved");

    if (A_i == 0 && A_f == 0)
      throw std::invalid_argument("vacuum on both side");

    if (A_f == 0)
      n = N_quanta_i / A_i;
    else if (A_i == 0)
      n = N_quanta_f / A_f;
    else
    {
      n = N_quanta_i / A_i;
      if (n != N_quanta_f / A_f)
        throw std::invalid_argument("ket and bra are not in the same shell");
    }

    // Building single-particle states
    u3::irrep x(n, 0, 0), x_conj(n, n, 0);
    std::vector<u3::gelfand> sps3 = u3::gelfandBasis(x),
                             sps3_conj = u3::gelfandBasis(x_conj);

    u4::irrep y(1, 0, 0, 0), y_conj(1, 1, 1, 0);
    std::vector<u4::gelfand> sps4 = u4::gelfandBasis(y),
                             sps4_conj = u4::gelfandBasis(y_conj);

    // Generate HWS for the ket irrep
    u3::gelfand i3hws(i.u3.n13(), i.u3.n23(), i.u3.n33(), i.u3.n13(), i.u3.n23(),
                      i.u3.n13());
    u4::gelfand i4hws(i.u4.n14(), i.u4.n24(), i.u4.n34(), i.u4.n44(), i.u4.n14(),
                      i.u4.n24(), i.u4.n34(), i.u4.n14(), i.u4.n24(), i.u4.n14());
    std::vector<UNxU4::SlaterDeterminant> SDarr_i;
    std::vector<Eigen::RowVectorXd> coeffs_i;
    if (A_i == 0) // vacuum
    {
      const boost::dynamic_bitset<> bits(2 * (n + 1) * (n + 2), 0);
      UNxU4::SlaterDeterminant SDi(bits);
      SDarr_i.push_back(SDi);
      Eigen::RowVectorXd c = Eigen::RowVectorXd::Constant(1, 1.0);
      coeffs_i.push_back(c);
    }
    else
    {
      std::filesystem::path path_i =
          U3xU4::get_HWS_file(A_i, n, i, sps3, sps4);
      std::tie(SDarr_i, coeffs_i) = U3xU4::read_HWS_data(path_i);
    }
    ai_max = coeffs_i[0].cols();
    // std::cout << "coeffs_i:" << std::endl;
    // for (int i = 0; i < coeffs_i.size(); i++)
    //     std::cout << coeffs_i[i] << std::endl;

    // Generate HWS for the bra irrep
    u3::gelfand f3hws(f.u3.n13(), f.u3.n23(), f.u3.n33(), f.u3.n13(), f.u3.n23(),
                      f.u3.n13());
    u4::gelfand f4hws(f.u4.n14(), f.u4.n24(), f.u4.n34(), f.u4.n44(), f.u4.n14(),
                      f.u4.n24(), f.u4.n34(), f.u4.n14(), f.u4.n24(), f.u4.n14());
    std::vector<UNxU4::SlaterDeterminant> SDarr_f;
    std::vector<Eigen::RowVectorXd> coeffs_f;
    if (A_f == 0) // vacuum
    {
      const boost::dynamic_bitset<> bits(2 * (n + 1) * (n + 2), 0);
      UNxU4::SlaterDeterminant SDf(bits);
      SDarr_f.push_back(SDf);
      Eigen::RowVectorXd c = Eigen::RowVectorXd::Constant(1, 1.0);
      coeffs_f.push_back(c);
    }
    else
    {
      std::filesystem::path path_f =
          U3xU4::get_HWS_file(A_f, n, f, sps3, sps4);
      std::tie(SDarr_f, coeffs_f) = U3xU4::read_HWS_data(path_f);
    }
    af_max = coeffs_f[0].cols();
    // std::cout << "coeffs_f:" << std::endl;
    // for (int i = 0; i < coeffs_f.size(); i++)
    //     std::cout << coeffs_f[i] << std::endl;

    // Get outer multiplicities due to Wigner-Eckart theorem
    rhomax = u3::Rho(i.u3, T.u3, f.u3);
    etamax = u4::Eta(i.u4, T.u4, f.u4);

    if (rhomax == 0)
      throw std::invalid_argument("i3 x t3 does not produce f3");
    if (etamax == 0)
      throw std::invalid_argument("i4 x t4 does not produce f4");

    // Array of matricies containing full matrix elements:
    // index = alpha_i * max(alpha_f) + alpha_f
    std::vector<Eigen::MatrixXd> FMEs;
    for (std::size_t ai = 0; ai < ai_max; ai++)
    {
      for (std::size_t af = 0; af < af_max; af++)
      {
        Eigen::MatrixXd fme(rhomax * etamax, rho0max * eta0max);
        fme.setZero();
        FMEs.push_back(fme);
      }
    }

    // A matrix containing coupling coefficients
    Eigen::MatrixXd CGCs(rhomax * etamax, rhomax * etamax);
    CGCs.setZero();

    // Generate tensor components of the tensor
    std::vector<u3::gelfand> g3Arr = u3::gelfandBasis(T.u3);
    std::vector<std::size_t> g3Arr0;
    int missing3 = (i.u3.n13() + i.u3.n23() + i.u3.n33() + 
                    T.u3.n13() + T.u3.n23() + T.u3.n33() - 
                    f.u3.n13() - f.u3.n23() - f.u3.n33()) / 3;

    for (std::size_t i = 0; i < g3Arr.size(); i++)
      if ((i3hws.e22() + g3Arr[i].e22()) == (f3hws.e22() + missing3) &&
          (i3hws.e11() + g3Arr[i].e11()) == (f3hws.e11() + missing3))
        g3Arr0.push_back(i);

    std::vector<u4::gelfand> g4Arr = u4::gelfandBasis(T.u4);
    std::vector<std::size_t> g4Arr0;
    int missing4 =
        (i.u4.n14() + i.u4.n24() + i.u4.n34() + i.u4.n44() + 
         T.u4.n14() + T.u4.n24() + T.u4.n34() + T.u4.n44() - 
         f.u4.n14() - f.u4.n24() - f.u4.n34() - f.u4.n44()) / 4;

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
        U3xU4::Tadat_gelfand Tadat(n, sps3, sps3_conj, sps4, sps4_conj, ads,
                                   ats, prod_ads, prod_ats, T, rho0max, eta0max, gt3, gt4);

        auto [coeffs, adArr, aArr] = Tadat.expand(cgcs, wigU3U2db, wigU4U3db);
        /*
                                        for (std::size_t t = 0; t <
           coeffs.size(); t++)
                                        {
                                        std::cout << coeffs[t] << "\t";

                                        if (!adArr.empty())
                                        for (std::size_t j = 0; j <
           adArr[t].size(); j++)
                                        {
                                                adArr[t][j].print();
                                                std::cout << "\t";
                                        }

                                        if (!aArr.empty())
                                        for (std::size_t j = 0; j <
           aArr[t].size(); j++)
                                        {
                                                aArr[t][j].print();
                                                std::cout << "\t";
                                        }

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
            for (std::size_t f = 0; f < SDarr_f.size(); f++)
            {
              auto SDf = SDarr_f[f];
              if (SDi == SDf && !annihilated)
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
      for (int r = 0; r < RMEs[i].rows(); r++)
        for (int c = 0; c < RMEs[i].cols(); c++)
          if (std::fabs(RMEs[i](r,c)) < 1e-10)
            RMEs[i](r,c) = 0.0;
    }

    return RMEs;
  }
} // namespace U3xU4