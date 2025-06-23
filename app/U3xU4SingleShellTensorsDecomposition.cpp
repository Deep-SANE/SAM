#include <iostream>
#include <utility>

#include <U3basis.h>
#include <U4basis.h>

#include <U3xU4SingleShellTensorsDecomposition.h>

#ifdef SAM_USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef SAM_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  u34utils::SO3_CGC cgcs;
  u3::WigU3U2_db wigU3U2db(cgcs);
  u4::WigU4U3_db wigU4U3db(cgcs, wigU3U2db);

  std::vector<std::pair<u3::irrep, u4::irrep>> prod_ads, prod_ats;

  /* Example 1: n = 2
                  (3,1,0)       (3,3,0)             (3,2,0)    (4,1,0; 4,0; 2)
      { [[ad x ad]         x ad]          x [at x at]         }
                  (1,1,0,0)     (2,1,0,0)           (1,1,0,0)  (1,0,0,0; 1,0,0; 1,0; 0)
  */
  /*
  std::size_t n = 2, ads = 3, ats = 2;

  u3::irrep B1_0(3, 1, 0), B1(3, 3, 0), B2(3, 2, 0), B(4, 1, 0);
  u3::gelfand b(4,1,0, 4,0, 2);

  u4::irrep P1_0(1, 1, 0, 0), P1(2, 1, 0, 0), P2(1, 1, 0, 0), P(1, 0, 0, 0);
  u4::gelfand p(1,0,0,0, 1,0,0, 1,0, 0);

  prod_ads.emplace_back(B1_0, P1_0);
  prod_ads.emplace_back(B1, P1);
  prod_ats.emplace_back(B2, P2);
  */ 

  /* Example 2: n = 1
        (1,0,0)       (1,1,0)    (0,0,0; 0,0; 0)
    { ad          x at          }
        (1,0,0,0)     (1,1,1,0)  (0,0,0,0; 0,0,0; 0,0; 0)
  */
  /*
  std::size_t n = 1, ads = 1, ats = 1;

  u3::irrep B1(1,0,0), B2(1,1,0), B(0,0,0);
  u3::gelfand b(0,0,0, 0,0, 0);

  u4::irrep P1(1,0,0,0), P2(1,1,1,0), P(0,0,0,0);
  u4::gelfand p(0,0,0,0, 0,0,0, 0,0, 0);
  */
  
  /* Example 3: n = 3
        (3,0,0)              (3,0,0)   (6,0,0; 6,0; 3)
    { ad          x [at x at]         }
        (1,0,0,0)            (1,1,0,0) (2,1,0,0; 1,1,0; 1,0; 0)
  */

  std::size_t n = 3, ads = 1, ats = 2;

  u3::irrep B1(3,0,0), B2(3,0,0), B(6,0,0);
  u3::gelfand b(6,0,0, 6,0, 3);

  u4::irrep P1(1,0,0,0), P2(1,1,0,0), P(2,1,0,0);
  u4::gelfand p(2,1,0,0, 1,1,0, 1,0, 0);

  prod_ats.emplace_back(B2,P2);

  // Construct single particle states
  u3::irrep x(n, 0, 0), x_conj(n, n, 0);
  std::vector<u3::gelfand> sps3 = u3::gelfandBasis(x),
                           sps3_conj = u3::gelfandBasis(x_conj);

  u4::irrep y(1, 0, 0, 0), y_conj(1, 1, 1, 0);
  std::vector<u4::gelfand> sps4 = u4::gelfandBasis(y),
                           sps4_conj = u4::gelfandBasis(y_conj);

  // Multiplicities of the final tensor
  auto rhomax = u3::Rho(B1, B2, B);
  auto etamax = u4::Eta(P1, P2, P);

  if (!b.check() || !p.check() || b.n13() != B.n13() || b.n23() != B.n23() ||
      b.n33() != B.n33() || p.n14() != P.n14() || p.n24() != P.n24() ||
      p.n34() != P.n34() || p.n44() != P.n44())
  {
    std::cout << "Tensor components invalid!";
    return 1;
  }

  U3xU4::Tadat_gelfand Tadat(n, sps3, sps3_conj, sps4, sps4_conj, ads, ats, 
                             prod_ads, prod_ats, B, b, rhomax, P, p, etamax);

  auto [coeffs, adArr, aArr] = Tadat.expand(cgcs, wigU3U2db, wigU4U3db);

  Eigen::RowVectorXd norm(rhomax * etamax);
  norm.setZero();

  for (std::size_t i = 0; i < coeffs.size(); i++)
  {
    std::cout << coeffs[i] << "\t";
    norm = norm + coeffs[i].cwiseProduct(coeffs[i]);

    if (!adArr.empty())
      for (std::size_t j = 0; j < adArr[i].size(); j++)
      {
        adArr[i][j].print();
        std::cout << "\t";
      }

    if (!aArr.empty())
      for (std::size_t j = 0; j < aArr[i].size(); j++)
      {
        assert(!aArr[i][j].isTensor());
        aArr[i][j].print();
        std::cout << "\t";
      }

    std::cout << std::endl;
  }

  std::cout << "Normalization: " << norm << std::endl;

#ifdef SAM_USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}