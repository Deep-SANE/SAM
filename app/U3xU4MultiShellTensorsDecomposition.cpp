#include <chrono>
#include <iostream>
#include <utility>

#include <Databases.h>
#include <U3xU4irrep.h>
#include <U3xU4MultiShellTensors.h>

#ifdef USE_SLEPC
#include <slepcsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;

  // Example 1
  /*
  U3xU4::irrep Gamma1(2,1,0, 2,1,1,0), Gamma2(2,2,0, 1,1,1,0),
        Gamma3(3,0,0, 1,0,0,0), Omega1(4,3,0, 3,2,2,0), Omega2(7,3,0, 4,2,2,0);

  U3xU4::SingleShellTensor tadat1(1,1,1,{},{},Gamma1,1,1),
                           tadat2(2,0,1,{},{},Gamma2,1,1),
                           tadat3(3,1,0,{},{},Gamma3,1,1);

  std::vector<U3xU4::SingleShellTensor> 
    single_shell_tensors = {tadat1,tadat2,tadat3};

  std::vector<U3xU4::irrep> coupled_irreps = {Omega1,Omega2};

  HO::Table_nljmjmt labels(3);
  U3xU4::TwoBodyMultiShellTensor myTensor(single_shell_tensors,
    coupled_irreps,1,1,1,4,1,4,0,0,0,0,{},labels);
  */

  // Example 2

  U3xU4::irrep Gamma1(0, 0, 0, 1, 0, 0, 0), Gamma2(1, 1, 0, 1, 1, 1, 0),
      Omega1(1, 1, 0, 2, 1, 1, 0), Gamma3(2, 2, 0, 1, 1, 1, 0),
      Omega2(3, 3, 0, 3, 2, 2, 0), Gamma4(3, 0, 0, 1, 0, 0, 0),
      Omega3(6, 3, 0, 4, 2, 2, 0);

  U3xU4::SingleShellTensor 
      tadat1(0, 1, 0, {}, {}, Gamma1, 1, 1),
      tadat2(1, 0, 1, {}, {}, Gamma2, 1, 1),
      tadat3(2, 0, 1, {}, {}, Gamma3, 1, 1),
      tadat4(3, 1, 0, {}, {}, Gamma4, 1, 1);

  std::vector<U3xU4::SingleShellTensor> 
    single_shell_tensors = {tadat1, tadat2, tadat3, tadat4};

  std::vector<U3xU4::irrep> coupled_irreps = {Omega1, Omega2, Omega3};

  HO::Table_nljmjmt labels(3);
  U3xU4::TwoBodyMultiShellTensor myTensor(single_shell_tensors, coupled_irreps,
                                          1, 1, 1, 4, 1, 4, 0, 0, 0, 0, {},
                                          labels);

  // Example 3
  /*
  U3xU4::irrep Gamma1(0, 0, 0, 3, 1, 1, 0), Gamma2(1, 1, 0, 1, 1, 1, 0),
      Omega1(1, 1, 0, 3, 1, 0, 0);
  U3xU4::SingleShellTensor 
    tadat1(0, 2, 1, {U3xU4::irrep(0, 0, 0, 2, 0, 0, 0)}, {}, Gamma1, 1, 1),
    tadat2(1, 0, 1, {}, {}, Gamma2, 1, 1);

  std::vector<U3xU4::SingleShellTensor> single_shell_tensors = {tadat1, tadat2};

  std::vector<U3xU4::irrep> coupled_irreps = {Omega1};

  HO::Table_nljmjmt labels(1);
  U3xU4::TwoBodyMultiShellTensor myTensor(single_shell_tensors, coupled_irreps,
                                          1, 1, 1, 2, 1, 2, 0, 0, 0, 0, {},
  labels);
  */

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto [coeffsArr, opsArr] = myTensor.expand(dbs);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  Eigen::RowVectorXd norm(myTensor.rhomax * myTensor.etamax *
                          myTensor.kappamax * myTensor.zetamax);
  norm.setZero();

  for (std::size_t i = 0; i < coeffsArr.size(); i++)
  {
    std::cout << coeffsArr[i] << "\t";
    norm = norm + coeffsArr[i].cwiseProduct(coeffsArr[i]);

    for (const auto& op : opsArr[i])
      std::visit([](auto&& o) { std::cout << o << "\t"; }, op);

    std::cout << std::endl;
  }

  std::cout << "Normalization: " << norm << std::endl;

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

#ifdef USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}
