#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include <U4basis.h>
#include <UNtoU3.h>
#include <UNxU4SlaterDeterminant.h>

void generate_U4_irreps(int A, int n, std::vector<u4::irrep>& u4_irreps)
{
  int nsps = (n + 1) * (n + 2) / 2;
  if (A > 4 * nsps)
    return;
  for (int n14 = std::min(A, nsps); n14 >= 0; --n14)
  {
    for (int n24 = n14; n24 >= 0; --n24)
    {
      if (n14 + n24 > A)
        continue;
      for (int n34 = n24; n34 >= 0; --n34)
      {
        if (n14 + n24 + n34 > A)
          continue;
        int n44 = A - (n14 + n24 + n34);
        if (n44 <= n34 && n44 >= 0)
          u4_irreps.push_back(u4::irrep(n14, n24, n34, n44));
      }
    }
  }
}

std::vector<int> UN_to_UN(const std::vector<size_t>& un5)
{
  int N = std::reduce(un5.begin(), un5.end());
  std::vector<int> un(N, 0);

  size_t index = 0;
  for (size_t i = 0; i <= 4; ++i)
  {
    int A = 4 - i;
    // n ... number of labels with A particles
    int n = un5[i];
    for (size_t j = 1; j <= n; ++j)
      un[index++] = A;
  }
  return un;
}

int main()
{
  int n;
  std::cout << "HO shell number:" << std::endl;
  std::cin >> n;
  int N = (n + 1) * (n + 2) / 2;
  
  int A;
  std::cout << "Enter number of nucleons (max:" << 2 * (n + 1) * (n + 2)
            << "):" << std::endl;
  std::cin >> A;


  std::vector<u4::irrep> u4_irreps;
  generate_U4_irreps(A, n, u4_irreps);
  for (auto u4 : u4_irreps)
  {
    // find conjugate to u4 irrep
    std::vector<std::size_t> un5 = UNxU4::U4_to_UN(N, u4);
    std::vector<int> un = UN_to_UN(un5);

    std::cout << "[";
    for (size_t i = 0; i < un.size() - 1; ++i)
      std::cout << un[i] << ",";
    std::cout << un.back() << "] x ";

    std::cout << "[" << u4.n14() << "," << u4.n24() << "," << u4.n34() << ","
              << u4.n44() << "]" << std::endl;
    untou3::UNtoU3<4> gen;
    gen.generateXYZ(n);

    gen.generateU3Weights({static_cast<short>(un5[0]), static_cast<short>(un5[1]),
                           static_cast<short>(un5[2]), static_cast<short>(un5[3]),
                           static_cast<short>(un5[4])},
                          N);

    int multi;
    for (const auto& pair : gen.multMap())
    {
    	const auto& weight = pair.first;
    	multi = gen.getLevelDimensionality(weight);
      if (multi > 0)
      { 
        int lambda = weight[0] - weight[1];
        int mu = weight[1] - weight[2];
        // std::cout << "\t" << weight[0] << " " << weight[1] << " " << weight[2] << "\t" << "mult:" << multi << std::endl;
        std::cout << "\t(" << lambda << " " << mu << ")" << "\t" << "mult:" << multi << std::endl;
      }
    }
  }
}
