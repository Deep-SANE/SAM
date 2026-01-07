#include <iostream>

#include <qcx_utils.h>

#include <U3xU4SingleShellTensors.h>

int main(int argc, char* argv[])
{
  qcx_utils::initialize(&argc, &argv);

  int n = 2, num_ads = 1, num_ats = 3;

  std::vector<U3xU4::SingleShellTensor> tensors =
      U3xU4::GenerateSingleShellTensors(n, num_ads, num_ats);

  for (auto t : tensors)
    std::cout << t << std::endl;

  qcx_utils::finalize();
}
