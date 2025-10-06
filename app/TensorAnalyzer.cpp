#include <TensorAnalyzer.h>

int main()
{
  // auto tensors_stats = sam::U4_tensor_analyzer("N2LO_opt_bare");
  auto tensors_stats = sam::U4_tensor_analyzer("N2LO_opt_srg2.0");

  for (const auto& tensor_stat : tensors_stats)
    std::cout << tensor_stat.first << ":\t" << tensor_stat.second << std::endl;

  return 0;
}
