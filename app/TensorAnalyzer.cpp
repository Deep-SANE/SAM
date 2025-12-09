#include <TensorAnalyzer.h>

int main()
{
  /* Choose one:
    N4LO_EMN500_bare
    N3LO_EMN500_bare
    N2LO_opt_bare
    CoMNumberOp
    Kinetic
  */
  std::string name = "N4LO_EMN500_bare";
  auto tensors_U3_stats = sam::U3_tensor_analyzer_write(name);
  std::cout << "U3 tensor characters:" << std::endl;
  for (const auto& [irrep, strength] : tensors_U3_stats)
    std::cout << irrep << ":\t" << strength << std::endl;

  auto tensors_U4_stats = sam::U4_tensor_analyzer_write(name);
  std::cout << "U4 tensor characters:" << std::endl;
  for (const auto& [irrep, strength] : tensors_U4_stats)
    std::cout << irrep << ":\t" << strength << std::endl;

  return 0;
}
