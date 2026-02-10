#include <iostream>

#include <sam.h>

#include <U3xU4MultiShellTensors.h>
#include <U3xU4SingleShellTensors.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  std::vector<std::array<int, 3>> groups = {{0, 2, 1}, {1, 0, 1}, {8, 1, 0}};

  std::vector<std::vector<U3xU4::SingleShellTensor>> Tadat_lists;

  for (auto group : groups)
  {
    Tadat_lists.push_back(U3xU4::GenerateSingleShellTensors(
        std::get<0>(group), std::get<1>(group), std::get<2>(group)));
  }

  std::vector<U3xU4::CouplingPath> all_results;
  U3xU4::GenerateMultiShellTensors(Tadat_lists, 0, {}, {}, {}, {}, all_results);

  for (const auto& [Tadat_choices, coupling_info_list] : all_results)
  {
    for (const auto& Tadat : Tadat_choices)
      std::cout << Tadat << " ";

    std::cout << "| coupled irreps: ";
    for (const auto& path : coupling_info_list)
      std::cout << "rhoMax" << path.rhomax() << "etaMax" << path.etamax() << path << " ";

    std::cout << std::endl;
  }

  sam::finalize();
}
