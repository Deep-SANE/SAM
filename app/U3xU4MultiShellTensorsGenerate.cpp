#include <iostream>

#include <U3xU4MultiShellTensors.h>
#include <U3xU4SingleShellTensors.h>

int main()
{
  std::vector<std::array<int, 3>> groups = {{1, 0, 1},
											                      {5, 1, 1},
											                      {8, 1, 0}};

  std::vector<std::vector<U3xU4::Tadat>> Tadat_lists;

  for (auto group : groups)
  {
    Tadat_lists.push_back(U3xU4::GenerateSingleShellTensors(
          std::get<0>(group), std::get<1>(group), std::get<2>(group)));
  }

  std::vector<U3xU4::CouplingPath> all_results;
  U3xU4::CoupleSingShellTensors(Tadat_lists, 0, {}, {}, {}, {}, all_results);

  for (const auto& [Tadat_choices, coupling_info_list] : all_results)
  {
    for (const auto& Tadat : Tadat_choices) 
      std::cout << Tadat << " ";

    std::cout << "| coupled irreps: ";
    for (const auto& [irrep, rhomax, etamax] : coupling_info_list) 
      std::cout << "rhoMax" << rhomax << "etaMax" << etamax << irrep << " ";

    std::cout << std::endl;
  }

  return 0;
}