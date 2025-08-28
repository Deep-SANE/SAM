#include <U3xU4MultiShellTensors.h>

namespace U3xU4
{
  // A function to couple single-shell tensors into multi-shell tensors
  // using CouplingPath = std::pair<
  //  std::vector<U3xU4::Tadat>,            // list of selected single-shell
  //  tensors std::vector<std::tuple<U3xU4::irrep, int, int>>>;  // list of all
  //  coupled (irrep,rhomax,etamax)

  void CoupleSingShellTensors(
      const std::vector<std::vector<U3xU4::SingleShellTensor>>&
          Tadat_lists,                          // list of lists of Tadat
      std::size_t depth,                        // recursion depth
      std::vector<U3xU4::SingleShellTensor> Tadat_choices,  // current choices
      std::vector<U3xU4::irrep> coupled_irreps, // current coupled irreps
      std::vector<int> rhomax_list,             // rhomax values
      std::vector<int> etamax_list,             // etamax values
      std::vector<CouplingPath>& all_results)   // final results
  {
    // If we've chosen one Tadat from every list, we stop and store the path
    if (depth == Tadat_lists.size())
    {
      std::vector<std::tuple<U3xU4::irrep, int, int>> coupling_data;
      for (std::size_t i = 0; i < coupled_irreps.size(); ++i)
        coupling_data.emplace_back(coupled_irreps[i], rhomax_list[i],
                                   etamax_list[i]);

      all_results.emplace_back(Tadat_choices, coupling_data);
      return;
    }

    // Access the depth-th list to process its possible Tadat
    const auto& current_list = Tadat_lists[depth];

    // Start with the first list
    if (Tadat_choices.empty())
    {
      for (const auto& Tadat : current_list)
        CoupleSingShellTensors(Tadat_lists, depth + 1, {Tadat}, {}, {}, {},
                               all_results);
    }
    else
    {
      for (auto& Tadat : current_list)
      {
        auto prev_irrep = coupled_irreps.empty() ? Tadat_choices.back().T
                                                 : coupled_irreps.back();
        auto prev_rhomax = rhomax_list.empty() ? Tadat_choices.back().rho0max
                                               : rhomax_list.back();
        auto prev_etamax = etamax_list.empty() ? Tadat_choices.back().eta0max
                                               : etamax_list.back();

        auto [new_irreps, rhomaxs, etamaxs] =
            U3xU4::decomposition(prev_irrep, Tadat.T);

        for (std::size_t i = 0; i < new_irreps.size(); i++)
        {
          auto new_Tadat_choices = Tadat_choices;
          auto new_coupled_irreps = coupled_irreps;
          auto new_rhomax_list = rhomax_list;
          auto new_etamax_list = etamax_list;

          new_Tadat_choices.push_back(Tadat);
          new_coupled_irreps.push_back(new_irreps[i]);
          new_rhomax_list.push_back(prev_rhomax * rhomaxs[i]);
          new_etamax_list.push_back(prev_etamax * etamaxs[i]);

          CoupleSingShellTensors(Tadat_lists, depth + 1, new_Tadat_choices,
                                 new_coupled_irreps, new_rhomax_list,
                                 new_etamax_list, all_results);
        }
      }
    }
  }
} // namespace U3xU4
