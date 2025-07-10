#include <cassert>
#include <cmath>
#include <tuple>

#include <TwoBodyOperatorJT.h>

namespace JT
{
  // ME: / na lla jja mmja mmta | O | nc llc jjc mmjc mmtc \
    // 	   \ nb llb jjb mmjb mmtb |	  | nd lld jjb mmjd mmtd /
  double TwoBodyOperator::sphericalME(int na, int lla, int jja, int mmja,
                                      int mmta, int nb, int llb, int jjb,
                                      int mmjb, int mmtb, int nc, int llc,
                                      int jjc, int mmjc, int mmtc, int nd,
                                      int lld, int jjd, int mmjd, int mmtd)
  {
    double x = 0.0;
    switch (type)
    {
    case 1: // Two-body part of L^2

      break;
    case 2: // Two-body part of L.S

      break;
    case 3: // Two-body part of CoM number operator

      break;
    default:
      throw std::invalid_argument("This operator has not been implemented");
    }
  }

  // Generate a list of al {na+1, nb+1, -nc-1, -nd-1} combinations such that
  // 0 <= na, nb, nc, nd <= maxShell
  std::vector<std::array<int, 4>> TwoBodyOperator::GenerateAllCombinationsHOShells
    	(int maxShell)
  {
  	std::vector<std::array<int, 4>> na_nb_nc_nd;
	for (int na = 0; na <= maxShell; na++)
	{
		for (int nb = 0; nb <= maxShell; nb++)
		{
			for (int nc = 0; nc <= maxShell; nc++)
			{
				for (int nd = 0; nd <= maxShell; nd++)
				{
					//std::cout << na << "," << nb << "," << nc << "," << nd << std::endl;
					std::array<int, 4> ns = {na+1, nb+1, -nc-1, -nd-1};
					na_nb_nc_nd.push_back(ns);
				}
			}
		}
	}
	return na_nb_nc_nd;
  }

  // Restrict: 
  // 0 <= (na - valenceShell + (nb - valenceShell) <= Nmax
  // 0 <= (nc - valenceShell + (nd - valenceShell) <= Nmax
  std::vector<std::array<int, 4>> TwoBodyOperator::GenerateAllCombinationsHOShells
    	(int Nmax, int valenceShell, int maxShell)
  {
  	std::vector<std::array<int, 4>> na_nb_nc_nd;
	for (int na = 0; na <= maxShell; na++)
	{
		for (int nb = 0; nb <= maxShell; nb++)
		{
			int adxad_Nhw = 0;
			if (na > valenceShell)
				adxad_Nhw += (na - valenceShell);
			if (nb > valenceShell)
				adxad_Nhw += (nb - valenceShell);
			if (adxad_Nhw > Nmax)
				continue;

			for (int nc = 0; nc <= maxShell; nc++)
			{
				for (int nd = 0; nd <= maxShell; nd++)
				{
					int atxat_Nhw = 0;
					if (nc > valenceShell)
						atxat_Nhw += (nc - valenceShell);
					if (nd > valenceShell)
						atxat_Nhw += (nd - valenceShell);
					if (atxat_Nhw > Nmax)
						continue;

					//std::cout << na << "," << nb << "," << nc << "," << nd << std::endl;

					std::array<int, 4> ns = {na+1, nb+1, -nc-1, -nd-1};
					na_nb_nc_nd.push_back(ns);
				}
			}
		}
	}
	return na_nb_nc_nd;
  }

  // Perform permutation of tensor in increasing shell.
  // For tensors of the same shell, ad's are in the front of at's.
  // Exchange of 2 tensors always incurs a phase of -1.0.
  // No exchange of ad and a of the same shell occurs.
  static std::tuple<std::array<int, 4>, std::vector<int>, int>
  	sort_shells(const std::array<int, 4>& original)
  {
  	// index vector
    std::vector<int> indices = {0, 1, 2, 3};

    // sort indices based on absolute value of original
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
      return std::abs(original[i]) < std::abs(original[j]);
    });

    // build sorted array
    std::array<int, 4> sorted;
    for (int i = 0; i < 4; i++)
      sorted[i] = original[indices[i]];

    // build inverse permutation
    std::vector<int> inverse_indices(4);
    for (int i = 0; i < 4; i++)
      inverse_indices[indices[i]] = i;

    // compute permutation sign using cycle decomposition
    int sign = 1;
    std::vector<bool> visited(4, false);
    for (int i = 0; i < 4; i++) 
    {
      if (visited[i]) 
      	continue;
      int cycle_len = 0, j = i;
      while (!visited[j]) 
      {
        visited[j] = true;
        j = indices[j];
        cycle_len++;
      }
      if (cycle_len > 1 && (cycle_len % 2 == 0))
        sign *= -1;
    }

    return {sorted, inverse_indices, sign};
  }

  // Restores the original ordering using inverse permutation
  static std::array<int, 4> inverse_permutation(
  	const std::array<int, 4>& sorted, const std::vector<int>& inverse) 
  {
    std::array<int, 4> restored;
    for (int i = 0; i < 4; ++i)
      restored[i] = sorted[inverse[i]];
    return restored;
  }

  // Group tensors of the same shell
  // Results are contained in a list of 3-element arrays:
  // first: shell number
  // second: number of ad's
  // thrid: number of at's
  static std::vector<std::array<int, 3>> group_same_shell(
  	const std::array<int, 4>& sorted)
  {
  	std::vector<std::array<int, 3>> groups;
  	for (int i = 0; i < 4; i++)
  	{
  		int shell = std::abs(sorted[i]), ads = 0, ats = 0, j;
  		for (j = i; j < 4; j++)
  		{
  			if (sorted[j] == shell) 
  				ads++;
  			else if (sorted[j] == -shell) 
  				ats++;
  			else 
  				break;
  		}
  		std::array<int, 3> group = {shell-1, ads, ats};
  		groups.push_back(group);
  		i = j-1;
  	}
  	return groups;
  }

#define RESTRICT_WITH_NMAX
  std::vector<U3xU4::TwoBodyMultiShellTensor>
    TwoBodyOperator::expand(int Nmax, int valenceShell, int maxShell, 
    						u34utils::SO3_CGC& cgcs, 
    						u34utils::SO3_Racah9& racah9, 
    						u3::WigU3U2_db& wigU3U2db, 
    						u4::WigU4U3_db& wigU4U3db)
  {
  	std::vector<U3xU4::TwoBodyMultiShellTensor> tensors;

#ifdef RESTRICT_WITH_NMAX
  	auto all_na_nb_nc_nd = GenerateAllCombinationsHOShells(Nmax,valenceShell,maxShell);
#else
  	auto all_na_nb_nc_nd = GenerateAllCombinationsHOShells(maxShell);
#endif

  	for (int i = 0; i < all_na_nb_nc_nd.size(); i++)
  	{
  		auto [sorted_ns, inverse, global_phase] = sort_shells(all_na_nb_nc_nd[i]);
  		//std::cout << global_phase;
  		for (auto n : sorted_ns) std::cout << " " << n;
  		std::cout << std::endl;

  		auto groups_ns = group_same_shell(sorted_ns);
      std::vector<std::vector<U3xU4::Tadat>> Tadat_lists;

  		//std::cout << "Group tensors: " << std::endl;
  		for (auto group : groups_ns)
  		{
  			//for (auto y : group) std::cout << y << " ";
  			//std::cout << std::endl;

        Tadat_lists.push_back(U3xU4::GenerateSingleShellTensors(
          std::get<0>(group), std::get<1>(group), std::get<2>(group)));
  		}

      std::vector<U3xU4::CouplingPath> all_results;
      U3xU4::CoupleSingShellTensors(Tadat_lists, 0, {}, {}, {}, {}, all_results);

      for (const auto& [Tadat_choices, coupling_info_list] : all_results)
      {
        //for (const auto& Tadat : Tadat_choices) std::cout << Tadat << " ";

        //std::cout << "| coupled irreps: ";
        std::vector<U3xU4::irrep> coupled_irreps;
        int Rhomax, Etamax;
        for (const auto& [irrep, rhomax, etamax] : coupling_info_list) 
        {
          //std::cout << "rhoMax" << rhomax << "etaMax" << etamax << irrep << " ";
          assert(rhomax == 1);
          assert(etamax == 1);
          coupled_irreps.push_back(irrep);
          Rhomax = rhomax;
          Etamax = etamax;
        }

        //std::cout << std::endl;

        // decompose into L
        std::vector<u3::physicalvector> pv3s;
        // decompose into 2S and 2T
        std::vector<u4::physicalvector> pv4s;
        
        if (coupled_irreps.empty()) // Tadat_choices has one element only
        {
          pv3s = u3::physicalBasis(Tadat_choices[0].T.u3);
          pv4s = u4::physicalBasis(Tadat_choices[0].T.u4);
        }
        else 
        {
          pv3s = u3::physicalBasis(coupled_irreps.back().u3);
          pv4s = u4::physicalBasis(coupled_irreps.back().u4);
        }
          
        for (int pv3 = 0; pv3 < pv3s.size(); pv3++)
        {
          int kappa0 = pv3s[pv3].kappa, LL0 = 2 * pv3s[pv3].L;

          for (int pv4 = 0; pv4 < pv4s.size(); pv4++)
          {
            int zeta0 = pv4s[pv4].zeta, SS0 = pv4s[pv4].SS, TT0 = pv4s[pv4].TT;
              
            // couple LL and SS to JJ
            if (JJ < std::fabs(LL0 - SS0) || JJ > (LL0 + SS0) || TT0 != TT)
              continue;

            // compute the expansion coefficients ... TBC
            bool ignore = false;
            std::vector<double> coeff;

            if (ignore == false)
            {
              if (coupled_irreps.empty())
              {
                U3xU4::TwoBodyMultiShellTensor tensor(Tadat_choices, coupled_irreps, 
                  Tadat_choices[0].rho0max, Tadat_choices[0].eta0max, 
                  kappa0, LL0, zeta0, SS0, TT, MMT, JJ, MMJ, coeff);
                tensors.push_back(tensor);
              }
              else
              {
                U3xU4::TwoBodyMultiShellTensor tensor(Tadat_choices, coupled_irreps, 
                  Rhomax, Etamax, kappa0, LL0, zeta0, SS0, TT, MMT, JJ, MMJ, coeff);
                tensors.push_back(tensor);
              }
            }
          }
        }
      }
  	}

  	return tensors;
  }
} // namespace JT