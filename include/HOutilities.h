#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <stdexcept>

#include <so3Coeffs/SO3_Racah.h>

namespace std 
{
    template <>
    struct hash<std::tuple<short, short, short, short, short>> 
    {
        size_t operator()(const std::tuple<short, short, short, short, short>& key) 
        const noexcept 
        {
            size_t seed = 0;
            auto hasher = std::hash<short>();
            seed ^= hasher(std::get<0>(key)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(std::get<1>(key)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(std::get<2>(key)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(std::get<3>(key)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(std::get<4>(key)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}

namespace HO
{
	class Table_nljmjmt 
	{
	private:
		// key = (n, ll, jj, mmj, mmt)
		using key_type = std::tuple<int16_t, int16_t, int16_t, int16_t, int16_t>; 

    	std::unordered_map<key_type, int> state_to_index;
    	std::vector<key_type> index_to_state;
    	int dim;
	public:
		// Constructor
    	Table_nljmjmt(int maxShell) 
    	{
        	int index = 0;
        	for (int n = 0; n <= maxShell; n++) 
            	for (int l = n; l >= 0; l -= 2) 
                	for (int jj = 2 * l + 1; jj >= std::abs(2 * l - 1); jj -= 2)  
                    	for (int mmj = jj; mmj >= -jj; mmj -= 2) 
                        	for (int mmt = 1; mmt >= -1; mmt -= 2)
                        	{
                            	key_type key = {n, 2 * l, jj, mmj, mmt};
                            	state_to_index[key] = index;
                            	index_to_state.push_back(key);
                            	index++;
                        	}
            dim = index_to_state.size();
       	}

    	// Get index from quantum numbers
    	int getIndex(int n, int ll, int jj, int mmj, int mmt) const 
    	{
        	key_type key = {n, ll, jj, mmj, mmt};
        	auto it = state_to_index.find(key);
        	if (it == state_to_index.end())
            	throw std::out_of_range("Labels not found");
        	return it->second;
    	}

    	// Get quantum numbers from index
    	key_type getLabels(int index) const 
    	{
        	if (index < 0 || index >= static_cast<int>(dim))
            	throw std::out_of_range("Index out of bounds");
        	return index_to_state[index];
    	}
	};

	// < na lla jja || b+ || nb llb jjb >
	inline double bosonCreatorRME(int na, int lla, int jja, 
						   int nb, int llb, int jjb,
						   u34utils::SO3_Racah& racah)
	{
		if (na == nb + 1)
		{
			double phase = ((lla - llb + jjb - jja) / 2 % 2 == 0) ? 1.0 : -1.0; 
			double factor = phase * racah.compute(llb,1,2,jja,jjb,lla) / std::sqrt(lla + 1.0);
			if (lla == llb + 2)
				return factor * std::sqrt((llb / 2.0 + 1.0) * (nb + llb / 2.0 + 3.0));
			else if (lla == llb - 2)
				return -1.0 * factor * std::sqrt((llb / 2.0) * (nb - llb / 2.0 + 2.0));
			else
				return 0;
		}
		else
			return 0;
	}

	// < na lla jja || b- || nb llb jjb >
	inline double bosonAnnihilatorRME(int na, int lla, int jja, 
						   	   int nb, int llb, int jjb,
						   	   u34utils::SO3_Racah& racah)
	{
		if (na == nb - 1)
		{
			double phase = ((lla - llb + jjb - jja) / 2 % 2 == 0) ? 1.0 : -1.0; 
			double factor = phase * racah.compute(llb,1,2,jja,jjb,lla) / std::sqrt(lla + 1.0);
			if (lla == llb + 2)
				return factor * std::sqrt((llb / 2.0 + 1.0) * (nb - llb / 2.0));
			else if (lla == llb - 2)
				return -1.0 * factor * std::sqrt((llb / 2.0) * (nb + llb / 2.0 + 1.0));
			else
				return 0;
		}
		else
			return 0;
	}
} // namespace HO
