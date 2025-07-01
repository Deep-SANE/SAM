#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  void Tadat::print() const
  {
  	std::cout << "[ ";
  	if (ads > 0)
  	{
  		std::cout << ads << "ads ";
  		for (auto x : prod_ads) std::cout << x.first << x.second << " ";
  		if (ats > 0)
  			std::cout << "x ";
  	}
  	if (ats > 0)
  	{
  		std::cout << ats << "ats ";
  		for (auto x : prod_ats) std::cout << x.first << x.second << " ";
  	}
  	std::cout << "] rho0max = " << rho0max << t3 
  			  << " eta0max = " << eta0max << t4 << std::endl; 

  }

  // Find all single-shell tensors composed of 
  // num_ads creation tensors and num_ats annihilation tensors
  std::vector<U3xU4::Tadat> GenerateSingleShellTensors(int n, 
  										int num_ads, int num_ats)
  {
    std::vector<U3xU4::ListIrreps> coupled_ads, coupled_ats;

    u3::irrep d3(n,0,0), t3(n,n,0);
    u4::irrep d4(1,0,0,0), t4(1,1,1,0);

    // Couple creation tensors
    if (num_ads == 1)
    {
      U3xU4::ListIrreps ad;
      ad.emplace_back(d3, d4);
      coupled_ads.push_back(ad);
    }
    else if (num_ads == 2)
    {
      auto [d3xd3s, multi3] = u3::decomposition(d3,d3);
      auto [d4xd4s, multi4] = u4::decomposition(d4,d4);

      for (int i = 0; i < d3xd3s.size(); i++)
        for (int j = 0; j < d4xd4s.size(); j++)
        {
          U3xU4::ListIrreps adxad;
          adxad.emplace_back(d3xd3s[i], d4xd4s[j]);
          coupled_ads.push_back(adxad);
        }
    }
    else if (num_ads == 3)
    {
      throw std::invalid_argument("Product of 3 ad's is not implemented yet");
    }

    // Couple annihilation tensors
    if (num_ats == 1)
    {
      U3xU4::ListIrreps at;
      at.emplace_back(t3, t4);
      coupled_ats.push_back(at);
    }
    else if (num_ats == 2)
    {
      auto [t3xt3s, multi3] = u3::decomposition(t3,t3);
      auto [t4xt4s, multi4] = u4::decomposition(t4,t4);

      for (int i = 0; i < t3xt3s.size(); i++)
        for (int j = 0; j < t4xt4s.size(); j++)
        {
          U3xU4::ListIrreps atxat;
          atxat.emplace_back(t3xt3s[i], t4xt4s[j]);
          coupled_ats.push_back(atxat);
        }
    }
    else if (num_ats == 3)
    {
      throw std::invalid_argument("Product of 3 at's is not implemented yet");
    }

    // Now couple to the final tensor
    std::vector<U3xU4::Tadat> final;

    if (num_ats == 0) // only ad's
    {
    	U3xU4::ListIrreps coupled_at; // empty
    	for (int i = 0; i < coupled_ads.size(); i++)
    	{
    		U3xU4::Tadat Tadxat(n, num_ads, num_ats,
          				coupled_ads[i], coupled_at,
          				coupled_ads[i].back().first, 1, 
          				coupled_ads[i].back().second, 1);
            final.emplace_back(Tadxat);
    	}
    }
    else if (num_ads == 0) // only at's
    {
    	U3xU4::ListIrreps coupled_ad; // empty
    	for (int i = 0; i < coupled_ats.size(); i++)
    	{
    		U3xU4::Tadat Tadxat(n, num_ads, num_ats,
          				coupled_ad, coupled_ats[i],
          				coupled_ats[i].back().first, 1, 
          				coupled_ats[i].back().second, 1);
            final.emplace_back(Tadxat);
    	}
    }
    else // both ad's and at's
    {
	  for (int i = 0; i < coupled_ads.size(); i++)
      {
      	for (int j = 0; j < coupled_ats.size(); j++)
      	{
          u3::irrep ads3 = coupled_ads[i].back().first,
           	        ats3 = coupled_ats[j].back().first;
          auto [T3s, multi3s] = u3::decomposition(ads3, ats3);

          u4::irrep ads4 = coupled_ads[i].back().second,
            	    ats4 = coupled_ats[j].back().second;
          auto [T4s, multi4s] = u4::decomposition(ads4, ats4);

          for (int k = 0; k < T3s.size(); k++)
          	for (int l = 0; l < T4s.size(); l++)
          	{
          		U3xU4::Tadat Tadxat(n, num_ads, num_ats,
          				coupled_ads[i], coupled_ats[j],
          				T3s[k], multi3s[k], T4s[l], multi4s[l]);
                final.emplace_back(Tadxat);
            }
      	}
      }
    }

    return final;
  }
} // namespace U3xU4