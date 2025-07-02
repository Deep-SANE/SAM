#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
  // Find all single-shell tensors composed of 
  // num_ads creation tensors and num_ats annihilation tensors
  std::vector<U3xU4::Tadat> GenerateSingleShellTensors(int n, 
  										int num_ads, int num_ats)
  {
    std::vector<std::vector<U3xU4::irrep>> coupled_ads, coupled_ats;

    U3xU4::irrep d(n,0,0, 1,0,0,0), t(n,n,0, 1,1,1,0);

    // Couple creation tensors
    if (num_ads == 1)
    {
      std::vector<U3xU4::irrep> ad(1,d);
      coupled_ads.push_back(ad);
    }
    else if (num_ads == 2)
    {
      auto [irreps, rhomaxs, etamaxs] = U3xU4::decomposition(d,d);

      for (auto irr : irreps)
      {
        std::vector<U3xU4::irrep> adxad(1,irr);
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
      std::vector<U3xU4::irrep> at(1,t);
      coupled_ats.push_back(at);
    }
    else if (num_ats == 2)
    {
      auto [irreps, rhomaxs, etamaxs] = U3xU4::decomposition(t,t);

      for (auto irr : irreps)
      {
        std::vector<U3xU4::irrep> atxat(1,irr);
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
    	std::vector<U3xU4::irrep> coupled_at; // empty
    	for (int i = 0; i < coupled_ads.size(); i++)
    	{
    		U3xU4::Tadat Tadxat(n, num_ads, num_ats,
          coupled_ads[i], coupled_at, coupled_ads[i].back(), 1, 1);
        final.emplace_back(Tadxat);
    	}
    }
    else if (num_ads == 0) // only at's
    {
    	std::vector<U3xU4::irrep> coupled_ad; // empty
    	for (int i = 0; i < coupled_ats.size(); i++)
    	{
    		U3xU4::Tadat Tadxat(n, num_ads, num_ats,
          coupled_ad, coupled_ats[i], coupled_ats[i].back(), 1, 1);
        final.emplace_back(Tadxat);
    	}
    }
    else // both ad's and at's
    {
	  for (int i = 0; i < coupled_ads.size(); i++)
      {
      	for (int j = 0; j < coupled_ats.size(); j++)
      	{
          auto [irreps, rhomaxs, etamaxs] = 
            U3xU4::decomposition(coupled_ads[i].back(), coupled_ats[j].back());

          for (int k = 0; k < irreps.size(); k++)
          {
          	U3xU4::Tadat Tadxat(n, num_ads, num_ats, coupled_ads[i], coupled_ats[j], 
                                                  irreps[k], rhomaxs[k], etamaxs[k]);
            final.emplace_back(Tadxat);
          }
      	}
      }
    }

    return final;
  } 
} // namespace U3xU4