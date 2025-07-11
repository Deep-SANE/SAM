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
    if (num_ads > 1)
    {
      auto [irreps1, rhomaxs1, etamaxs1] = U3xU4::decomposition(d,d);

      for (auto irr1 : irreps1)
      {
        if (num_ads == 2)
        {
          std::vector<U3xU4::irrep> adxad = {irr1};
          coupled_ads.push_back(adxad);
        }
        else // num_ads == 3
        {
          auto [irreps2, rhomaxs2, etamaxs2] = U3xU4::decomposition(irr1,d);
          for (auto irr2 : irreps2)
          {
            std::vector<U3xU4::irrep> adxadxad = {irr1, irr2};
            coupled_ads.push_back(adxadxad);
          }
        }
      }
    }

    // Couple annihilation tensors
    if (num_ats > 1)
    {
      auto [irreps1, rhomaxs1, etamaxs1] = U3xU4::decomposition(t,t);

      for (auto irr1 : irreps1)
      {
        if (num_ats == 2)
        {
          std::vector<U3xU4::irrep> atxat = {irr1};
          coupled_ats.push_back(atxat);
        }
        else
        {
          auto [irreps2, rhomaxs2, etamaxs2] = U3xU4::decomposition(irr1,t);
          for (auto irr2 : irreps2)
          {
            std::vector<U3xU4::irrep> atxatxat = {irr1, irr2};
            coupled_ats.push_back(atxatxat);
          }
        }
      }
    }

    //std::cout << "coupled ads: " << coupled_ads.size() << std::endl;
    //std::cout << "coupled ats: " << coupled_ats.size() << std::endl;

    // Now couple to the final tensor
    std::vector<U3xU4::Tadat> final;

    if (num_ats == 0) // only ad's
    {
      if (num_ads == 1)
        final.emplace_back(U3xU4::Tadat(n,1,0,{},{},d,1,1)); 
      else
      {
    	  for (auto coupled_ad : coupled_ads)
    	  {
    		  U3xU4::Tadat Tad(n, num_ads, 0,
            coupled_ad, {}, coupled_ad.back(), 1, 1);
          final.emplace_back(Tad);
        }
    	}
    }
    
    if (num_ads == 0) // only at's
    {
      if (num_ats == 1)
        final.emplace_back(U3xU4::Tadat(n,0,1,{},{},t,1,1)); 
      else
      {
        for (auto coupled_at : coupled_ats)
        {
          U3xU4::Tadat Tat(n, 0, num_ats,
            {}, coupled_at, coupled_at.back(), 1, 1);
          final.emplace_back(Tat);
        }
      }
    }

    if (num_ads == 1 && num_ats == 1)
    {
      auto [irreps, rhomaxs, etamaxs] = U3xU4::decomposition(d,t);
      for (auto irr : irreps)
        final.emplace_back(U3xU4::Tadat(n,1,1,{},{},irr,1,1));
    }
    else if (num_ads > 1 && num_ats == 1)
    {
      for (auto coupled_ad : coupled_ads)
      {
        auto [irreps, rhomaxs, etamaxs] = U3xU4::decomposition(
            coupled_ad.back(), t);

        for (auto i = 0; i < irreps.size(); i++)
        {
          U3xU4::Tadat Tadxat(n, num_ads, 1, 
            coupled_ad, {}, irreps[i], rhomaxs[i], etamaxs[i]);
          final.emplace_back(Tadxat);
        }
      }
    }
    else if (num_ads == 1 && num_ats > 1)
    {
      for (auto coupled_at : coupled_ats)
      {
        auto [irreps, rhomaxs, etamaxs] = U3xU4::decomposition(
              d, coupled_at.back());

        for (auto i = 0; i < irreps.size(); i++)
        {
          U3xU4::Tadat Tadxat(n, 1, num_ats, 
            {}, coupled_at, irreps[i], rhomaxs[i], etamaxs[i]);  
          final.emplace_back(Tadxat);
        }
      }
    }
    else
    {
	    for (auto coupled_ad : coupled_ads)
      {
    	  for (auto coupled_at : coupled_ats)
    	  {
          auto [irreps, rhomaxs, etamaxs] = U3xU4::decomposition(
            coupled_ad.back(), coupled_at.back());

          for (auto i = 0; i < irreps.size(); i++)
          {
        	  U3xU4::Tadat Tadxat(n, num_ads, num_ats, 
              coupled_ad, coupled_at, irreps[i], rhomaxs[i], etamaxs[i]);
            final.emplace_back(Tadxat);
          }
        }
      }
    }

    return final;
  } 
} // namespace U3xU4