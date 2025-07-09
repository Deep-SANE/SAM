#include <U3xU4MultiShellTensors.h>
#include <U3xU4SingleShellTensors.h>

namespace U3xU4
{
	//////////////////// TWO-BODY MULTI-SHELL TENSORS ////////////////////

	// Expand into single-shell tensors
    std::pair<std::vector<Eigen::RowVectorXd>, // coeffs
              std::vector<U3xU4::Tadats_phys>> // list of Tadat_phys's
    TwoBodyMultiShellTensor::expand_to_single_shell_tensors(
    	u34utils::SO3_CGC& cgcs, u34utils::SO3_Racah9& racah9,
        u3::WigU3SO3_db& wigU3SO3db, u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
    {
    	std::vector<Eigen::RowVectorXd> coeffsArr;
    	std::vector<U3xU4::Tadats_phys> tensorsArr;

    	return std::make_pair(coeffsArr,tensorsArr);
    }

	// Expand into uncoupled products of creation operators (ad) 
    // and annihilation operators (a)
    // coeffs have rhomax x etamax x kappamax x zetamax columns, index of each column is
    // ((rho * etamax + eta) * kappamax + kappa) * zetamax + zeta
    std::pair<std::vector<Eigen::RowVectorXd>, // coeffs
              std::vector<U3xU4::ads_ats_phys>> // list of ad's and a's
    TwoBodyMultiShellTensor::expand(u34utils::SO3_CGC& cgcs, 
    								u34utils::SO3_Racah9& racah9,
           							u3::WigU3SO3_db& wigU3SO3db, 
           							u4::WigU4SU2SU2_db& wigU4SU2SU2db) const
    {
    	std::vector<Eigen::RowVectorXd> coeffsArr;
    	std::vector<U3xU4::ads_ats_phys> opsArr;
    	//recall: ads_ats_phys = std::vector<std::variant<U3xU4::ad_phys,U3xU4::at_phys>>;

/*
    	if (coupled_irreps.size() == 0)
  		{
    		// Delegate to single-shell expansion 
    		const U3xU4::Tadat& tadat = single_shell_tensors[0];
    		U3xU4::Tadat_phys tadat_phys(tadat.n, tadat.ads, tadat.ats,
                tadat.prod_ads, tadat.prod_ats, tadat.T, rhomax, etamax, 
                LL, SS, TT, MMT, JJ, MMJ);

    		auto [coeffs, ads, ats] = tadat_phys.expand(cgcs, racah9, wigU3SO3db, wigU4SU2SU2db);

    		for (int i = 0; i < coeffs.size(); i++)
    		{
      			U3xU4::ads_ats_phys ops;
      			for (const auto& ad : ads[i])
        			ops.emplace_back(ad);
      			for (const auto& a : ats[i])
        			ops.emplace_back(a);

        		coeffsArr.push_back(std::move(coeffs[i]));
        		opsArr.push_back(std::move(ops));
  			}
  		}
  		else
  		{
  			// Uncouple the last single-shell tensor

  		}
*/

    	return std::make_pair(coeffsArr, opsArr);
    }
} // namespace U3xU4