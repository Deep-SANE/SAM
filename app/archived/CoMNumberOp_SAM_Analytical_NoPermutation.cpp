#include <print.h>
#include <utils.h>

using utils::print;

#include <u3lib.h>
#include <u4lib.h>
#include <U3basis.h>
#include <U4basis.h>

#include <GlobalConstants.h>
#include <HOutilities.h>
#include <U3xU4CouplingCoefficient.h>
#include <U3xU4irrep.h>

auto GenerateCombinationsHOShells(int Nmax, int valenceShell,
                                         int maxShell)
{
  std::vector<std::array<int, 4>> na_nb_nc_nds;

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

          std::array<int, 4> ns = {na, nb, nc, nd};
          na_nb_nc_nds.push_back(ns);
        }
      }
    }
  }
  return na_nb_nc_nds;
}

double ME_CoM_2B(int na, int lla, int jja, int mmja, int mmta, int nb,
                        int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                        int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                        int mmjd, int mmtd)
{
  double x = 0.0;

  auto& cgcs = utils::cgcs();
  auto& racah9 = utils::fanos();

  if (mmta == mmtc && mmtb == mmtd)
  {
    double cgc = 0.0;
    for (int mu = -1; mu <= 1; mu++)
      cgc += (mu == 0 ? 1.0 : -1.0) *
              cgcs.get({jjc, 2, jja, mmjc, 2 * mu, mmja}) *
              cgcs.get({jjd, 2, jjb, mmjd, -2 * mu, mmjb});
    x = cgc *
        (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc) *
         HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd) +
         HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc) *
          HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd));
  }

  return x;
}

double ME_CoM_2B_AS(int na, int lla, int jja, int mmja, int mmta, int nb,
                           int llb, int jjb, int mmjb, int mmtb, int nc, int llc,
                           int jjc, int mmjc, int mmtc, int nd, int lld, int jjd,
                           int mmjd, int mmtd)
{
  return (ME_CoM_2B(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                    nc, llc, jjc, mmjc, mmtc, nd, lld, jjd, mmjd, mmtd) -
          ME_CoM_2B(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                    nd, lld, jjd, mmjd, mmtd, nc, llc, jjc, mmjc, mmtc)) / 4.0; 
}

double RME_CoM_2B_AS(int na, int lla, int jja, int nb, int llb, int jjb,
							int nc, int llc, int jjc, int nd, int lld, int jjd,
							int JJ, int TT)
{
  auto& cgcs = utils::cgcs();

  //Since the RME is the same as ME, choose MJ=J, MT=T

  /*
  double norm_ab = 1.0, norm_cd = 1.0;
  if (na == nb && lla == llb && jja == jjb) norm_ab /= SQRT2;
  if (nc == nd && llc == lld && jjc == jjd) norm_cd /= SQRT2;
  */

  double x = 0.0;

  for (int mmja = jja; mmja >= -jja; mmja -= 2)
  {
  	int mmjb = JJ - mmja;
  	double cgcJ_ab = cgcs.get({jja,jjb,JJ,mmja,mmjb,JJ});

  	for (int mmta = 1; mmta >= -1; mmta -= 2)
  	{
  	  int mmtb = TT - mmta;
	  double cgcT_ab = cgcs.get({1,1,TT,mmta,mmtb,TT});

  	  for (int mmjc = jjc; mmjc >= -jjc; mmjc -= 2)
  	  {
  	  	int mmjd = JJ - mmjc;
  	  	double cgcJ_cd = cgcs.get({jjc,jjd,JJ,mmjc,mmjd,JJ});

  	  	for (int mmtc = 1; mmtc >= -1; mmtc -= 2)
  	  	{
  	  	  int mmtd = TT - mmtc;
		  double cgcT_cd = cgcs.get({1,1,TT,mmtc,mmtd,TT});

  	  	  x += cgcJ_ab * cgcJ_cd * cgcT_ab * cgcT_cd *
  	  	  	ME_CoM_2B_AS(na, lla, jja, mmja, mmta, nb, llb, jjb, mmjb, mmtb,
                    nc, llc, jjc, mmjc, mmtc, nd, lld, jjd, mmjd, mmtd);
  	  	}
  	  }
  	}
  }

  return x;// * norm_ab * norm_cd;
}

int main(int argc, char* argv[])
{
  utils::initialize(&argc, &argv);
  u3::initialize();
  u4::initialize();

  auto& racah9 = utils::fanos();
  auto& wigU3SO3db = u3::wigU3SO3db();
  auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

  int Nmax = 2, valenceShell = 0, maxShell = 2;
  auto na_nb_nc_nds =
      GenerateCombinationsHOShells(Nmax, valenceShell, maxShell);

  u4::irrep create(1,0,0), destroy(0,0,1);
  auto [u4fs, etamaxfs] = u4::decomposition(create, create);
  auto [u4is, etamaxis] = u4::decomposition(destroy, destroy);


  for (auto [na, nb, nc, nd] : na_nb_nc_nds)
  {
  	u3::irrep u3a(na,0), u3b(nb,0), u3c(0,nc), u3d(0,nd);

    auto [u3fs, rhomaxfs] = u3::decomposition(u3a, u3b);
    auto [u3is, rhomaxis] = u3::decomposition(u3d, u3c);

    for (const auto& u3f : u3fs)
    {
      auto kfLfs = u3::physicalBasis(u3f);

      for (const auto& u3i : u3is)
      {
      	auto kiLis = u3::physicalBasis(u3i);

      	auto [u30s, rhomax0s] = u3::decomposition(u3f, u3i);
      	for (size_t i = 0; i < u30s.size(); i++)
      	{
      	  auto u30 = u30s[i];
      	  auto rhomax0 = rhomax0s[i];
      	  auto k0L0s = u3::physicalBasis(u30);

      	  for (const auto& u4f : u4fs)
      	  {
      	  	auto zfSSfTTfs = u4::physicalBasis(u4f);

      	  	for (const auto& u4i : u4is)
      	  	{
      	  	  auto ziSSiTTis = u4::physicalBasis(u4i);

      	  	  auto [u40s, etamax0s] = u4::decomposition(u4f, u4i);
      	  	  for (size_t j = 0; j < u40s.size(); j++)
      		    {
      		  	  auto u40 = u40s[j];
        		  	auto etamax0 = etamax0s[j];
        	  		auto z0SS0TT0s = u4::physicalBasis(u40);
      	  		
        	  		for (const auto& k0L0 : k0L0s)
        	  		{
      	    		  int L0 = k0L0.L(), k0max = k0L0.kappa();

        	  		  for (const auto& z0SS0TT0 : z0SS0TT0s)
        	  		  {
        	  		  	if (z0SS0TT0.TT() != 0) continue;
        	  		  	int SS0 = z0SS0TT0.SS(), z0max = z0SS0TT0.zeta();

      	    		  	for (int rho0 = 0; rho0 < rhomax0; rho0++)
      	     		  	{
      	  	  	  	  for (int k0 = 0; k0 < k0max; k0++)
        	  		  	  {
        	  		  	  	for (int eta0 = 0; eta0 < etamax0; eta0++)
        	  		  	  	{
        	  		  	  	  for (int z0 = 0; z0 < z0max; z0++)
      	     		  	  	  {
/*
      	     		  	  	  	print("\nNew tensor calculated\n");
      	  	  	  	  	  	print("Tensor [[({},0)x({},0)]{}{} x 
                                  [(0,{})x(0,{})]{}{}]{}{}{}{}{}{}{}{}\n",
      	  		  	  	  	  	    na,nb,u3f,u4f, nd,nc,u3i,u4i, 
                                  rho0,u30,eta0,u40, k0,L0, z0,SS0);
*/
      	  		  	  	  	  double strength = 0.0;

      	  		  	  	  	  for (const auto& kfLf : kfLfs)
      	  		  	  	  	  {
      	  		  	  	  	    int kfmax = kfLf.kappa(), Lf = kfLf.L();

      	  		  	  	  	    for (const auto& kiLi : kiLis)
      	  		  	  	  	    {
      	  		  	  	  	  	  int kimax = kiLi.kappa(), Li = kiLi.L();

        	  		  	  	  	  	auto [allowL0, cgcL0] = wigU3SO3db.get(
                      	  		  	  	  	  		u3f.lam(), u3f.mu(), Lf,
                      	  		  	  	  	  		u3i.lam(), u3i.mu(), Li,
      	  		                   	  	  	  		u30.lam(), u30.mu(), L0);
        	  		  	  	  	  	//print("Lf:{}, Li:{}, L0:{}\n",Lf,Li,L0);

        	  		  	  	  	  	if (!allowL0) continue;

        	  		  	  	  	  	for (int la = na; la >= 0; la -= 2)
        	  		  	  	  	  	{
      	     		  	  	  	  	  for (int lb = nb; lb >= 0; lb -= 2)
      	     		  	  	  	  	  {
      	  		  	  	  	  	  	  auto [allowLab, cgcLab] = wigU3SO3db.get(
      	  		  	  	  	  	  		 na,0,la, nb,0,lb, u3f.lam(), u3f.mu(), Lf);

      	  		  	  	  	  	  	  if (!allowLab) continue;

      	  		  	  	  	  	  	  for (int lc = nc; lc >= 0; lc -= 2)
      	  		  	  	  	  	  	  {
        	  		  	  	  	  	  	  for (int ld = nd; ld >= 0; ld -= 2)
        	  		  	  	  	  	  	  {
        	  		  	  	  	  	  	  	auto [allowLdc, cgcLdc] = wigU3SO3db.get(
          	  		  	  	  	  	  		  0,nd,ld, 0,nc,lc, u3i.lam(), u3i.mu(), Li);

      	     		  	  	  	  	  		  if (!allowLdc) continue;

        	     		  	  	  	  	  		for (int jja = 2*la + 1; jja >= abs(2*la - 1); jja -= 2)
        	  	  	  	  	  	  	  		{
        	  		     	  	  	  	  		  for (int jjb = 2*lb + 1; jjb >= abs(2*lb - 1); jjb -= 2)
        	  		     	  	  	  	  		  {
      	     		  	    	  	  	  		  	//double norm_ab = 1.0;
      	     		  	     	  	  	  		  	//if (na == nb && la == lb && jja == jjb) norm_ab /= SQRT2;

      	  	  	  	  	     	  	  		  	for (int jjc = 2*lc + 1; jjc >= abs(2*lc - 1); jjc -= 2)
      	  		     	  	    	  	  		  	{
      	  		     	  	  	     	  		  	  for (int jjd = 2*ld + 1; jjd >= abs(2*ld - 1); jjd -= 2)
      	  		  	     	  	     	  		  	  {
      	  		  	  	     	  	     		  	  	//double norm_cd = 1.0;
 												                        //if (nc == nd && lc == ld && jjc == jjd) norm_cd /= SQRT2;

        	  		  	  	  	     	  		  	  	for (int JJ = jja + jjb; JJ >= abs(jja - jjb); JJ -= 2)
        	  		  	  	  	  	     		  	  	{
        	  		  	  	  	  	     		  	  	  if (JJ > (jjc + jjd) || JJ < abs(jjc - jjd)) continue;

        	  		  	  	  	  	  	  	  	  	  for (int TT = 2; TT >= 0; TT -= 2)
      	    		  	  	  	  	  		     	  	  {
      	     		  	  	  	  	  		     	  	  	auto rme = RME_CoM_2B_AS(na,2*la,jja, nb,2*lb,jjb,
						  													                                     nc,2*lc,jjc, nd,2*ld,jjd,
							   												                                     JJ,TT);

      	  		     	  	  	  	  		  	  	    	double phase_exchange = ((jjc + jjd - JJ - TT) % 4 == 0) ? 1.0 : -1.0;
  								  					                      auto rme_exchange = RME_CoM_2B_AS(na,2*la,jja, nb,2*lb,jjb,
										    											                                        nd,2*ld,jjd, nc,2*lc,jjc,
												    									                                        JJ,TT);
  													                        assert(std::fabs(rme - rme_exchange * phase_exchange) < 1e-12);

#ifdef ADD_SU3LIB_PHASE_U3SO3
                                                    int phase = (2*nc + 2*nd + 2 - jjc - jjd + JJ + TT);
#else                                                    
        	  		  	  	  	     	  		  	  	  	int phase = (nc + nd + lc + ld + 2 - jjc - jjd + JJ + TT);
#endif        	  		  	  	  	  	  		  	  	  	                                                      
        	  		  	  	  	  	  		  	  	  	  double factor = (phase % 4 == 0) ? 1.0 : -1.0;
        	  		  	  	  	  	  		  	  	  	  rme *= (factor * std::sqrt((JJ + 1.0) * (TT + 1.0))
      	     		  	  	  	  	  		  	  	  			     );// norm_ab / norm_cd);

      	     		  	  	  	  	  		  	  	  	  if (std::fabs(rme) < 1e-12) continue;

        	  	  	  	  	  	  	  		  	  	  	for (const auto& zfSSfTTf : zfSSfTTfs)
        	  		     	  	  	  	  		  	  	  	{
        	  		     	  	  	  	  		  	  	  	  assert(zfSSfTTf.zeta() == 1);
        	  		  	     	  	  	  		  	  	  	  if (zfSSfTTf.TT() != TT) continue;
      	     		  	  	     	  	  		  	  	  	  int SSf = zfSSfTTf.SS();

      	     		  	  	     	  	  		  	  	  	  auto [allowSTab, cgcSTab] = wigU4SU2SU2db.get(
      	  	  	  	  	  	     	  		  				      1,0,0,1,1, 1,0,0,1,1, u4f.P1(),u4f.P2(),u4f.P3(),SSf,TT);

      	  		     	  	  	  	     					        if (!allowSTab) continue;

      	  		     	  	  	  	     					        double racah9_ab = racah9.get({2*la,1,jja, 2*lb,1,jjb, 2*Lf,SSf,JJ});
      	  		  	     	  	  	  	  				        if (std::fabs(racah9_ab) < 1e-12) continue;
      	  		  	  	     	  	  		   			        //print("la:{}, jja:{}, lb:{}, jjb:{}, Lf:{}, SSf:{}, JJ:{}\n",
      	  		  	  	     	  	  			  		        //		la,jja,lb,jjb,Lf,SSf,JJ);

      	  		  	  	  	     	  				   	        for (const auto& ziSSiTTi : ziSSiTTis)
      	  		  	  	  	  	     					          {
        	  		  	  	  	  	    					  	      assert(ziSSiTTi.zeta() == 1);
        	  		  	  	  	  	     					  	      if (ziSSiTTi.TT() != TT) continue;
        	  		  	  	  	  	  	  				  	      int SSi = ziSSiTTi.SS();

        	  		  	  	  	  	  		   			  	      auto [allowSTdc, cgcSTdc] = wigU4SU2SU2db.get(
      	     		  	  	  	  	  		    				        0,0,1,1,1, 0,0,1,1,1, u4i.P1(),u4i.P2(),u4i.P3(),SSi,TT);

      	     		  	  	  	  	  				  	  	      if (!allowSTdc) continue;

      	  	  	  	  	  	  	  					    	      double racah9_dc = racah9.get({2*ld,1,jjd, 2*lc,1,jjc, 2*Li,SSi,JJ});
      	  		     	  	  	  	  					     	      if (std::fabs(racah9_dc) < 1e-12) continue;
      	  		     	  	  	  	  					  	        //print("ld:{}, jjd:{}, lc:{}, jjc:{}, Li:{}, SSi:{}, JJ:{}\n",
      	  		   	    	  	  	  					  	        //	ld,jjd,lc,jjc,Li,SSi,JJ);

      	  		  	    	  	  	  					  	        auto [allowST0, cgcST0] = wigU4SU2SU2db.get(
      	  		  	  	     	  	  		  				                u4f.P1(),u4f.P2(),u4f.P3(),SSf,TT, 
      	  		  	  	     	  	  		  				                u4i.P1(),u4i.P2(),u4i.P3(),SSi,TT, 
      	  		  	  	  	     	  		  				                u40.P1(),u40.P2(),u40.P3(),SS0,0);

      	  		  	  	   	  	     					            if (!allowST0) continue;

        	  		  	  	  	  	     					          double racah9_0 = racah9.get({2*Lf,SSf,JJ, 2*Li,SSi,JJ, 2*L0,SS0,0});
        	  		  	  	  	  	  	  				          if (std::fabs(racah9_0) < 1e-12) continue;

        	  		  	  	  	  	  		   			          for (int kf = 0; kf < kfmax; kf++)
        	  		  	  	  	  	  			  		          {
      	     		  	  	  	  	  				   	            for (int ki = 0; ki < kimax; ki++)
      	     		  	  	  	  	  					              {
      	  	  	  	  	     	  	  					      	      double s = cgcL0((kf * kimax + ki) * k0max + k0,rho0)
      	  		     	  	  	    	  					      			           * cgcLab(kf,0) * cgcLdc(ki,0) * rme 
      	  		     	  	  	     	  					      			           * racah9_ab * cgcSTab(0,0) * racah9_dc * cgcSTdc(0,0)
      	  		  	     	  	  	     					      			           * racah9_0 * cgcST0(z0,eta0);
      	  		  	  	     	  	  	   				      	      //print("Add to strength: {}\n",s);
      	  		  	  	     	  	  		  			      	      strength += s;
      	  		  	  	  	     	  			   		            }
      	  		  	  	  	  	     				  	          }
      	  		  	  	  	  	  	  				          }
      	  		  	  	  	  	  		     	  	    	}
      	  		  	  	  	  	  		    	  	    }
      	  		  	  	  	  	  		  	     	  } 
      	  		  	  	  	  	  		  	      }
      	  		  	  	  	  	  		    	  }
      	  		  	  	  	  	  		      }
      	  		  	  	  	  	  		    }
      	  		  	  	  	  	  	    }
      	  		  	  	  	  	  	  }
      	  		  	  	  	  	    }
      	  		  	  	  	  	  }
      	  		  	  	  	    }
      	  		  	  	  	  }

      	     		  	  	  	if (std::fabs(strength) > 1e-12)
      	  	  	  	  	  	{
      	  		     	  	  	  print("Tensor [[({},0)x({},0)]{}{} x [(0,{})x(0,{})]{}{}]{}{}{}{}{}{}{}{} => strength: {}\n",
      	  		  	  	  	  	                   na,nb,u3f,u4f, nd,nc,u3i,u4i, rho0,u30,eta0,u40, k0,L0, z0,SS0, strength);
      	  		  	  	  	  }
        	  		  	  	  }
        	  		  	  	}
        	  		  	  }
        	  		  	}
      	     		  }
      	     		}
      	  	  }
      	  	}
      	  }
      	}
      }
    }
  }

  u4::finalize();
  u3::finalize();
  utils::finalize();
}
