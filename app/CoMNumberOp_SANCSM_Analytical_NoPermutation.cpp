#include <qcx_print.h>
using qcx_utils::print;

#include <U3basis.h>

#include <Databases.h>
#include <GlobalConstants.h>
#include <HOutilities.h>

#ifdef USE_SLEPC
#include <slepcsys.h>
#endif

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
                 int mmjd, int mmtd, sam::Databases& dbs)
{
  double x = 0.0;

  auto& cgcs = dbs.cgcs;
  auto& racah9 = dbs.racah9;

  if (mmta == mmtc && mmtb == mmtd)
  {
    double cgc = 0.0;
    for (int mu = -1; mu <= 1; mu++)
      cgc += (mu == 0 ? 1.0 : -1.0) *
              cgcs.compute(jjc, 2, jja, mmjc, 2 * mu, mmja) *
              cgcs.compute(jjd, 2, jjb, mmjd, -2 * mu, mmjb);
    x = cgc *
        (HO::bosonCreatorRME(na, lla, jja, nc, llc, jjc, racah9) *
         HO::bosonAnnihilatorRME(nb, llb, jjb, nd, lld, jjd, racah9) +
         HO::bosonAnnihilatorRME(na, lla, jja, nc, llc, jjc, racah9) *
          HO::bosonCreatorRME(nb, llb, jjb, nd, lld, jjd, racah9));
  }

  return x;
}

// Compute antisymmetrized matrix element for neutrons only
double ME_CoM_2B_AS(int na, int lla, int jja, int mmja, 
					int nb, int llb, int jjb, int mmjb, 
					int nc, int llc, int jjc, int mmjc, 
					int nd, int lld, int jjd, int mmjd, sam::Databases& dbs)
{
  return (ME_CoM_2B(na, lla, jja, mmja, 1, nb, llb, jjb, mmjb, 1,
                    nc, llc, jjc, mmjc, 1, nd, lld, jjd, mmjd, 1,
                    dbs) -
          ME_CoM_2B(na, lla, jja, mmja, 1, nb, llb, jjb, mmjb, 1,
                    nd, lld, jjd, mmjd, 1, nc, llc, jjc, mmjc, 1,
                    dbs)) / 4.0; 
}

double RME_CoM_2B_AS(int na, int lla, int jja, int nb, int llb, int jjb,
					 int nc, int llc, int jjc, int nd, int lld, int jjd,
					 int JJ, sam::Databases& dbs)
{
  auto& cgcs = dbs.cgcs;

  //Since the RME is the same as ME, choose MJ=J

  /*
  double norm_ab = 1.0, norm_cd = 1.0;
  if (na == nb && lla == llb && jja == jjb) norm_ab /= SQRT2;
  if (nc == nd && llc == lld && jjc == jjd) norm_cd /= SQRT2;
  */

  double x = 0.0;

  for (int mmja = jja; mmja >= -jja; mmja -= 2)
  {
  	int mmjb = JJ - mmja;
  	double cgcJ_ab = cgcs.compute(jja,jjb,JJ,mmja,mmjb,JJ);

  	for (int mmjc = jjc; mmjc >= -jjc; mmjc -= 2)
    {
  	  int mmjd = JJ - mmjc;
      double cgcJ_cd = cgcs.compute(jjc,jjd,JJ,mmjc,mmjd,JJ);

  	  x += cgcJ_ab * cgcJ_cd *
  	  	   ME_CoM_2B_AS(na, lla, jja, mmja, nb, llb, jjb, mmjb,
                        nc, llc, jjc, mmjc, nd, lld, jjd, mmjd, dbs);
  	  	
  	}
  }

  return x;// * norm_ab * norm_cd;
}

int main(int argc, char* argv[])
{
#ifdef USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  sam::Databases dbs;
  auto& racah9 = dbs.racah9;
  auto& wigU3SO3db = dbs.wigU3SO3db;

  int Nmax = 2, valenceShell = 0, maxShell = 2;
  auto na_nb_nc_nds =
      GenerateCombinationsHOShells(Nmax, valenceShell, maxShell);


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

      	  for (int Sf : {1,0})
      	  {
      	  	for (int Si : {1,0})
      	  	{
      	  	  for (const auto& k0L0 : k0L0s)
      	  	  {
      	    	int L0 = k0L0.L, k0max = k0L0.kappa;

      	    	for (int rho0 = 0; rho0 < rhomax0; rho0++)
      	    	{
      	    	  for (int k0 = 0; k0 < k0max; k0++)
      	    	  {
      	    	  	double strength = 0.0;

      	    	  	for (const auto& kfLf : kfLfs)
      	  		  	{
      	  		  	  int kfmax = kfLf.kappa, Lf = kfLf.L;

      	  		  	  for (const auto& kiLi : kiLis)
      	  		  	  {
      	  		  	  	int kimax = kiLi.kappa, Li = kiLi.L;

      	  		  	  	auto [allowL0, cgcL0] = wigU3SO3db.get(
      	  		  	  	  	  		u3f.lam(), u3f.mu(), Lf,
      	  		  	  	  	  		u3i.lam(), u3i.mu(), Li,
      	  		  	  	  	  		u30.lam(), u30.mu(), L0);
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

      	  		  	  	  	  	     	  auto rme = RME_CoM_2B_AS(na,2*la,jja, nb,2*lb,jjb,
																   nc,2*lc,jjc, nd,2*ld,jjd,
																   JJ,dbs);

#ifdef ADD_SU3LIB_PHASE_U3SO3
      	  		  	  	  	  	  		  int phase = (2*nc + 2*nd - jjc - jjd + JJ);
#else
                                      int phase = (nc + nd + lc + ld - jjc - jjd + JJ);
#endif                                      
        	  		  	  	  	  	  	  double factor = (phase % 4 == 0) ? 1.0 : -1.0;
        	  		  	  	  	  	  	  rme *= factor * std::sqrt((JJ + 1.0));// norm_ab / norm_cd);
      	     		  	  	  	  	  	  if (std::fabs(rme) < 1e-12) continue;

      	     		  	  	  	  	  	  double racah9_ab = racah9.compute(2*la,1,jja, 2*lb,1,jjb, 2*Lf,2*Sf,JJ);
      	  		  	  	  	  	  	  	  if (std::fabs(racah9_ab) < 1e-12) continue;

      	  		  	  	  	  	  	  	  double racah9_dc = racah9.compute(2*ld,1,jjd, 2*lc,1,jjc, 2*Li,2*Si,JJ);
      	  		     	  	  	  	  	  if (std::fabs(racah9_dc) < 1e-12) continue;

      	  		     	  	  	  	  	  double racah9_0 = racah9.compute(2*Lf,2*Sf,JJ, 2*Li,2*Si,JJ, 2*L0,2*L0,0);
      	  		  	  	  	  	  	  	  if (std::fabs(racah9_0) < 1e-12) continue;

      	  		  	  	  	  	  	  	  for (int kf = 0; kf < kfmax; kf++)
      	  		  	  	  	  	  		  {
      	  		  	  	  	  	  			for (int ki = 0; ki < kimax; ki++)
      	  		  	  	  	  	  			{
      	  		  	  	  	  	  			  double s = cgcL0((kf * kimax + ki) * k0max + k0,rho0)
      	  		     	  	  	  	  				   * cgcLab(kf,0) * cgcLdc(ki,0) * rme 
      	  		     	  	  	  	  				   * racah9_ab * racah9_dc * racah9_0;
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

      	  		  	if (std::fabs(strength) > 1e-12)
      	  		  	{
      	  		  	  print("Tensor [[({},0)x({},0)]{}{} x [(0,{})x(0,{})]{}{}]{}{}{}{} => strength: {}\n",
      	  		  	  	  	na,nb,u3f,Sf, nd,nc,u3i,Si, rho0,u30,k0,L0, strength);
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


#ifdef USE_SLEPC
  SlepcFinalize();
#endif

  return 0;
}
