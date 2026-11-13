#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <eigensolver.h>
#include <readwrite.h>

#include <sam.h>

#include <OperatorJT.h>

int main(int argc, char* argv[])
{
  sam::initialize(&argc, &argv);

  // Specify Hamiltonia
  int Nmax = 2, hw = 16;
  std::string Hamil_name_i = "HO_Be7_e10_hw16_E342";
  std::string Hamil_name_f = "HO_Li7_e10_hw16_E342";

  // Specify operator name
  // std::string name = "GamowTellerMinus";
  // std::string name = "GamowTellerMinus_2B_N3LO_EM500_bare";
  // std::string name = "GamowTellerPlus";
  std::string name = "GamowTellerPlus_2B_N3LO_EM500_bare";

/*
  u4::irreps irrs4 = {{1,1,1,0}, {2,1,0,0}, {3,3,1,0}};
  u3::irreps irrs3_4 = {{7,0,0}, {6,1,0}, {5,2,0}, {4,3,0}, {3,1,0}},
             irrs3_6 = {{9,0,0}, {8,1,0}, {7,2,0}, {6,3,0}, {5,4,0}},
             irrs3_8 = {{11,0,0}, {10,1,0}, {9,2,0}, {8,3,0}, {7,4,0}};
 */

  int Zi = 4, Ni = 3, JJi = 1, TTi = 1;
  std::string dir_i = "Truncation_TT1";

  U3xU4::hw_subspace
    hw0i(0, {}, {}, {}, {TTi}),
    hw2i(2, {}, {}, {}, {TTi}),
    hw4i(4, {}, {}, {}, {TTi}),
    hw6i(6, {}, {}, {}, {TTi}),
    hw8i(8, {}, {}, {}, {TTi});

  U3xU4::ManyNucleonBasis kets(Ni, Zi, Nmax, JJi, {hw0i, hw2i, hw4i, hw6i, hw8i});
  kets.construct();
  auto ket_irreps = kets.irreps();
  std::cout << "Ket basis dimension: " << kets.dim() << std::endl;
  
  int Zf = 3, Nf = 4, JJf = 1, TTf = 1;
  std::string dir_f = "Truncation_TT1";

  U3xU4::hw_subspace
    hw0f(0, {}, {}, {}, {TTf}),
    hw2f(2, {}, {}, {}, {TTf}),
    hw4f(4, {}, {}, {}, {TTf}),
    hw6f(6, {}, {}, {}, {TTf}),
    hw8f(8, {}, {}, {}, {TTf});

  U3xU4::ManyNucleonBasis bras(Nf, Zf, Nmax, JJf, {hw0f, hw2f, hw4f, hw6f, hw8f});
  bras.construct();
  auto bra_irreps = bras.irreps();
  std::cout << "Bra basis dimension: " << bras.dim() << std::endl;

  int maxShell = std::max(kets.maxShell(), bras.maxShell());
  HO::Table_nljmjmt HO_labels(maxShell);
  U3xU4::Table_gelfand gelfand_labels(maxShell);

  JT::Operator Hamil_i(Hamil_name_i, Ni, Zi, hw, Nmax, HO_labels, gelfand_labels),
               Hamil_f(Hamil_name_f, Nf, Zf, hw, Nmax, HO_labels, gelfand_labels),
               op(name, Ni, Zi, hw, Nmax, HO_labels, gelfand_labels);
  op.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  // read initial wave functions
  // auto wfns_file_i = Hamil_i.wfns_file(kets);
  auto wfns_file_i = Hamil_i.wfns_file(kets, dir_i);
  auto [evals_i, evecs_i] = utils::readEigenPairs(wfns_file_i);

  // read final wave functions
  // auto wfns_file_f = Hamil_f.wfns_file(bras);
  auto wfns_file_f = Hamil_f.wfns_file(bras, dir_f);
  auto [evals_f, evecs_f] = utils::readEigenPairs(wfns_file_f);
  
  // transition
  
  auto strengths = op.transitionStrength(bras, kets, 
    {{evecs_f.col(0), evecs_i.col(0)}});

  std::cout << "Transition strengths of " << name << " are" << std::endl;
  for (int i = 0; i < strengths.size(); i++)
    std::cout << "Channel " << i << ": " << strengths[i] << std::endl;

  // Analyze contribution
/*
  auto mat = op.matrix_Eigen(bras, kets);

  u4::irreps irrs = {{1,1,1,0}, {2,1,0,0}, {3,2,2,0}, {3,0,0,0}, {3,3,1,0}, 
                     {4,2,1,0}, {4,3,0,0}, {5,1,1,0}, {5,2,0,0}, {6,1,0,0}, {7,0,0,0}};

  std::cout << "Irrep\tC2\tC3\tC4" << std::endl;
  for (const auto& irr : irrs)
    std::cout << irr << "\t" << irr.C2() << "\t" 
              << irr.C3() << "\t" << irr.C4() << std::endl;

  std::vector<std::vector<size_t>> ket_pos(irrs.size()), bra_pos(irrs.size());
  
  auto extract_pos = [&](const auto& irreps, const auto& space, 
                     std::vector<std::vector<size_t>>& pos)
  {
    for (size_t id = 0; id < irreps.size(); id++)
    {
      auto [gamma, omega] = irreps[id];
      auto offset_curr = irreps.offset(id);
      auto offset_next = (id == irreps.size() - 1)
                           ? space.dim() : irreps.offset(id + 1);
      auto irr = (omega.empty()) ? gamma.back().u4() : omega.back().u4();
      irr.normalize();
      auto it = std::find(irrs.begin(), irrs.end(), irr);
      assert(it != irrs.end());
      auto& po = pos[std::distance(irrs.begin(), it)];
      for (size_t row = offset_curr; row < offset_next; row++)
        po.push_back(row);
    }
  };

  extract_pos(ket_irreps, kets, ket_pos);
  extract_pos(bra_irreps, bras, bra_pos);

  std::vector<Eigen::VectorXd> ket_extract, bra_extract;

  auto extract_elem = [&](const auto& evecs, const auto& pos, auto& elems)
  {
    for (int i = 0; i < irrs.size(); i++)
    {
      const auto& po = pos[i];

      Eigen::VectorXd elem(po.size());
      for (size_t local = 0; local < po.size(); local++)
        elem(local) = evecs(po[local], 0);  // local index into elem
      
      elems.push_back(std::move(elem));
    }
  };

  extract_elem(evecs_i, ket_pos, ket_extract);
  extract_elem(evecs_f, bra_pos, bra_extract);

  auto MMJ = std::min(JJf, JJi);
  auto cgc = utils::cgcs().get({JJi, 2, JJf, MMJ, 0, MMJ});
  double factor = std::sqrt(JJf + 1.0) / cgc;

  for (int i = 0; i < irrs.size(); i++)
  {
    const auto& k = ket_extract[i];
    const auto& k_pos = ket_pos[i];

    for (int j = 0; j < irrs.size(); j++)
    {
      const auto& b = bra_extract[j];
      const auto& b_pos = bra_pos[j];

      Eigen::SparseMatrix<double> mat_extract(b.size(), k.size());

      for (size_t local_b = 0; local_b < b_pos.size(); local_b++)
        for (size_t local_k = 0; local_k < k_pos.size(); local_k++)
        {
          double m = mat.coeff(b_pos[local_b], k_pos[local_k]);
          if (std::abs(m) > 1e-10)
            mat_extract.insert(local_b, local_k) = m;
        }

      double me = b.transpose() * mat_extract * k;
      std::cout << "Contribution of (" << j << "," << i << "): " << me * factor << std::endl;
    }
  }
*/

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  ////////////// DON'T EDIT ANYTHING HERE //////////////
  
  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;


  sam::finalize();
}
