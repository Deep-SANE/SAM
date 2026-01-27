#include <chrono>
#include <iomanip>
#include <iostream>

#include <eigensolver.h>
#include <qcx_readwrite.h>
#include <qcx_utils.h>

#include <u3lib.h>
#include <u4lib.h>

#include <OperatorJT.h>

int main(int argc, char* argv[])
{
  qcx_utils::initialize(&argc, &argv);
  u3::initialize();
  u4::initialize();
  /*
  int Z, N, Nmax, JJ, hw, write;

  std::cout << "Enter number of protons: ";
  std::cin >> Z;
  std::cout << "Enter number of neutrons: ";
  std::cin >> N;
  std::cout << "Enter Nmax: ";
  std::cin >> Nmax;
  std::cout << "Enter JJ: ";
  std::cin >> JJ;
  std::cout << "Enter hw (-1 for default): ";
  std::cin >> hw;
  std::cout << "Enter 1 to write matrix to binary file: ";
  std::cin >> write;
  */

  int Z = 3, N = 3, JJ = 2, Nmax = 4, hw = 20, write = 0;

  // Example 1: LL
  // std::string name = "L.L";

  // Example 2: LS
  // std::string name = "L.S";

  // Example 3: relative kinetic energy
  // std::string name = "Kinetic";  

  // Example 4: CoM HO operator
  // std::string name = "CoMNumberOp";  

  // Example 6: Hamiltonian with N2LO_opt_bare
  std::string name = "Hamiltonian with N2LO_opt_bare";

  // Example 7: Hamiltonian with N3LO_EMN500_bare
  // std::string name = "Hamiltonian with N3LO_EMN500_bare";

  // Example 8: Hamiltonian with N4LO_EMN500_bare
  // std::string name = "Hamiltonian with N4LO_EMN500_bare";

  ////////////// DON'T EDIT ANYTHING BELOW //////////////

  U3xU4::ManyNucleonBasis basis(N, Z, Nmax, JJ);
  basis.construct();
  std::cout << "Basis dimension: " << basis.dim() << std::endl;

  HO::Table_nljmjmt HO_labels(basis.maxShell());
  U3xU4::Table_gelfand gelfand_labels(basis.maxShell());

  JT::Operator op(name, N, Z, hw, Nmax, HO_labels, gelfand_labels);
  op.is();

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  auto mat = op.matrix(basis);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  auto [evals, evecs] = qcx_utils::eigensolver_symmetric(mat, 10);
  std::cout << "Eigenvalues are" << std::endl;
  for (auto ev : evals)
  {
    if (std::fabs(ev) < 1e-10)
      std::cout << 0 << std::endl;
    else
      std::cout << std::setprecision(8) << ev << std::endl;
  }

  if (write == 1)
  {
    auto filename = op.filename(basis);
    qcx_utils::writeBinary(filename, mat);
  }

  // L.L and L.S are analytical
  if (name == "L.L" || name == "L.S")
  {
    auto irreps = basis.irreps();
    auto labels = basis.decomposition();
    std::vector<double> analyticals;

    for (size_t irrep_id = 0; irrep_id < irreps.size(); irrep_id++)
    {
      auto alphamax = irreps[irrep_id].alphamax();

      for (const auto& [kmax, LL, zmax, SS, TT, offset] : labels[irrep_id])
      {
        for (size_t j = 0; j < alphamax * kmax * zmax; j++)
        {
          if (name == "L.L")
            analyticals.push_back(LL * (LL + 2.0) / 4.0);
          else
          {
            double jj = JJ * (JJ + 2.0) / 4.0, 
                   ll = LL * (LL + 2.0) / 4.0,
                   ss = SS * (SS + 2.0) / 4.0;
            analyticals.push_back((jj - ll - ss) / 2.0);
          }
        }
      }
    }
  
    // std::cout << "Total matrix representation is " << std::endl;
    // std::cout << mat << std::endl;
    for (int k = 0; k < mat.outerSize(); ++k) 
      for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) 
      {
        if (std::fabs(it.value()) > 1e-10)
        {
          std::cout << "(" << it.row() << "," << it.col() << "): "
                    << it.value() << "\t";
          if (it.row() != it.col())
            std::cout << "Off-diagonal\t";
          if (std::fabs(analyticals[it.row()] - it.value()) > 1e-10)
            std::cout << "Not analytical";
          std::cout << std::endl;
        }
      }
  }
 
  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  u3::finalize();
  u4::finalize();
  qcx_utils::finalize();
}
