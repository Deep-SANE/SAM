#include <ranges>

#include <OperatorJT.h>

namespace JT
{
  // Expectation value
  using Col_t = Eigen::MatrixXd::ColXpr;
  // within a single basis (default: block-distributed symmetric CSR + vec-mat-
  // vec, MPI scalar reduce); users can pass multiple eigenstate pairs
  std::vector<double> Operator::expectationValue(
    const U3xU4::ManyNucleonBasis& basis,
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    std::vector<double> res;
    res.reserve(states.size());

    auto dim = basis.dim();

    auto mat = matrix_CSR(basis);

    for (auto& [bra, ket] : states)
    {
      assert(bra.size() == static_cast<Eigen::Index>(dim));
      assert(ket.size() == static_cast<Eigen::Index>(dim));

      // contiguous copies for the (symmetric) vec-mat-vec
      Eigen::VectorXd bv = bra;
      Eigen::VectorXd kv = ket;

      double val = mat.quadratic_form(bv.data(), kv.data());
#ifdef USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      res.push_back(val);
    }

    if (name_ == "M1")
    {
      double factor = std::sqrt(4.0 * PI / 3.0);
      for (auto& r : res) r *= factor;
    }

    if (name_ == "Radius")
    {
      double A = N_ + Z_, com = 1.5 * HO::oscilatorLength2(hw_, -1);
      for (auto& r2 : res) r2 = std::sqrt((r2 - com) / A);
    }

    if (name_ == "Radius_p")
    {
      double A = N_ + Z_, com = 1.5 * HO::oscilatorLength2(hw_, -1);
      for (auto& r2 : res) r2 = std::sqrt(r2 / Z_ - com / A);
    }

    if (name_ == "Radius_n")
    {
      double A = N_ + Z_, com = 1.5 * HO::oscilatorLength2(hw_, -1);
      for (auto& r2 : res) r2 = std::sqrt(r2 / N_ - com / A);
    }

    return res;
  }

  // within a single basis, dense Eigen matrix; users can pass multiple
  // eigenstate pairs
  std::vector<double> Operator::expectationValue_Eigen(
    const U3xU4::ManyNucleonBasis& basis,
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    std::vector<double> res;
    res.reserve(states.size());

    auto dim = basis.dim();

    auto mat = matrix_Eigen(basis);

    for (auto& [bra, ket] : states)
    {
      assert(bra.size() == dim);
      assert(ket.size() == dim);
      double val = bra.transpose() * mat * ket;
      res.push_back(val);
    }

    if (name_ == "M1")
    {
      double factor = std::sqrt(4.0 * PI / 3.0);
      for (auto& r : res) r *= factor;
    }

    if (name_ == "Radius") 
    {
      double A = N_ + Z_, com = 1.5 * HO::oscilatorLength2(hw_, -1);
      for (auto& r2 : res) r2 = std::sqrt((r2 - com) / A);
    }

    if (name_ == "Radius_p") 
    {
      double A = N_ + Z_, com = 1.5 * HO::oscilatorLength2(hw_, -1);
      for (auto& r2 : res) r2 = std::sqrt(r2 / Z_ - com / A);
    }

    if (name_ == "Radius_n") 
    {
      double A = N_ + Z_, com = 1.5 * HO::oscilatorLength2(hw_, -1);
      for (auto& r2 : res) r2 = std::sqrt(r2 / N_ - com / A);
    }

    return res;
  }

  // transition between two bases, dense Eigen matrix; users can pass multiple
  // eigenstate pairs
  std::vector<double> Operator::expectationValue_Eigen(
    const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    std::vector<double> res;
    res.reserve(states.size());

    auto bras_dim = bras.dim();
    auto kets_dim = kets.dim();

    auto mat = matrix_Eigen(bras, kets);

    for (auto& [bra, ket] : states)
    {
      assert(bra.size() == bras_dim);
      assert(ket.size() == kets_dim);
      double val = bra.transpose() * mat * ket;
      res.push_back(val);
    }

    return res;
  }

  // Transition strength
  // within a single basis; users can pass multiple eigenstate pairs
  std::vector<double> Operator::transitionStrength(
    const U3xU4::ManyNucleonBasis& basis, 
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    auto& cgcs = utils::cgcs();
    
    auto res = expectationValue(basis, states);
    
    double factor = 1.0;

    if (name_ == "E2")
    {
      auto JJ = basis.JJ();
      auto cgc = cgcs.get({JJ, 4, JJ, JJ, 0, JJ});
      factor = 5.0 / (16.0 * PI) / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }
    else if (name_ == "M1")
    {
      auto JJ = basis.JJ();
      auto cgc = cgcs.get({JJ, 2, JJ, JJ, 0, JJ});
      factor = 1.0 / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }

    return res;
  }

  // within a single basis, dense Eigen matrix; users can pass multiple
  // eigenstate pairs
  std::vector<double> Operator::transitionStrength_Eigen(
    const U3xU4::ManyNucleonBasis& basis,
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    auto& cgcs = utils::cgcs();

    auto res = expectationValue_Eigen(basis, states);

    double factor = 1.0;

    if (name_ == "E2")
    {
      auto JJ = basis.JJ();
      auto cgc = cgcs.get({JJ, 4, JJ, JJ, 0, JJ});
      factor = 5.0 / (16.0 * PI) / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }
    else if (name_ == "M1")
    {
      auto JJ = basis.JJ();
      auto cgc = cgcs.get({JJ, 2, JJ, JJ, 0, JJ});
      factor = 1.0 / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }

    return res;
  }

  // transition between two bases; users can pass multiple eigenstate pairs
  std::vector<double> Operator::transitionStrength(
    const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    auto& cgcs = utils::cgcs();

    auto JJf = bras.JJ();
    auto JJi = kets.JJ();
    auto MMJ = std::min(JJf, JJi);

    auto res = expectationValue(bras, kets, states);

    if (name_ == "E2")
    { 
      auto cgc = cgcs.get({JJi, 4, JJf, MMJ, 0, MMJ});
      double factor = 5.0 / (16.0 * PI) * (JJf + 1.0) / (JJi + 1.0) / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }
    else if (name_ == "M1")
    {
      auto cgc = cgcs.get({JJi, 2, JJf, MMJ, 0, MMJ});
      double factor = (JJf + 1.0) / (JJi + 1.0) / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }
    else if (name_ == "GamowTellerPlus" || name_ == "GamowTellerMinus" ||
             name_ == "GamowTellerPlus_2B_N3LO_EM500_bare" ||
             name_ == "GamowTellerMinus_2B_N3LO_EM500_bare")
    {
      auto cgc = cgcs.get({JJi, 2, JJf, MMJ, 0, MMJ});
      double factor = std::sqrt(JJf + 1.0) / cgc;
      for (auto& r : res) r *= factor;
    }

    return res;
  }

  // transition between two bases, default CSR path: build the block-distributed
  // rectangular CSR and evaluate bra^T M ket as a single-pass vec-mat-vec
  // (CSRMatrix::quadratic_form) with the per-rank partial scalar reduced via
  // MPI. Never materializes the full matrix on any rank -- the memory-lean path
  // for 2-body two-basis operators.
  std::vector<double> Operator::expectationValue(
    const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    std::vector<double> res;
    res.reserve(states.size());

    auto bras_dim = bras.dim();
    auto kets_dim = kets.dim();

    // GamowTellerPlus (2B) comes back transposed, in (kets x bras) layout
    // (see matrix_CSR). Swap the bra/ket roles in the vec-mat-vec so we still
    // evaluate the requested operator without transposing the matrix.
    const bool swap = (name_ == "GamowTellerPlus_2B_N3LO_EM500_bare");

    auto mat = matrix_CSR(bras, kets);

    for (auto& [bra, ket] : states)
    {
      assert(bra.size() == static_cast<Eigen::Index>(bras_dim));
      assert(ket.size() == static_cast<Eigen::Index>(kets_dim));

      // contiguous copies for the SpMV
      Eigen::VectorXd bv = bra;
      Eigen::VectorXd kv = ket;

      // left^T * M * right, local partial when block-distributed.
      //   normal: M is (bras x kets) -> left = bra (nrows), right = ket (ncols)
      //   swap:   M is (kets x bras) -> left = ket (nrows), right = bra (ncols)
      double val = swap
        ? mat.quadratic_form(kv.data(), bv.data())
        : mat.quadratic_form(bv.data(), kv.data());

#ifdef USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      res.push_back(val);
    }

    return res;
  }

  // transition between two bases, dense Eigen matrix; users can pass multiple
  // eigenstate pairs
  std::vector<double> Operator::transitionStrength_Eigen(
    const U3xU4::ManyNucleonBasis& bras, const U3xU4::ManyNucleonBasis& kets,
    const std::vector<std::pair<Col_t, Col_t>>& states)
  {
    auto& cgcs = utils::cgcs();

    auto JJf = bras.JJ();
    auto JJi = kets.JJ();
    auto MMJ = std::min(JJf, JJi);

    auto res = expectationValue_Eigen(bras, kets, states);

    if (name_ == "E2")
    {
      auto cgc = cgcs.get({JJi, 4, JJf, MMJ, 0, MMJ});
      double factor = 5.0 / (16.0 * PI) * (JJf + 1.0) / (JJi + 1.0) / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }
    else if (name_ == "M1")
    {
      auto cgc = cgcs.get({JJi, 2, JJf, MMJ, 0, MMJ});
      double factor = (JJf + 1.0) / (JJi + 1.0) / cgc / cgc;
      for (auto& r : res) r *= (r * factor);
    }
    else if (name_ == "GamowTellerPlus" || name_ == "GamowTellerMinus" ||
             name_ == "GamowTellerPlus_2B_N3LO_EM500_bare" ||
             name_ == "GamowTellerMinus_2B_N3LO_EM500_bare")
    {
      auto cgc = cgcs.get({JJi, 2, JJf, MMJ, 0, MMJ});
      double factor = std::sqrt(JJf + 1.0) / cgc;
      for (auto& r : res) r *= factor;
    }

    return res;
  }
} // namespace JT
