#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include <HOutilities.h>

namespace U3xU4
{
  ///////////////////// UNCOUPLED EXPANSION STORAGE /////////////////////

  // Flat, row-major matrix of expansion coefficients: one contiguous buffer
  // instead of a heap allocation per row (replaces the old vector<RowVectorXd>).
  struct ExpansionCoeffs
  {
    std::vector<double> data;   // rows() * ncols, row-major
    uint8_t ncols = 0;          // expansion multiplicity, small

    // Getters
    std::size_t rows() const { return ncols ? data.size() / ncols : 0; }
    bool empty() const { return data.empty(); }
    const double* row(std::size_t i) const { return data.data() + i * ncols; }

    // Row manipulation
    void push_row(const double* src) { data.insert(data.end(), src, src + ncols); }
    void reserve_rows(std::size_t n) { data.reserve(n * ncols); }

    // Per-column sum of squares over all rows (a normalization check: each
    // column should sum to ~1 for a properly normalized expansion).
    std::vector<double> norm() const
    {
      std::vector<double> n(ncols, 0.0);
      for (std::size_t r = 0, nr = rows(); r < nr; ++r)
      {
        const double* p = row(r);
        for (uint32_t c = 0; c < ncols; ++c) n[c] += p[c] * p[c];
      }
      return n;
    }
  };

  // Uncoupled expansion shared by every *_phys tensor (Tad/Tat/SingleShell/
  // MultiShell): a flat coefficient matrix plus a row-major list of the
  // operators in each term, stored as bare Table_nljmjmt indices with a
  // creation/annihilation flag. nops operators per row (uniform). ad_phys/
  // at_phys objects are built only for printing; getStrengths() reads labels
  // straight from the table via op_labels().
  struct Expansion
  {
    ExpansionCoeffs coeffs;          // nrows x ncols
    uint8_t nops = 0;                // operators per row (uniform), <= ~8
    std::vector<uint16_t> op_index;  // nrows * nops : Table_nljmjmt indices
    std::vector<uint8_t>  op_is_ad;  // nrows * nops : 1 = ad (creation), 0 = a~
    const HO::Table_nljmjmt* labels = nullptr;   // runtime only (not serialized)

    // Getters
    std::size_t rows() const { return coeffs.rows(); }
    uint32_t ncols() const { return coeffs.ncols; }
    bool empty() const { return coeffs.empty(); }
    const double* coeff_row(std::size_t r) const { return coeffs.row(r); }

    // The k-th operator in term r
    uint16_t index(std::size_t r, std::size_t k) const
    { return op_index[r * nops + k]; }
    bool is_ad(std::size_t r, std::size_t k) const
    { return op_is_ad[r * nops + k]; }
    // labels from the table (no object required)
    const auto& op_labels(std::size_t r, std::size_t k) const
    { return labels->getLabels(index(r, k)); }

    // Fill term r's operator labels into the getME() label arrays starting at
    // offset o; returns the new offset. Reads indices -> labels directly.
    int fill_labels(std::size_t r, int o,
                    std::array<int, 4>& ns,  std::array<int, 4>& lls,
                    std::array<int, 4>& jjs, std::array<int, 4>& mmjs,
                    std::array<int, 4>& mmts) const
    {
      for (uint8_t k = 0; k < nops; ++k, ++o)
        std::tie(ns[o], lls[o], jjs[o], mmjs[o], mmts[o]) = op_labels(r, k);
      return o;
    }

    // Builders: one term at a time
    void push_coeff_row(const double* src) { coeffs.push_row(src); }
    void append_op(uint16_t idx, bool ad)
    { op_index.push_back(idx); op_is_ad.push_back(ad ? 1 : 0); }
    // append every operator of row r of src (used when combining sub-expansions)
    void append_ops_from(const Expansion& src, std::size_t r)
    {
      for (uint8_t k = 0; k < src.nops; ++k)
        append_op(src.index(r, k), src.is_ad(r, k));
    }

    friend std::ostream& operator<<(std::ostream& s, const Expansion& e);
  };
} // namespace U3xU4
