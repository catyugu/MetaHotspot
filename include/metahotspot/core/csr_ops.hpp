#pragma once

#include "core/csr_matrix.hpp"

namespace mhs::core {
    core::CSRMatrix<double> build_csr(std::size_t rows, std::size_t cols, core::Vector<core::Triplet<double>> entries);

    void spmv(core::CSRMatrixView<double> a, std::span<const double> x, std::span<double> y, double alpha = 1.0,
        double beta = 0.0);

    core::CSRMatrix<double> add_scaled(
        const core::CSRMatrix<double>& a, double alpha, const core::CSRMatrix<double>& b);
    // Refill an existing union (or superset) pattern. Output numerical storage
    // must be distinct from the input values; every output slot is overwritten.
    void add_scaled_into(const core::CSRMatrix<double>& a, double alpha, const core::CSRMatrix<double>& b,
        core::CSRMatrix<double>& result);
} // namespace mhs::core
