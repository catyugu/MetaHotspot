#pragma once

#include "core/csr_matrix.hpp"

namespace mhs::core {
    core::CSRMatrix<double> build_csr(std::size_t rows, std::size_t cols, core::Vector<core::Triplet<double>> entries);

    void spmv(core::CSRMatrixView<double> a, std::span<const double> x, std::span<double> y, double alpha = 1.0,
        double beta = 0.0);

    core::CSRMatrix<double> add_scaled(
        const core::CSRMatrix<double>& a, double alpha, const core::CSRMatrix<double>& b);
} // namespace mhs::core
