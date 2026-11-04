#pragma once

#include "core/csr_matrix.hpp"

namespace mhs::solver {
    /// Final linear system: A * x = b
    struct LinearSystemView {
        mhs::core::CSRMatrixView<double> A;
        std::span<const double> b;
    };
} // namespace mhs::solver
