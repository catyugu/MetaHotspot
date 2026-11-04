#pragma once

#include "core/csr_matrix.hpp"

namespace mhs::solver {
    /// Operators of the linearised system: C * dx/dt + K * x = f.
    struct Operators {
        mhs::core::CSRMatrix<double> K;
        mhs::core::CSRMatrix<double> C;
        mhs::core::Vector<double> f;
    };

    /// Final linear system: A * x = b
    struct LinearSystemView {
        mhs::core::CSRMatrixView<double> A;
        std::span<const double> b;
    };
} // namespace mhs::solver
