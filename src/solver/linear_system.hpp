#pragma once

#include "core/csr_matrix.hpp"

namespace mhs::solver {
    /// Final linear system: A * x = b
    struct LinearSystem {
        mhs::core::CSRMatrix<double> A;
        mhs::core::Vector<double> b;
    };
} // namespace mhs::solver
