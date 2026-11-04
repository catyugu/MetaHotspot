#pragma once

#include "core/csr_matrix.hpp"

#include "model/model.hpp"
#include <span>

namespace mhs::solver {

    /// Operators K, C, f of the linearised system: C * dx/dt + K * x = f.
    struct Operators {
        mhs::core::CSRMatrix<double> K;
        mhs::core::CSRMatrix<double> C;
        mhs::core::Vector<double> f;
    };

    /// Assemble thermal operators C * dx/dt + K * x = f.
    Operators assemble_thermal(const mhs::model::Model& model, std::span<const double> temperature, double time);

} // namespace mhs::solver
