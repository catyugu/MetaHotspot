#pragma once

#include "core/model.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <span>

namespace mhs::sim {

    /// Operators K, C, f of the linearised system: C * dx/dt + K * x = f.
    struct Operators {
        Eigen::SparseMatrix<double> K;
        Eigen::SparseMatrix<double> C;
        Eigen::VectorXd f;
    };

    /// Assemble thermal operators C * dx/dt + K * x = f.
    Operators assemble_thermal(const mhs::core::Model& model, std::span<const double> temperature, double time);

} // namespace mhs::sim
