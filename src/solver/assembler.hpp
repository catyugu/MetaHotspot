#pragma once

#include "runtime/model.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <functional>
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

    /// Pluggable assembly callback for experiments with extra DoFs.
    /// Signature: (full_state, time) -> Operators
    using Assemble = std::function<Operators(std::span<const double>, double)>;

} // namespace mhs::sim
