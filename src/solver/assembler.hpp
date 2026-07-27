#pragma once

#include "runtime/model.hpp"
#include "runtime/solution.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <functional>
#include <span>

namespace mhs::sim {

    struct AssemblyResult {
        Eigen::SparseMatrix<double> K;
        Eigen::SparseMatrix<double> C;
        Eigen::VectorXd f;
    };

    /// Runtime state used to evaluate state-dependent operators.
    ///
    /// Invariant:
    ///   - state.size() == layout.state_count
    struct AssembleContext {
        std::span<const double> state;
        double current_time = 0.0;
    };

    /// Assemble thermal operators C * dx/dt + K * x = f.
    /// Extracts the temperature slice from ctx.state via layout.temperature.
    AssemblyResult assemble_thermal(
        const mhs::core::Model& model,
        const mhs::core::StateLayout& layout,
        const AssembleContext& ctx);

    /// Pluggable assembly provider for experiments with extra DoFs.
    /// Signature: (full_state, time) -> AssemblyResult
    using AssemblyProvider = std::function<AssemblyResult(std::span<const double>, double)>;

} // namespace mhs::sim
