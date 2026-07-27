#pragma once

#include "Eigen/SparseCore"
#include "runtime/model.hpp"
#include <Eigen/Core>
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
    ///   - state.size() == model.cells.cell_to_grid.size()
    struct AssembleContext {
        std::span<const double> state;
        double current_time = 0.0;
    };

    class Assembler {
    public:
        explicit Assembler(const mhs::core::Model& model) : model_(model) { }
        ~Assembler() = default;

        /// Assemble C * dx/dt + K * x = f over the complete state layout.
        AssemblyResult assemble(const AssembleContext& ctx) const;

    private:
        const mhs::core::Model& model_;
    };

} // namespace mhs::sim
