#pragma once

#include "Eigen/SparseCore"
#include "runtime/model.hpp"
#include <Eigen/Core>
#include <vector>

namespace mhs::sim {

    struct AssemblyResult {
        Eigen::SparseMatrix<double> K;
        Eigen::SparseMatrix<double> C;
        Eigen::VectorXd f;
    };

    /// Runtime state used to evaluate state-dependent operators.
    ///
    /// Invariant (caller-enforced):
    ///   - state.size() == model.dofs.total_count
    struct AssembleContext {
        const std::vector<double>& state;
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
