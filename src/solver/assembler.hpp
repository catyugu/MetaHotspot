#pragma once

#include "Eigen/SparseCore"
#include "runtime/model.hpp"
#include <Eigen/Core>

namespace mhs::sim {

    struct AssemblyResult {
        Eigen::SparseMatrix<double> K;
        Eigen::VectorXd f;
        Eigen::VectorXd M_diag;
    };
    /// Minimum data needed by Assembler::assemble to evaluate one cell sweep.
    ///
    /// Invariant (caller-enforced):
    ///   - T.size() == N_active
    struct AssembleContext {
        Eigen::Ref<const Eigen::VectorXd> T;
        double current_time = 0.0;
    };

    class Assembler {
    public:
        explicit Assembler(const mhs::core::Model& model) : model_(model) { }
        ~Assembler() = default;

        /// Build K, f, M_diag in a single sweep over the active grid.
        /// Diffusion coefficients, BC terms, heat sources, and mass coefficients
        /// are all evaluated at ctx.T / ctx.current_time.
        AssemblyResult assemble(const AssembleContext& ctx) const;

    private:
        const mhs::core::Model& model_;
    };

} // namespace mhs::sim
