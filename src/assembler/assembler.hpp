#pragma once

#include <Eigen/Core>

#include "data/linear_system.hpp"
#include "data/model.hpp"

namespace mhs::sim {

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