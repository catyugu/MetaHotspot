#pragma once

#include "data/internal_model.hpp"
#include "data/linear_system.hpp"

namespace mhs::sim {

    class Assembler {
    public:
        explicit Assembler(const mhs::core::InternalModel& model) : model_(model) { }
        ~Assembler() = default;

        /// Build K, f, M_diag in a single sweep over the active grid.
        /// Diffusion and BC terms use state.T; the mass term uses
        /// state.accepted.current() (or state.T if history is empty) to stay
        /// constant across Newton iterations.
        AssemblyResult assemble(const mhs::core::GlobalState& state) const;

    private:
        const mhs::core::InternalModel& model_;

        /// Merge advection contributions (upwind) into K and f.
        /// Called after the main diffusion assembly when fluid cells exist.
        void assembleAdvection(Eigen::SparseMatrix<double>& K, Eigen::VectorXd& f,
            const mhs::core::GlobalState& state) const;
    };

} // namespace mhs::sim
