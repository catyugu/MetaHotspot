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
        /// state.history.latest() (or state.T if history is empty) to stay
        /// constant across Newton iterations.
        AssemblyResult assemble(const mhs::core::GlobalState& state) const;

    private:
        const mhs::core::InternalModel& model_;
    };

} // namespace mhs::sim
