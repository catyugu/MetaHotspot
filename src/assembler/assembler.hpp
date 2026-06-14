#pragma once

#include "data/internal_model.hpp"
#include "data/linear_system.hpp"

namespace mhs::sim {

    class Assembler {
    public:
        explicit Assembler(const mhs::core::InternalModel& model) : model_(model) { }
        ~Assembler() = default;

        /// Build only the static (time-independent) operator: K + f_static.
        /// Steady-state and time-scheme-aware code paths both start here.
        StaticOpsResult assemble_static(const mhs::core::GlobalState& state) const;

        /// Build the lumped diagonal mass vector M_diag for transient terms.
        /// M_diag[c] = rho(c) * c_p(c) * vol(c) (evaluated at history.latest()).
        MassOpsResult assemble_mass(const mhs::core::GlobalState& state) const;

    private:
        const mhs::core::InternalModel& model_;
    };

} // namespace mhs::sim
