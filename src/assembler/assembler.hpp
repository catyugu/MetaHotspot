#pragma once

#include <Eigen/Sparse>

#include "data/internal_model.hpp"

namespace mhs::sim {

    struct LinearSystem {
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        Eigen::VectorXd residual;
    };

    /// Static (time-independent) operator result: stiffness K and static load
    /// vector f_static (boundary fluxes / heat sources / convective terms).
    /// Note: K is stored as -K internally (assemble convention) so the diagonal
    /// is non-positive for Neumann boundaries.  LinearSystem builder will
    /// re-add the mass contribution to give a positive-definite LHS.
    struct StaticOpsResult {
        Eigen::SparseMatrix<double> K;
        Eigen::VectorXd f_static;
    };

    /// Mass operator result: lumped diagonal mass vector M_diag, length N_active.
    /// M_diag[c] = rho(c) * c_p(c) * vol(c), evaluated at the T_prev state to
    /// keep the coefficient constant across Newton iterations (per legacy
    /// comment in the old assemble()).
    struct MassOpsResult {
        Eigen::VectorXd M_diag;
    };

    class Assembler {
    public:
        explicit Assembler(const mhs::core::InternalModel& model) : model_(model) { }
        ~Assembler() = default;

        /// Build only the static (time-independent) operator: K + f_static.
        /// Steady-state and time-scheme-aware code paths both start here.
        StaticOpsResult assemble_static(const mhs::core::GlobalState& state) const;

        /// Build the lumped diagonal mass vector M_diag for transient terms.
        /// M_diag[c] = rho(c) * c_p(c) * vol(c) (evaluated at state.T_prev to
        /// match the legacy BDF1 coefficient stability).
        MassOpsResult assemble_mass(const mhs::core::GlobalState& state) const;

    private:
        const mhs::core::InternalModel& model_;
    };

} // namespace mhs::sim
