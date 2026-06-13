#pragma once

#include "assembler/assembler.hpp"
#include "data/internal_model.hpp"
#include "data/time_step_buffer.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace mhs::sim::time_scheme {

    enum class TimeSchemeKind { Bdf1, Bdf2, AdaptiveBdf };

    /// Time-scheme parameters.  Values are read from IO; the algorithm
    /// implementation clamps internal usage to physical bounds.
    struct TimeSchemeConfig {
        TimeSchemeKind kind = TimeSchemeKind::Bdf1;

        // Fixed step (BDF1/BDF2) and adaptive control
        double initial_dt = 1.0;
        double min_dt     = 1e-9;
        double max_dt     = 1.0;

        // Adaptive error tolerances
        double abs_tol = 1e-6;
        double rel_tol = 1e-3;

        // Order cap for adaptive runs
        std::size_t max_order = 2;

        // Safety factor for adaptive dt update
        double safety = 0.9;

        // Output cadence
        double output_dt = 0.0; // 0 == single output at t_end

        // Termination guard
        int max_internal_steps = 100000;
    };

    struct StepDecision {
        double       dt    = 0.0;
        std::size_t  order = 1;
    };

    enum class AcceptDecision { Accept, Reject };

    /// Abstract time-scheme.  Owns no mutable state — the history buffer and
    /// candidate fields are passed in/out.  Implementations are pure.
    class TimeScheme {
    public:
        virtual ~TimeScheme() = default;

        /// Seed history with the initial temperature (called once at the start
        /// of the transient loop).
        virtual void initialize(
            mhs::core::TimeStepBuffer& history, mhs::core::GlobalState& state) const = 0;

        /// Decide the next (dt, order) given the current time and the history
        /// snapshots available so far.
        virtual StepDecision select_step(
            const mhs::core::TimeStepBuffer& history, double current_t) const = 0;

        /// Build the LinearSystem for a given order/dt from a (K, f_static)
        /// and (M_diag) decomposition.  The time scheme fills in the
        /// discretization-specific coefficients.
        virtual LinearSystem build_system(const StaticOpsResult& sops, const MassOpsResult& mops,
            const mhs::core::TimeStepBuffer& history, std::size_t order, double dt) const = 0;

        /// Decide whether to accept the candidate T.  For BDF1, always Accept.
        /// Implementations may also return an error estimate to feed into the
        /// controller's next-step dt choice.
        virtual AcceptDecision accept_or_reject(const mhs::core::TimeStepBuffer& history_before,
            const std::vector<double>& T_candidate, const std::vector<double>& error_estimate) const = 0;

        virtual const TimeSchemeConfig& config() const = 0;
    };

    /// Backward Euler (1st order) with fixed step.  Declared in bdf1_scheme.hpp.
    // (class Bdf1Scheme : public TimeScheme { ... })  // see bdf1_scheme.hpp

    /// Factory: returns the right TimeScheme subclass for a given config.
    /// Forward-declared; defined in time_scheme.cpp.
    std::unique_ptr<TimeScheme> create_scheme(const TimeSchemeConfig& cfg);

} // namespace mhs::sim::time_scheme
