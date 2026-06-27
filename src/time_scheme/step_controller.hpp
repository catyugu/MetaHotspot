#pragma once

#include "time_scheme/output_time_grid.hpp"
#include <cstddef>
#include <vector>

namespace mhs::sim::time_scheme {

    /// Strategy for coupling step-size control to the output-time grid.
    enum class StepStrategy {
        Free, ///< dt driven purely by error control; output via interpolation.
        Strict, ///< dt is clamped to hit output times exactly.
        Intermediate, ///< dt lands at least one non-output point between consecutive
                      ///< output times.  After planting, behaves like Strict until
                      ///< the output point is reached.
        Manual ///< Fixed dt, no error-based adjustment.
    };

    /// Manages the interaction between the physics-driven time step and the
    /// user-requested output time grid.
    ///
    /// Usage inside the time loop:
    ///   step_ctrl.rebuild(duration);                        // once before the loop
    ///   while (t < duration) {
    ///     dt = step_ctrl.prepare(dt_phys, t);               // pre-step adjustment
    ///     // ... solve: t → t + dt ...
    ///     t = t + dt_actual;
    ///     auto out = step_ctrl.flush_outputs(t);            // post-step consumption
    ///     for (double t_out : out) { ... record/interpolate ... }
    ///   }
    ///
    /// Thread safety: not thread-safe.  One StepController per transient solve.
    class StepController {
    public:
        StepController() = default;

        /// Construct with strategy and parameter bounds.
        /// \param output_dt  User-requested output interval.  ≤ 0 means every
        ///                   accepted step is an output point (no grid).
        /// \param fixed_dt   Step size used in Manual mode (ignored otherwise).
        StepController(StepStrategy strategy, double output_dt, double min_dt, double max_dt, double fixed_dt = 1.0);

        /// (Re)initialise for a new transient solve.  Builds the output-time grid
        /// from the known duration and resets all internal cursors.
        void rebuild(double duration);

        /// Pre-step: given a physics-suggested dt and the current time, return an
        /// adjusted dt that respects the chosen strategy and the remaining duration.
        double prepare(double dt_suggested, double current_t, double duration);

        /// Post-step: return output times that have been crossed since the last
        /// call to flush_outputs().  Each output time is returned at most once.
        std::vector<double> flush_outputs(double current_t);

        StepStrategy strategy() const noexcept { return strategy_; }

    private:
        StepStrategy strategy_ = StepStrategy::Free;
        OutputTimeGrid grid_; ///< Output-time grid (empty = record every step).
        double min_dt_ = 1e-12;
        double max_dt_ = 1.0;
        double fixed_dt_ = 1.0;
        double output_dt_ = 0.0;

        std::size_t next_idx_ = 0; ///< First unconsumed grid index.
        double last_flushed_t_ = 0.0; ///< Last time returned by flush_outputs().
        bool planted_ = false; ///< Intermediate mode: have we planted a solve
                               ///< point inside the current output interval?
    };

} // namespace mhs::sim::time_scheme
