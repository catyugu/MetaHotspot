#pragma once

#include <cstddef>
#include <vector>

namespace mhs::sim::time_scheme {

    class OutputTimeGrid {
    public:
        OutputTimeGrid() = default;

        /// Uniform grid: t_i = i * output_step for i in [0, floor(duration / output_step)].
        /// If output_step <= 0, the grid is empty (solver-only output).
        explicit OutputTimeGrid(double duration, double output_step)
        {
            if (duration <= 0.0 || output_step <= 0.0)
                return;
            const std::size_t n = static_cast<std::size_t>(duration / output_step);
            times_.reserve(n + 1);
            for (std::size_t i = 0; i <= n; ++i)
                times_.push_back(static_cast<double>(i) * output_step);
        }

        /// Explicit times (non-uniform).  Must be sorted ascending.
        explicit OutputTimeGrid(std::vector<double> times) : times_(std::move(times)) { }

        const std::vector<double>& times() const noexcept { return times_; }
        std::size_t size() const noexcept { return times_.size(); }
        bool empty() const noexcept { return times_.empty(); }

    private:
        std::vector<double> times_;
    };

    /// Strategy for coupling step-size control to the output-time grid.
    enum class StepStrategy {
        Free, ///< dt driven purely by error control; output via interpolation.
        Strict, ///< dt is clamped to hit output times exactly.
        Intermediate, ///< dt lands at least one non-output point between consecutive output times.
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
        /// Construct with strategy, step bounds, and Manual-mode fixed step.
        /// `output_dt` is set later by rebuild().
        StepController(StepStrategy strategy, double min_dt, double max_dt, double fixed_dt = 1.0);

        /// (Re)initialise for a new transient solve.  Builds the output-time grid
        /// from the known duration and the requested output interval, and resets
        /// all internal cursors.  `output_dt ≤ 0` means every accepted step is an
        /// output point (no grid).
        void rebuild(double duration, double output_dt);

        /// Pre-step: given a physics-suggested dt and the current time, return an
        /// adjusted dt that respects the chosen strategy and the remaining duration.
        double prepare(double dt_suggested, double current_t, double duration);

        /// Post-step: return output times that have been crossed since the last
        /// call to flush_outputs().  Each output time is returned at most once.
        std::vector<double> flush_outputs(double current_t);

        StepStrategy strategy() const noexcept { return strategy_; }
        double min_dt() const noexcept { return min_dt_; }
        double max_dt() const noexcept { return max_dt_; }

    private:
        StepStrategy strategy_ = StepStrategy::Free;
        OutputTimeGrid grid_; ///< Output-time grid (empty = record every step).
        double min_dt_ = 1e-8;
        double max_dt_ = 1.0;
        double fixed_dt_ = 1.0;

        std::size_t next_idx_ = 0; ///< First unconsumed grid index.
        double last_flushed_t_ = 0.0; ///< Last time returned by flush_outputs().
        bool planted_ = false; ///< Intermediate mode: have we planted a solve
                               ///< point inside the current output interval?
    };

} // namespace mhs::sim::time_scheme
