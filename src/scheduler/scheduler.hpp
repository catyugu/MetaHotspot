#pragma once

#include "data/internal_model.hpp"
#include "linear_solver/linear_solver.hpp"
#include "scheduler/probe_recorder.hpp"
#include "time_scheme/error_controller.hpp"
#include "time_scheme/integrator.hpp"
#include "time_scheme/step_controller.hpp"
#include <memory>
#include <vector>

namespace mhs::sim {

    /// Scheduler for transient (and steady) thermal solves.
    ///
    /// Architecture: three orthogonal components composed in the time loop.
    ///   - Integrator     (pure linear algebra: BDF1 / BDF2)
    ///   - StepController (output-time strategy: Free / Strict / Intermediate / Manual)
    ///   - ErrorController (LTE estimation + PI-like step-size suggestion)
    ///
    /// The old TimeScheme virtual hierarchy is fully replaced.
    class Scheduler {
    public:
        Scheduler() = default;
        ~Scheduler() = default;

        // --- mandatory setup ---
        void setModel(mhs::core::InternalModel* model);
        void setSolver(std::unique_ptr<LinearSolver> solver);

        // --- time integration (default: Bdf1) ---
        void setIntegrator(time_scheme::IntegratorKind kind) noexcept { integrator_ = kind; }

        // --- step strategy (default: Free) ---
        void setStepStrategy(time_scheme::StepStrategy strategy);
        void setStepBounds(double min_dt, double max_dt);
        void setFixedDt(double dt) noexcept { fixed_dt_ = dt; }

        // --- error control (ignored in Manual mode) ---
        void setTolerance(double abs_tol) noexcept { err_cfg_.abs_tol = abs_tol; }
        void setSafety(double safety) noexcept { err_cfg_.safety = safety; }

        // --- run ---
        void run();

        // --- results ---
        const std::vector<double>& solution() const noexcept { return solution_; }
        double currentTime() const noexcept { return state_.current_time; }
        const std::vector<mhs::core::ProbeTrace>& probeTraces() const noexcept { return probe_recorder_.traces(); }

    private:
        // Erase copies.
        Scheduler(const Scheduler&) = delete;
        Scheduler& operator=(const Scheduler&) = delete;

        // State.
        mhs::core::InternalModel* model_ = nullptr;
        std::unique_ptr<LinearSolver> solver_;
        mhs::core::GlobalState state_;
        std::vector<double> solution_;
        ProbeRecorder probe_recorder_;

        // Composition.
        time_scheme::IntegratorKind integrator_ = time_scheme::IntegratorKind::Bdf1;
        time_scheme::StepController step_ctrl_;
        time_scheme::ErrorControlConfig err_cfg_;

        // Bounds.
        double min_dt_ = 1e-12;
        double max_dt_ = 1.0;
        double fixed_dt_ = 1.0; // used only in Manual mode
    };

} // namespace mhs::sim
