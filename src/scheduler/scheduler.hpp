#pragma once

#include "data/internal_model.hpp"
#include "linear_solver/linear_solver.hpp"
#include "scheduler/probe_recorder.hpp"
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
        time_scheme::StepController step_ctrl_;
    };

} // namespace mhs::sim
