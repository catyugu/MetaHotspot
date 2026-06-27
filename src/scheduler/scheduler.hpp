#pragma once

#include "data/internal_model.hpp"
#include "linear_solver/linear_solver.hpp"
#include "scheduler/probe_recorder.hpp"
#include <memory>
#include <vector>

namespace mhs::sim {

    /// Scheduler for transient (and steady) thermal solves.
    ///
    /// Architecture: three orthogonal components composed in the time loop.
    ///   - Integrator     (pure linear algebra: BDF1 / BDF2)
    ///   - StepController (output-time strategy: Free / Strict / Intermediate / Manual)
    ///   - ErrorController (LTE estimation + PI-like step-size suggestion)
    class Scheduler {
    public:
        Scheduler() = default;

        void setModel(mhs::core::InternalModel* model);
        void setSolver(std::unique_ptr<LinearSolver> solver);

        void run();

        const std::vector<double>& solution() const noexcept { return solution_; }
        double currentTime() const noexcept { return state_.current_time; }
        const std::vector<mhs::core::ProbeTrace>& probeTraces() const noexcept { return probe_recorder_.traces(); }

    private:
        Scheduler(const Scheduler&) = delete;
        Scheduler& operator=(const Scheduler&) = delete;

        mhs::core::InternalModel* model_ = nullptr;
        std::unique_ptr<LinearSolver> solver_;
        mhs::core::GlobalState state_;
        std::vector<double> solution_;
        ProbeRecorder probe_recorder_;
    };

} // namespace mhs::sim
