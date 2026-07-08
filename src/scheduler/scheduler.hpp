#pragma once

#include "data/model.hpp"
#include "data/solution_history.hpp"
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

        void setModel(mhs::core::Model* model);
        void setSolver(std::unique_ptr<LinearSolver> solver);

        void run();

        const std::vector<double>& solution() const noexcept { return solution_; }
        double currentTime() const noexcept { return step_.current_time; }
        const std::vector<mhs::core::ProbeTrace>& probeTraces() const noexcept { return probe_recorder_.traces(); }

    private:
        Scheduler(const Scheduler&) = delete;
        Scheduler& operator=(const Scheduler&) = delete;

        struct StepState {
            double current_time = 0.0;
            int time_step = 0;
            double dt = 0.0;
            mhs::core::SolutionHistory accepted {0, 1};
            std::vector<double> T;
        };

        mhs::core::Model* model_ = nullptr;
        std::unique_ptr<LinearSolver> solver_;
        StepState step_;
        std::vector<double> solution_;
        ProbeRecorder probe_recorder_;
    };

} // namespace mhs::sim
