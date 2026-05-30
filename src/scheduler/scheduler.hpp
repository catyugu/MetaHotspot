#pragma once

#include "model/internal_model.hpp"
#include "solver/solver.hpp"

namespace mhs {

    struct SchedulerConfig {
        double transient_duration = 0.0;
        double time_step = 1.0;
        int max_newton_iterations = 50;
        double newton_tolerance = 1e-6;
        double underrelaxation = 1.0;
        bool is_steady = false;
        int ring_buffer_capacity = 5;
    };

    class Scheduler {
    public:
        Scheduler() = default;
        explicit Scheduler(const SchedulerConfig& config) : config_(config) { }
        ~Scheduler() = default;

        void setModel(model::InternalModel* model) { model_ = model; }
        void setSolver(std::unique_ptr<Solver> solver);

        void run();
        const std::vector<double>& solution() const;

    private:
        bool solve_nonlinear_step();
        void step_time(double dt);

        model::InternalModel* model_ = nullptr;
        std::unique_ptr<Solver> solver_;
        SchedulerConfig config_;
        model::GlobalState state_;
        std::vector<double> solution_;
        double current_time_ = 0.0;
        int current_step_ = 0;
    };

} // namespace mhs