#pragma once

#include <functional>

#include "common/internal_model.hpp"
#include "solver/solver.hpp"

namespace mhs {

    struct SchedulerConfig {
        double transient_duration = 0.0;
        double time_step = 1.0;
        int max_nonlinear_iterations = 50;
        double nonlinear_tolerance = 1e-6;
        double underrelaxation = 1.0;
        bool is_steady = false;
    };

    // 时间步 / 稳态求解完成后的回调。
    // - 瞬态: 在每步 nonlinear::solve 完成后、current_time += dt 之前触发。
    // - 稳态: 在 run() 末尾触发一次，保持接口统一。
    // - t=0 时刻（初始状态）会在循环开始前额外触发一次，以记录初值。
    // cell_T 与 GlobalState::T 同步（cell 中心温度）。
    using StepCallback = std::function<void(double time, int step, const std::vector<double>& cell_T)>;

    class Scheduler {
    public:
        Scheduler() = default;
        explicit Scheduler(const SchedulerConfig& config) : config_(config) { }
        ~Scheduler() = default;

        void setModel(InternalModel* model) { model_ = model; }
        void setSolver(std::unique_ptr<Solver> solver);

        // 注册时间步回调；空 function 等价于无回调（稳态默认行为）。
        void setCallback(StepCallback callback) { callback_ = std::move(callback); }

        void run();
        const std::vector<double>& solution() const;

    private:
        InternalModel* model_ = nullptr;
        std::unique_ptr<Solver> solver_;
        SchedulerConfig config_;
        GlobalState state_;
        std::vector<double> solution_;
        StepCallback callback_;
    };

} // namespace mhs