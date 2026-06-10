#pragma once

#include "common/internal_model.hpp"
#include "linear_solver/linear_solver.hpp"
#include "scheduler/probe_recorder.hpp"

#include <memory>
#include <vector>

namespace mhs::sim {

    class Scheduler {
    public:
        Scheduler() = default;
        ~Scheduler() = default;

        void setModel(mhs::core::InternalModel* model);
        void setSolver(std::unique_ptr<LinearSolver> solver);

        void run();
        const std::vector<double>& solution() const;

        // 求解结束时的当前时刻（稳态恒为 0.0；瞬态为最后一个步末的时间）。
        // postprocessor 调 FieldContext.t 时需要此值。
        double currentTime() const { return state_.current_time; }

        // 探针温度时间序列：仅当 (Transient && observation_points 非空) 时非空。
        // 长度与 model->observation_points 一一对应；每个 trace 的 times/values
        // 长度为 steps+1（包含 t=0 初始 + N 个步末）。
        const std::vector<mhs::core::ProbeTrace>& probeTraces() const { return probe_recorder_.traces(); }

    private:
        mhs::core::InternalModel* model_ = nullptr;
        std::unique_ptr<LinearSolver> solver_;
        mhs::core::GlobalState state_;
        std::vector<double> solution_;
        ProbeRecorder probe_recorder_;
    };

} // namespace mhs::sim
