#pragma once

#include "model/internal_model.hpp"
#include "solver/solver.hpp"

namespace mhs {

    class Scheduler {
    public:
        Scheduler() = default;
        ~Scheduler() = default;

        void setModel(model::InternalModel* model) { model_ = model; }

        void setSolver(std::unique_ptr<Solver> solver);

        void run();

        const std::vector<double>& solution() const;

    private:
        void stepTime(double dt);
        void solveNonlinear(double t);

        model::InternalModel* model_ = nullptr;
        std::unique_ptr<Solver> solver_;
        std::vector<double> solution_;
    };

} // namespace mhs
