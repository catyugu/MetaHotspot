#include "scheduler.hpp"
#include <vector>

namespace mhs {

    void Scheduler::setSolver(std::unique_ptr<Solver> solver)
    {
        solver_ = std::move(solver);
    }

    void Scheduler::run()
    {
    }

    void Scheduler::stepTime(double dt)
    {
        (void)dt;
    }

    void Scheduler::solveNonlinear(double t)
    {
        (void)t;
    }

    const std::vector<double>& Scheduler::solution() const
    {
        return solution_;
    }

} // namespace mhs
