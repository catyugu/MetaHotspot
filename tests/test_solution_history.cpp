#include "solver/solve.hpp"

#include <Eigen/Sparse>
#include <array>
#include <gtest/gtest.h>
#include <vector>

TEST(SolutionHistoryTest, RecordsEveryOutputStateInRowMajorOrder)
{
    mhs::sim::Study study {mhs::core::StudyType::Transient, 0.5, 0.25};
    mhs::sim::SolveOptions options;
    options.linear_solver = mhs::sim::SolveOptions::LinearSolverType::AmgCg;
    options.integrator = mhs::sim::SolveOptions::Integrator::Bdf1;
    options.step_strategy = mhs::sim::SolveOptions::StepStrategy::Fixed;
    options.fixed_dt = 0.25;
    options.min_dt = 0.25;
    options.max_dt = 0.25;

    mhs::sim::SystemAssembler assemble = [](std::span<const double>, double) {
        mhs::sim::Operators operators;
        operators.K.resize(1, 1);
        operators.K.insert(0, 0) = 1.0;
        operators.C.resize(1, 1);
        operators.C.insert(0, 0) = 1.0;
        operators.f = Eigen::VectorXd::Ones(1);
        return operators;
    };

    const std::array initial {0.0};
    const auto solution = mhs::sim::solve_system(study, assemble, initial, options);

    ASSERT_TRUE(solution.converged);
    ASSERT_EQ(solution.snapshot_times, (std::vector<double> {0.0, 0.25, 0.5}));
    ASSERT_EQ(solution.snapshot_states.size(), 3u);
    EXPECT_NEAR(solution.snapshot_states[0], 0.0, 1e-14);
    EXPECT_NEAR(solution.snapshot_states[1], 0.2, 1e-14);
    EXPECT_NEAR(solution.snapshot_states[2], 0.36, 1e-14);
    EXPECT_NEAR(solution.state[0], solution.snapshot_states.back(), 1e-14);
}

TEST(SolutionHistoryTest, SteadySolveProducesOneSnapshot)
{
    mhs::sim::Study study {mhs::core::StudyType::Steady, 0.0, 1.0};
    mhs::sim::SolveOptions options;
    options.linear_solver = mhs::sim::SolveOptions::LinearSolverType::AmgCg;

    mhs::sim::SystemAssembler assemble = [](std::span<const double>, double) {
        mhs::sim::Operators operators;
        operators.K.resize(1, 1);
        operators.K.insert(0, 0) = 2.0;
        operators.C.resize(1, 1);
        operators.f = Eigen::VectorXd::Constant(1, 6.0);
        return operators;
    };

    const std::array initial {0.0};
    const auto solution = mhs::sim::solve_system(study, assemble, initial, options);

    ASSERT_TRUE(solution.converged);
    ASSERT_EQ(solution.snapshot_times, (std::vector<double> {0.0}));
    ASSERT_EQ(solution.snapshot_states.size(), 1u);
    EXPECT_NEAR(solution.snapshot_states[0], 3.0, 1e-14);
}