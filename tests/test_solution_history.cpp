#include "core/csr_ops.hpp"
#include "core/vector_ops.hpp"
#include "solver/solve.hpp"

#include "core/vector.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

TEST(SolutionHistoryTest, RecordsEveryOutputStateInRowMajorOrder)
{
    mhs::solver::Study study {mhs::model::StudyType::Transient, 0.5, 0.25};
    mhs::solver::SolveOptions options;
    options.linear_solver = mhs::solver::SolveOptions::LinearSolverType::Amg;
    options.time_tolerance = 1e-7;
    options.max_time_step = 0.1;

    mhs::solver::Operators operators;
    mhs::solver::SystemAssembler assemble = [&](std::span<const double>, double) -> const mhs::solver::Operators& {
        operators.K = mhs::core::build_csr(1, 1, {{0, 0, 1.0}});
        operators.C = mhs::core::build_csr(1, 1, {{0, 0, 1.0}});
        operators.f = mhs::core::Vector<double>(1, 1.0);
        return operators;
    };

    const std::array initial {0.0};
    const auto solution = mhs::solver::solve_system(study, assemble, initial, options);

    ASSERT_TRUE(solution.converged);
    ASSERT_EQ(solution.snapshot_times, (mhs::core::Vector<double> {0.0, 0.25, 0.5}));
    ASSERT_EQ(solution.snapshot_states.size(), 3u);
    EXPECT_NEAR(solution.snapshot_states[0], 0.0, 1e-14);
    EXPECT_NEAR(solution.snapshot_states[1], 1.0 - std::exp(-0.25), 2e-6);
    EXPECT_NEAR(solution.snapshot_states[2], 1.0 - std::exp(-0.5), 2e-6);
    EXPECT_NEAR(solution.state[0], solution.snapshot_states.back(), 1e-14);
}

TEST(SolutionHistoryTest, SteadySolveProducesOneSnapshot)
{
    mhs::solver::Study study {mhs::model::StudyType::Steady, 0.0, 1.0};
    mhs::solver::SolveOptions options;
    options.linear_solver = mhs::solver::SolveOptions::LinearSolverType::Amg;

    mhs::solver::Operators operators;
    mhs::solver::SystemAssembler assemble = [&](std::span<const double>, double) -> const mhs::solver::Operators& {
        operators.K = mhs::core::build_csr(1, 1, {{0, 0, 2.0}});
        operators.C = mhs::core::CSRMatrix<double>(1, 1);
        operators.f = mhs::core::Vector<double>(1, 6.0);
        return operators;
    };

    const std::array initial {0.0};
    const auto solution = mhs::solver::solve_system(study, assemble, initial, options);

    ASSERT_TRUE(solution.converged);
    ASSERT_EQ(solution.snapshot_times, (mhs::core::Vector<double> {0.0}));
    ASSERT_EQ(solution.snapshot_states.size(), 1u);
    EXPECT_NEAR(solution.snapshot_states[0], 3.0, 1e-14);
}
