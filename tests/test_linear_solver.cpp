#include "numerics/linear/linear_solver.hpp"
#include "solver/solve.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

namespace {

    // 1D Laplacian stencil: tridiagonal, symmetric positive definite.
    Eigen::SparseMatrix<double> make_spd_tridiagonal(int n)
    {
        Eigen::SparseMatrix<double> A(n, n);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<std::size_t>(3 * n));
        for (int i = 0; i < n; ++i) {
            triplets.emplace_back(i, i, 2.0);
            if (i > 0)
                triplets.emplace_back(i, i - 1, -1.0);
            if (i < n - 1)
                triplets.emplace_back(i, i + 1, -1.0);
        }
        A.setFromTriplets(triplets.begin(), triplets.end());
        return A;
    }

    constexpr int kSize = 16;
    const Eigen::VectorXd kExact = Eigen::VectorXd::LinSpaced(kSize, 1.0, 4.0);

} // namespace

// The iterative backend accepts an initial guess and warm-starts from it.
TEST(LinearSolver, IterativeWarmStartConvergesFromGuess)
{
    const auto A = make_spd_tridiagonal(kSize);
    const Eigen::VectorXd b = A * kExact;

    auto spec = mhs::sim::SolverSpec {mhs::sim::SolverType::EigenBiCGSTAB, {1e-10, 2000}};

    // Cold start (zero guess).
    auto cold = mhs::sim::create_solver(spec);
    mhs::sim::solver_compute(cold, A);
    const Eigen::VectorXd x_cold = mhs::sim::solver_solve(cold, b, Eigen::VectorXd::Zero(kSize));
    const int cold_iters = mhs::sim::solver_iterations(cold);
    ASSERT_TRUE(mhs::sim::solver_success(cold));

    // Warm start (exact guess): should converge in very few iterations.
    auto warm = mhs::sim::create_solver(spec);
    mhs::sim::solver_compute(warm, A);
    const Eigen::VectorXd x_warm = mhs::sim::solver_solve(warm, b, kExact);
    const int warm_iters = mhs::sim::solver_iterations(warm);
    ASSERT_TRUE(mhs::sim::solver_success(warm));

    EXPECT_NEAR((x_cold - kExact).norm(), 0.0, 1e-7);
    EXPECT_NEAR((x_warm - kExact).norm(), 0.0, 1e-7);
    // Warm start must not cost more Krylov iterations than a cold start.
    EXPECT_LE(warm_iters, cold_iters);
    EXPECT_LE(warm_iters, 3);
}

// The iterative interface requires a matching-size initial guess.
TEST(LinearSolver, IterativeRejectsMismatchedInitialGuess)
{
    const auto A = make_spd_tridiagonal(kSize);
    const Eigen::VectorXd b = A * kExact;

    auto solver = mhs::sim::create_solver(mhs::sim::SolverSpec {mhs::sim::SolverType::EigenBiCGSTAB, {1e-10, 2000}});
    mhs::sim::solver_compute(solver, A);

    Eigen::VectorXd wrong_size(3);
    wrong_size.setZero();
    EXPECT_THROW(mhs::sim::solver_solve(solver, b, wrong_size), std::invalid_argument);
}

// The direct backend ignores the initial guess entirely.
TEST(LinearSolver, DirectIgnoresInitialGuess)
{
    const auto A = make_spd_tridiagonal(kSize);
    const Eigen::VectorXd b = A * kExact;

    auto solver = mhs::sim::create_solver(mhs::sim::SolverSpec {mhs::sim::SolverType::EigenSparseLU, {}});
    mhs::sim::solver_compute(solver, A);

    const Eigen::VectorXd with_guess = mhs::sim::solver_solve(solver, b, kExact);
    ASSERT_TRUE(mhs::sim::solver_success(solver));
    const Eigen::VectorXd no_guess = mhs::sim::solver_solve(solver, b);
    ASSERT_TRUE(mhs::sim::solver_success(solver));

    EXPECT_NEAR((with_guess - kExact).norm(), 0.0, 1e-8);
    EXPECT_NEAR((no_guess - kExact).norm(), 0.0, 1e-8);
}

// The default factory returns a working direct solver (Pardiso, or SparseLU
// fallback when MKL is disabled).
TEST(LinearSolver, DefaultFactoryYieldsWorkingDirectSolver)
{
    const auto A = make_spd_tridiagonal(kSize);
    const Eigen::VectorXd b = A * kExact;

    auto solver = mhs::sim::create_solver();
    mhs::sim::solver_compute(solver, A);
    const Eigen::VectorXd x = mhs::sim::solver_solve(solver, b);

    ASSERT_TRUE(mhs::sim::solver_success(solver));
    EXPECT_NEAR((x - kExact).norm(), 0.0, 1e-8);
}

// End-to-end: the nonlinear solver drives an iterative backend through the
// dispatch helpers and seeds each linear solve with the previous iterate.
TEST(LinearSolver, NonlinearSolveWarmStartsIterativeBackend)
{
    mhs::sim::Study study {mhs::core::StudyType::Steady, 0.0, 1.0};
    mhs::sim::SolveOptions options;
    options.linear_solver = mhs::sim::SolveOptions::LinearSolverType::EigenBiCGSTAB;

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
    ASSERT_EQ(solution.snapshot_states.size(), 1u);
    EXPECT_NEAR(solution.snapshot_states[0], 3.0, 1e-10);
}
