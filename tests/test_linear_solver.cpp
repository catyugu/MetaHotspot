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

    constexpr auto kBiCGSTABSpec = mhs::sim::SolverSpec {mhs::sim::SolverType::EigenBiCGSTAB, {1e-10, 2000}};

    // Right-hand side and a computed solver over the SPD test system.
    struct SolverFixture {
        Eigen::VectorXd b;
        mhs::sim::SolverHandle solver;
    };

    SolverFixture make_fixture(const mhs::sim::SolverSpec& spec = {})
    {
        const auto A = make_spd_tridiagonal(kSize);
        SolverFixture fixture;
        fixture.b = A * kExact;
        fixture.solver = mhs::sim::create_solver(spec);
        mhs::sim::solver_compute(fixture.solver, A);
        return fixture;
    }

} // namespace

// The iterative backend accepts an initial guess and warm-starts from it.
TEST(LinearSolver, IterativeWarmStartConvergesFromGuess)
{
    auto cold_fixture = make_fixture(kBiCGSTABSpec);

    // Cold start (zero guess).
    const Eigen::VectorXd x_cold
        = mhs::sim::solver_solve(cold_fixture.solver, cold_fixture.b, Eigen::VectorXd::Zero(kSize));
    const int cold_iters = mhs::sim::solver_iterations(cold_fixture.solver);
    ASSERT_TRUE(mhs::sim::solver_success(cold_fixture.solver));

    // Warm start (exact guess): should converge in very few iterations.
    auto warm_fixture = make_fixture(kBiCGSTABSpec);
    const Eigen::VectorXd x_warm = mhs::sim::solver_solve(warm_fixture.solver, warm_fixture.b, kExact);
    const int warm_iters = mhs::sim::solver_iterations(warm_fixture.solver);
    ASSERT_TRUE(mhs::sim::solver_success(warm_fixture.solver));

    EXPECT_NEAR((x_cold - kExact).norm(), 0.0, 1e-7);
    EXPECT_NEAR((x_warm - kExact).norm(), 0.0, 1e-7);
    // Warm start must not cost more Krylov iterations than a cold start.
    EXPECT_LE(warm_iters, cold_iters);
    EXPECT_LE(warm_iters, 3);
}

// The iterative interface requires a matching-size initial guess.
TEST(LinearSolver, IterativeRejectsMismatchedInitialGuess)
{
    auto fixture = make_fixture(kBiCGSTABSpec);

    Eigen::VectorXd wrong_size(3);
    wrong_size.setZero();
    EXPECT_THROW(mhs::sim::solver_solve(fixture.solver, fixture.b, wrong_size), std::invalid_argument);
}

// The direct backend ignores the initial guess entirely.
TEST(LinearSolver, DirectIgnoresInitialGuess)
{
    auto fixture = make_fixture(mhs::sim::SolverSpec {mhs::sim::SolverType::EigenSparseLU, {}});

    const Eigen::VectorXd with_guess = mhs::sim::solver_solve(fixture.solver, fixture.b, kExact);
    ASSERT_TRUE(mhs::sim::solver_success(fixture.solver));
    const Eigen::VectorXd no_guess = mhs::sim::solver_solve(fixture.solver, fixture.b);
    ASSERT_TRUE(mhs::sim::solver_success(fixture.solver));

    EXPECT_NEAR((with_guess - kExact).norm(), 0.0, 1e-8);
    EXPECT_NEAR((no_guess - kExact).norm(), 0.0, 1e-8);
}

// The default factory returns a working direct solver (Pardiso, or SparseLU
// fallback when MKL is disabled).
TEST(LinearSolver, DefaultFactoryYieldsWorkingDirectSolver)
{
    auto fixture = make_fixture(); // default spec

    const Eigen::VectorXd x = mhs::sim::solver_solve(fixture.solver, fixture.b);

    ASSERT_TRUE(mhs::sim::solver_success(fixture.solver));
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
