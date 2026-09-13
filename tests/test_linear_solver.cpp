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

    // The default thermal linear solver is AMG-preconditioned CG (AmgCg).
    constexpr auto kAmgSpec = mhs::sim::SolverSpec {mhs::sim::SolverType::AmgCg, {1e-10, 2000}};

    // Right-hand side and a computed solver over the SPD test system.
    struct SolverFixture {
        Eigen::VectorXd b;
        mhs::sim::SolverPtr solver;
    };

    SolverFixture make_fixture(const mhs::sim::SolverSpec& spec = {})
    {
        const auto A = make_spd_tridiagonal(kSize);
        SolverFixture fixture;
        fixture.b = A * kExact;
        fixture.solver = mhs::sim::create_solver(spec);
        fixture.solver->compute(A);
        return fixture;
    }

} // namespace

// The iterative backend accepts an initial guess and warm-starts from it.
TEST(LinearSolver, IterativeWarmStartConvergesFromGuess)
{
    auto cold_fixture = make_fixture(kAmgSpec);

    // Cold start (zero guess).
    const Eigen::VectorXd x_cold = cold_fixture.solver->solve(cold_fixture.b, Eigen::VectorXd::Zero(kSize));
    const int cold_iters = cold_fixture.solver->iterations();
    ASSERT_TRUE(cold_fixture.solver->success());

    // Warm start (exact guess): should converge in very few iterations.
    auto warm_fixture = make_fixture(kAmgSpec);
    const Eigen::VectorXd x_warm = warm_fixture.solver->solve(warm_fixture.b, kExact);
    const int warm_iters = warm_fixture.solver->iterations();
    ASSERT_TRUE(warm_fixture.solver->success());

    EXPECT_NEAR((x_cold - kExact).norm(), 0.0, 1e-7);
    EXPECT_NEAR((x_warm - kExact).norm(), 0.0, 1e-7);
    // Warm start must not cost more Krylov iterations than a cold start.
    EXPECT_LE(warm_iters, cold_iters);
    EXPECT_LE(warm_iters, 5);
}

// The iterative interface requires a matching-size initial guess.
TEST(LinearSolver, IterativeRejectsMismatchedInitialGuess)
{
    auto fixture = make_fixture(kAmgSpec);

    Eigen::VectorXd wrong_size(3);
    wrong_size.setZero();
    EXPECT_THROW(fixture.solver->solve(fixture.b, wrong_size), std::invalid_argument);
}

// The default factory returns a working self-tuning AMG solver (AmgCg),
// which is iterative (no MKL needed) and warm-starts from a zero guess.
TEST(LinearSolver, DefaultFactoryYieldsWorkingIterativeSolver)
{
    auto fixture = make_fixture(); // default spec = AmgCg

    const Eigen::VectorXd x = fixture.solver->solve(fixture.b, Eigen::VectorXd::Zero(kSize));

    ASSERT_TRUE(fixture.solver->success());
    EXPECT_NEAR((x - kExact).norm(), 0.0, 1e-8);
}

// The direct backend ignores the initial guess entirely.
#ifdef MHS_ENABLE_PARDISO
TEST(LinearSolver, DirectIgnoresInitialGuess)
{
    auto fixture = make_fixture(mhs::sim::SolverSpec {mhs::sim::SolverType::Pardiso, {}});

    const Eigen::VectorXd with_guess = fixture.solver->solve(fixture.b, kExact);
    ASSERT_TRUE(fixture.solver->success());
    const Eigen::VectorXd no_guess = fixture.solver->solve(fixture.b, Eigen::VectorXd::Zero(kSize));
    ASSERT_TRUE(fixture.solver->success());

    EXPECT_NEAR((with_guess - kExact).norm(), 0.0, 1e-8);
    EXPECT_NEAR((no_guess - kExact).norm(), 0.0, 1e-8);
}
#endif // MHS_ENABLE_PARDISO

// End-to-end: the nonlinear solver drives an iterative backend through the
// direct backend calls and seeds each linear solve with the previous iterate.
TEST(LinearSolver, NonlinearSolveWarmStartsIterativeBackend)
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
    ASSERT_EQ(solution.snapshot_states.size(), 1u);
    EXPECT_NEAR(solution.snapshot_states[0], 3.0, 1e-10);
}
