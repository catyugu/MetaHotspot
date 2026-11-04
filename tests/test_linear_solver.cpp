#include "core/csr_ops.hpp"
#include "core/vector_ops.hpp"
#include "linear/solver.hpp"
#include "solver/solve.hpp"
#include <cmath>

#include <gtest/gtest.h>

#include "core/vector.hpp"
#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
    double error_norm(std::span<const double> x, std::span<const double> y)
    {
        double norm = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            norm = std::hypot(norm, x[i] - y[i]);
        return norm;
    }

    // 1D Laplacian stencil: tridiagonal, symmetric positive definite.
    mhs::core::CSRMatrix<double> make_spd_tridiagonal(int n)
    {
        mhs::core::CSRMatrix<double> A(n, n);
        mhs::core::Vector<mhs::core::Triplet<double>> triplets;
        triplets.reserve(static_cast<std::size_t>(3 * n));
        for (int i = 0; i < n; ++i) {
            triplets.emplace_back(i, i, 2.0);
            if (i > 0)
                triplets.emplace_back(i, i - 1, -1.0);
            if (i < n - 1)
                triplets.emplace_back(i, i + 1, -1.0);
        }
        A = mhs::core::build_csr(A.rows(), A.cols(), std::move(triplets));
        return A;
    }

    constexpr int kSize = 16;
    const auto kExact = [] {
        mhs::core::Vector<double> x(kSize);
        for (int i = 0; i < kSize; ++i)
            x[i] = 1.0 + 3.0 * i / (kSize - 1);
        return x;
    }();

    // The default thermal linear solver is AMG-preconditioned CG (Amg).
    constexpr auto kAmgSpec = mhs::linear::SolverSpec {mhs::linear::SolverType::Amg, {1e-10, 2000}};

    // Right-hand side and a computed solver over the SPD test system.
    struct SolverFixture {
        mhs::core::Vector<double> b;
        mhs::linear::SolverPtr solver;
    };

    SolverFixture make_fixture(const mhs::linear::SolverSpec& spec = {})
    {
        const auto A = make_spd_tridiagonal(kSize);
        SolverFixture fixture;
        fixture.b.resize(kSize);
        mhs::core::spmv(A.view(), kExact, fixture.b);
        fixture.solver = mhs::linear::create_solver(spec);
        fixture.solver->compute(A.view());
        return fixture;
    }

} // namespace

// The iterative backend accepts an initial guess and warm-starts from it.
TEST(Solver, IterativeWarmStartConvergesFromGuess)
{
    auto cold_fixture = make_fixture(kAmgSpec);

    // Cold start (zero guess).
    mhs::core::Vector<double> x_cold(kSize);
    const auto x_cold_result = cold_fixture.solver->solve(cold_fixture.b, x_cold);
    const int cold_iters = x_cold_result.iterations;
    ASSERT_TRUE(x_cold_result.success);

    // Warm start (exact guess): should converge in very few iterations.
    auto warm_fixture = make_fixture(kAmgSpec);
    mhs::core::Vector<double> x_warm = kExact;
    const auto x_warm_result = warm_fixture.solver->solve(warm_fixture.b, x_warm);
    const int warm_iters = x_warm_result.iterations;
    ASSERT_TRUE(x_warm_result.success);

    EXPECT_NEAR(error_norm(x_cold, kExact), 0.0, 1e-7);
    EXPECT_NEAR(error_norm(x_warm, kExact), 0.0, 1e-7);
    // Warm start must not cost more Krylov iterations than a cold start.
    EXPECT_LE(warm_iters, cold_iters);
    EXPECT_LE(warm_iters, 5);
}

// The iterative interface requires a matching-size initial guess.
TEST(Solver, IterativeRejectsMismatchedInitialGuess)
{
    auto fixture = make_fixture(kAmgSpec);

    mhs::core::Vector<double> wrong_size(3);

    EXPECT_THROW(fixture.solver->solve(fixture.b, wrong_size), std::invalid_argument);
}

// The default factory returns a working self-tuning AMG solver (Amg),
// which is iterative (no MKL needed) and warm-starts from a zero guess.
TEST(Solver, DefaultFactoryYieldsWorkingIterativeSolver)
{
    auto fixture = make_fixture(); // default spec = Amg

    mhs::core::Vector<double> x(kSize);
    const auto x_result = fixture.solver->solve(fixture.b, x);

    ASSERT_TRUE(x_result.success);
    EXPECT_NEAR(error_norm(x, kExact), 0.0, 1e-8);
}

// The direct backend ignores the initial guess entirely.
#ifdef MHS_ENABLE_PARDISO
TEST(Solver, DirectIgnoresInitialGuess)
{
    auto fixture = make_fixture(mhs::linear::SolverSpec {mhs::linear::SolverType::Pardiso, {}});

    mhs::core::Vector<double> with_guess = kExact;
    const auto with_guess_result = fixture.solver->solve(fixture.b, with_guess);
    ASSERT_TRUE(with_guess_result.success);
    mhs::core::Vector<double> no_guess(kSize);
    const auto no_guess_result = fixture.solver->solve(fixture.b, no_guess);
    ASSERT_TRUE(no_guess_result.success);

    EXPECT_NEAR(error_norm(with_guess, kExact), 0.0, 1e-8);
    EXPECT_NEAR(error_norm(no_guess, kExact), 0.0, 1e-8);
}
#endif // MHS_ENABLE_PARDISO

// End-to-end: the nonlinear solver drives an iterative backend through the
// direct backend calls and seeds each linear solve with the previous iterate.
TEST(Solver, NonlinearSolveWarmStartsIterativeBackend)
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
    ASSERT_EQ(solution.snapshot_states.size(), 1u);
    EXPECT_NEAR(solution.snapshot_states[0], 3.0, 1e-10);
}

TEST(CoreIndex, SparseCapacityAndTripletBounds)
{
    using namespace mhs::core;
    static_assert(sizeof(SparseIndex) == 4);
    static_assert(std::is_aggregate_v<Triplet<double>>);
    const auto sparse_max = static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max());
    EXPECT_NO_THROW(detail::checked_sparse_size(sparse_max));
    EXPECT_THROW(build_csr(sparse_max + 1, 1, {}), std::length_error);
    EXPECT_THROW(build_csr(1, 1, {{-1, 0, 1.0}}), std::out_of_range);
    EXPECT_THROW(build_csr(1, 1, {{0, 1, 1.0}}), std::out_of_range);
}
