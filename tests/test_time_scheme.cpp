#include "engine/linear_system.hpp"
#include "engine/solution_history.hpp"
#include "engine/time_integration.hpp"

#include <Eigen/Sparse>
#include <gtest/gtest.h>

// ============================================================================
// BDF1 (Backward Euler) linear-system construction
// ============================================================================

namespace {

    /// Helper: build a simple 3-DOF AssemblyResult with known values.
    /// K = diag(2, 4, 6), f = (10, 20, 30), M_diag = (1, 2, 3)
    mhs::sim::AssemblyResult make_known_3dof_ops()
    {
        const int N = 3;
        Eigen::SparseMatrix<double> K(N, N);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.emplace_back(0, 0, 2.0);
        triplets.emplace_back(1, 1, 4.0);
        triplets.emplace_back(2, 2, 6.0);
        K.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXd f(N);
        f << 10.0, 20.0, 30.0;

        Eigen::VectorXd M_diag(N);
        M_diag << 1.0, 2.0, 3.0;

        return {std::move(K), std::move(f), std::move(M_diag)};
    }

    TEST(TimeSchemeBdf1, Known3Dof)
    {
        auto ops = make_known_3dof_ops();
        const double dt = 0.5;

        // SolutionHistory with a single snapshot (t=0).
        mhs::core::SolutionHistory hist(3, 2);
        std::vector<double> T0 = {100.0, 200.0, 300.0};
        hist.initialize(T0, 0.0);

        auto ls = mhs::sim::time_scheme::build_system(mhs::sim::time_scheme::IntegratorKind::Bdf1, ops, hist, dt);

        ASSERT_EQ(ls.A.rows(), 3);
        ASSERT_EQ(ls.A.cols(), 3);
        ASSERT_EQ(ls.b.size(), 3);

        // A = K + M_diag / dt
        // A[0,0] = 2 + 1/0.5 = 4
        // A[1,1] = 4 + 2/0.5 = 8
        // A[2,2] = 6 + 3/0.5 = 12
        EXPECT_DOUBLE_EQ(ls.A.coeff(0, 0), 4.0);
        EXPECT_DOUBLE_EQ(ls.A.coeff(1, 1), 8.0);
        EXPECT_DOUBLE_EQ(ls.A.coeff(2, 2), 12.0);

        // b = f + M_diag * T_prev / dt
        // b[0] = 10 + 1*100/0.5 = 210
        // b[1] = 20 + 2*200/0.5 = 820
        // b[2] = 30 + 3*300/0.5 = 1830
        EXPECT_DOUBLE_EQ(ls.b(0), 210.0);
        EXPECT_DOUBLE_EQ(ls.b(1), 820.0);
        EXPECT_DOUBLE_EQ(ls.b(2), 1830.0);
    }

    TEST(TimeSchemeBdf1, ZeroMassMatrix)
    {
        // Steady-like: M_diag = 0 => A = K, b = f
        const int N = 2;
        Eigen::SparseMatrix<double> K(N, N);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.emplace_back(0, 0, 5.0);
        triplets.emplace_back(1, 1, 10.0);
        K.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXd f(N);
        f << 1.0, 2.0;

        Eigen::VectorXd M_diag = Eigen::VectorXd::Zero(N);

        mhs::sim::AssemblyResult ops {std::move(K), std::move(f), std::move(M_diag)};

        mhs::core::SolutionHistory hist(2, 2);
        hist.initialize({50.0, 60.0}, 0.0);

        auto ls = mhs::sim::time_scheme::build_system(mhs::sim::time_scheme::IntegratorKind::Bdf1, ops, hist, 0.5);

        EXPECT_DOUBLE_EQ(ls.A.coeff(0, 0), 5.0);
        EXPECT_DOUBLE_EQ(ls.A.coeff(1, 1), 10.0);
        EXPECT_DOUBLE_EQ(ls.b(0), 1.0);
        EXPECT_DOUBLE_EQ(ls.b(1), 2.0);
    }

} // namespace

// ============================================================================
// BDF2 linear-system construction
// ============================================================================

namespace {

    TEST(TimeSchemeBdf2, Known3Dof)
    {
        auto ops = make_known_3dof_ops();
        const double dt = 0.5;

        // Initialize with 2 snapshots: T at t=0 and t=0.4
        mhs::core::SolutionHistory hist(3, 2);
        std::vector<double> T0 = {100.0, 200.0, 300.0}; // t=0
        std::vector<double> T1 = {110.0, 205.0, 305.0}; // t=0.4
        hist.initialize(T0, 0.0);
        hist.accept(T1, 0.4);

        auto ls = mhs::sim::time_scheme::build_system(mhs::sim::time_scheme::IntegratorKind::Bdf2, ops, hist, dt);

        ASSERT_EQ(ls.A.rows(), 3);
        ASSERT_EQ(ls.b.size(), 3);

        // h_n = 0.5, h_nm1 = 0.4, delta = 0.5/0.4 = 1.25
        // alpha0 = (1 + 2*1.25) / (0.5 * (1 + 1.25)) = (3.5) / (0.5 * 2.25) = 3.5 / 1.125 = 3.111...
        // A[i,i] = K[i,i] + alpha0 * M_diag[i]
        const double delta_val = 0.5 / 0.4;
        const double alpha0 = (1.0 + 2.0 * delta_val) / (0.5 * (1.0 + delta_val));
        EXPECT_DOUBLE_EQ(ls.A.coeff(0, 0), 2.0 + alpha0 * 1.0);
        EXPECT_DOUBLE_EQ(ls.A.coeff(1, 1), 4.0 + alpha0 * 2.0);
        EXPECT_DOUBLE_EQ(ls.A.coeff(2, 2), 6.0 + alpha0 * 3.0);

        // alpha1 = -(1+delta) / h_n = -(1+1.25)/0.5 = -2.25/0.5 = -4.5
        // alpha2 = delta^2 / (h_n*(1+delta)) = 1.5625/(0.5*2.25) = 1.5625/1.125 = 1.38888...
        const double alpha1 = -(1.0 + delta_val) / 0.5;
        const double alpha2 = (delta_val * delta_val) / (0.5 * (1.0 + delta_val));

        // b = f - M_diag .* (alpha1*T_n + alpha2*T_nm1)
        // T_n = T1 (latest), T_nm1 = T0
        Eigen::Vector3d M_vec(1.0, 2.0, 3.0);
        Eigen::Vector3d T_n(110.0, 205.0, 305.0);
        Eigen::Vector3d T_nm1(100.0, 200.0, 300.0);
        Eigen::Vector3d expected_b;
        expected_b << 10.0, 20.0, 30.0;
        expected_b -= M_vec.cwiseProduct(alpha1 * T_n + alpha2 * T_nm1);

        EXPECT_DOUBLE_EQ(ls.b(0), expected_b(0));
        EXPECT_DOUBLE_EQ(ls.b(1), expected_b(1));
        EXPECT_DOUBLE_EQ(ls.b(2), expected_b(2));
    }

} // namespace

// ============================================================================
// build_system dispatch (public API)
// ============================================================================

namespace {

    TEST(TimeSchemeBuildSystem, Bdf2FallsBackToBdf1WithInsufficientHistory)
    {
        auto ops = make_known_3dof_ops();
        // Only 1 snapshot — not enough for BDF2.
        mhs::core::SolutionHistory hist(3, 2);
        hist.initialize({100.0, 200.0, 300.0}, 0.0);

        auto ls = mhs::sim::time_scheme::build_system(mhs::sim::time_scheme::IntegratorKind::Bdf2, ops, hist, 1.0);

        // Should be BDF1: A[0,0] = 2 + 1/1 = 3
        EXPECT_DOUBLE_EQ(ls.A.coeff(0, 0), 3.0);
        EXPECT_DOUBLE_EQ(ls.b(0), 110.0);
    }

} // namespace

// ============================================================================
// Error estimation (LTE + PI step-size suggestion)
// ============================================================================

namespace {

    mhs::core::SolutionHistory make_hist(
        const std::vector<double>& T_init, const std::vector<double>& T_prev, double dt_prev, double t0 = 0.0)
    {
        mhs::core::SolutionHistory hist(static_cast<int>(T_init.size()), 2);
        hist.initialize(T_init, t0);
        hist.accept(T_prev, t0 + dt_prev);
        return hist;
    }

    TEST(TimeSchemeErrorEstimate, ExactSteadyStateReturnsZero)
    {
        // If trial_T equals the previous solution, LTE should be zero.
        std::vector<double> T = {300.0, 300.0, 300.0};
        auto hist = make_hist(T, T, 0.5);

        mhs::sim::time_scheme::ErrorControlConfig cfg;
        cfg.abs_tol = 1e-4;
        cfg.safety = 0.9;

        auto est = mhs::sim::time_scheme::estimate_error(hist, T, 0.5, cfg);

        // error_ratio should be near 0 (steady state).
        EXPECT_LT(est.error_ratio, 1e-10);
        // suggested_factor should be clamped to max (2.0) since err ≈ 0.
        EXPECT_DOUBLE_EQ(est.suggested_factor, 2.0);
    }

    TEST(TimeSchemeErrorEstimate, LargeErrorReducesStep)
    {
        // Trial T differs drastically from history → large LTE → suggested_factor < 1.
        std::vector<double> T0 = {300.0, 300.0};
        std::vector<double> T1 = {300.0, 300.0};
        auto hist = make_hist(T0, T1, 0.5);

        std::vector<double> trial_T = {1000.0, 2000.0};
        mhs::sim::time_scheme::ErrorControlConfig cfg;
        cfg.abs_tol = 1e-4;
        cfg.safety = 0.9;

        auto est = mhs::sim::time_scheme::estimate_error(hist, trial_T, 0.5, cfg);

        EXPECT_GT(est.error_ratio, 1.0); // Should exceed tolerance
        EXPECT_LT(est.suggested_factor, 1.0); // Should suggest smaller step
    }

    TEST(TimeSchemeErrorEstimate, Bdf2ErrorWithPrevious)
    {
        // With 2 prior snapshots, the BDF2 error estimate uses second differences.
        std::vector<double> T0 = {300.0};
        std::vector<double> T1 = {310.0};
        auto hist = make_hist(T0, T1, 0.5);

        std::vector<double> trial_T = {315.0};
        mhs::sim::time_scheme::ErrorControlConfig cfg;
        cfg.abs_tol = 1e-4;
        cfg.safety = 0.9;

        // Before recording trial_T into hist, size() == 2 → 2 prior snapshots
        // Actually hist has T0 and T1 = size 2
        auto est = mhs::sim::time_scheme::estimate_error(hist, trial_T, 0.5, cfg);

        // Manually: T_curr=315, T_prev=310, T_prev2=300
        // ratio = 0.5/0.5 = 1.0
        // err_vec = |(315-310) - 1.0*(310-300)| = |5 - 10| = 5
        // Normalised by max(|315|, 1) = 315 → err = 5/315 ≈ 0.01587
        // error_ratio = 0.01587 / 1e-4 ≈ 158.7
        // fac = 0.9 * (1e-4 / max(0.01587, zero_guard))^0.5 = 0.9 * 0.0063^0.5 = 0.9*0.0794 = 0.071
        // Clamped to [0.5, 2.0] → 0.5
        EXPECT_GT(est.error_ratio, 1.0);
        EXPECT_NEAR(est.suggested_factor, 0.5, 0.01);
    }

    TEST(TimeSchemeErrorEstimate, SingleSnapshotUsesFirstDifference)
    {
        // With exactly 1 snapshot (startup), the error estimate uses |T_curr - T_prev| only.
        std::vector<double> T0 = {300.0};
        mhs::core::SolutionHistory hist(1, 2);
        hist.initialize(T0, 0.0);
        // size() == 1 → single snapshot path in estimate_error

        std::vector<double> trial_T = {310.0};
        mhs::sim::time_scheme::ErrorControlConfig cfg;
        cfg.abs_tol = 1e-4;
        cfg.safety = 0.9;

        auto est = mhs::sim::time_scheme::estimate_error(hist, trial_T, 0.5, cfg);

        // err_vec = |310-300| = 10, normalised by max(310,1)=310 → 0.03225
        // error_ratio = 0.03225 / 1e-4 = 322.5
        EXPECT_GT(est.error_ratio, 100.0);
        // suggested_factor = 0.9 * (1e-4/0.03225)^0.5 = 0.9*0.0557 = 0.0501 → clamped to 0.5
        EXPECT_NEAR(est.suggested_factor, 0.5, 0.01);
    }

    TEST(TimeSchemeErrorEstimate, EmptySolution)
    {
        mhs::core::SolutionHistory hist(0, 2);
        std::vector<double> trial_T;

        mhs::sim::time_scheme::ErrorControlConfig cfg;
        auto est = mhs::sim::time_scheme::estimate_error(hist, trial_T, 0.5, cfg);

        EXPECT_DOUBLE_EQ(est.error_ratio, 0.0);
        EXPECT_DOUBLE_EQ(est.suggested_factor, 1.0);
    }

} // namespace

// ============================================================================
// StepController (strategy + output-time grid)
// ============================================================================

namespace {

    TEST(StepController, FreeStrategyReturnsSuggestedDt)
    {
        mhs::sim::time_scheme::StepController ctrl(mhs::sim::time_scheme::StepStrategy::Free, 1e-6, 10.0);
        ctrl.rebuild(10.0, 0.5);

        double dt = ctrl.prepare(0.3, 0.0, 10.0);
        EXPECT_DOUBLE_EQ(dt, 0.3); // Free: returns dt_suggested as-is
    }

    TEST(StepController, ManualReturnsFixedDt)
    {
        mhs::sim::time_scheme::StepController ctrl(mhs::sim::time_scheme::StepStrategy::Manual, 1e-6, 10.0, 0.05);
        ctrl.rebuild(10.0, 0.5);

        double dt = ctrl.prepare(0.3, 0.0, 10.0);
        EXPECT_DOUBLE_EQ(dt, 0.05); // fixed_dt overrides suggested
    }

    TEST(StepController, ManualRespectsRemainingDuration)
    {
        mhs::sim::time_scheme::StepController ctrl(mhs::sim::time_scheme::StepStrategy::Manual, 1e-6, 10.0, 0.5);
        ctrl.rebuild(1.0, 0.5);

        // At t=0.9, remaining = 0.1 < fixed_dt=0.5 → should clamp to 0.1
        double dt = ctrl.prepare(0.5, 0.9, 1.0);
        EXPECT_NEAR(dt, 0.1, 1e-12);
    }

    TEST(StepController, FlushOutputsWithGrid)
    {
        mhs::sim::time_scheme::StepController ctrl(mhs::sim::time_scheme::StepStrategy::Free, 1e-6, 10.0);
        ctrl.rebuild(5.0, 1.0);

        auto out = ctrl.flush_outputs(0.0);
        EXPECT_TRUE(out.empty()); // No output at t=0; t=0 is a grid point but is behind last_flushed

        out = ctrl.flush_outputs(0.3);
        EXPECT_TRUE(out.empty()); // No output times crossed

        out = ctrl.flush_outputs(1.5);
        ASSERT_EQ(out.size(), 1u);
        EXPECT_DOUBLE_EQ(out[0], 1.0);

        out = ctrl.flush_outputs(2.5);
        ASSERT_EQ(out.size(), 1u);
        EXPECT_DOUBLE_EQ(out[0], 2.0);

        // Flush remaining up to end.
        out = ctrl.flush_outputs(5.0);
        ASSERT_EQ(out.size(), 3u);
        EXPECT_DOUBLE_EQ(out[0], 3.0);
        EXPECT_DOUBLE_EQ(out[1], 4.0);
        EXPECT_DOUBLE_EQ(out[2], 5.0);
    }

    TEST(StepController, FlushOutputsWithoutGrid)
    {
        mhs::sim::time_scheme::StepController ctrl(mhs::sim::time_scheme::StepStrategy::Free, 1e-6, 10.0);
        ctrl.rebuild(5.0, 0.0); // no output grid

        auto out = ctrl.flush_outputs(0.3);
        ASSERT_EQ(out.size(), 1u);
        EXPECT_DOUBLE_EQ(out[0], 0.3);

        // Same time again → no new output.
        out = ctrl.flush_outputs(0.3);
        EXPECT_TRUE(out.empty());

        out = ctrl.flush_outputs(1.5);
        ASSERT_EQ(out.size(), 1u);
        EXPECT_DOUBLE_EQ(out[0], 1.5);
    }

    TEST(StepController, IntermediatePlantsInternalStep)
    {
        mhs::sim::time_scheme::StepController ctrl(mhs::sim::time_scheme::StepStrategy::Intermediate, 1e-6, 10.0);
        ctrl.rebuild(5.0, 1.0);

        // Flush past t=0 so next_idx points at t=1.0, then snap.
        ctrl.flush_outputs(0.0);
        double dt = ctrl.prepare(1.2, 0.0, 5.0);
        EXPECT_DOUBLE_EQ(dt, 0.5);
    }

    TEST(StepController, RebuildResetsState)
    {
        mhs::sim::time_scheme::StepController ctrl(mhs::sim::time_scheme::StepStrategy::Free, 1e-6, 10.0);
        ctrl.rebuild(5.0, 1.0);
        ctrl.flush_outputs(5.0);

        // Rebuild with new parameters
        ctrl.rebuild(10.0, 2.0);
        auto out = ctrl.flush_outputs(2.0);
        // t=0 is behind last_flushed (=0.0 after rebuild) so not emitted again.
        // t=2.0 is the first output.
        ASSERT_EQ(out.size(), 1u);
        EXPECT_DOUBLE_EQ(out[0], 2.0);
    }

    TEST(StepController, StrictEnsuresExactOutputTimes)
    {
        mhs::sim::time_scheme::StepController ctrl(mhs::sim::time_scheme::StepStrategy::Strict, 1e-6, 5.0);
        ctrl.rebuild(3.0, 1.0);

        // Run through a sequence and verify dt always lands exactly on grid.
        double t = 0.0;
        std::vector<double> output_times;
        while (t < 3.0 - 1e-12) {
            double dt = ctrl.prepare(0.3, t, 3.0);
            t += dt;
            auto out = ctrl.flush_outputs(t);
            output_times.insert(output_times.end(), out.begin(), out.end());
        }

        ASSERT_EQ(output_times.size(), 3u);
        EXPECT_DOUBLE_EQ(output_times[0], 1.0);
        EXPECT_DOUBLE_EQ(output_times[1], 2.0);
        EXPECT_DOUBLE_EQ(output_times[2], 3.0);
    }

    TEST(StepController, DtClampedToBounds)
    {
        mhs::sim::time_scheme::StepController ctrl(mhs::sim::time_scheme::StepStrategy::Free, 0.01, 0.5);
        ctrl.rebuild(10.0, 1.0);

        // Suggested dt above max → clamped to max_dt.
        double dt = ctrl.prepare(100.0, 0.0, 10.0);
        EXPECT_DOUBLE_EQ(dt, 0.5);

        // Suggested dt below min → clamped to min_dt.
        dt = ctrl.prepare(1e-10, 0.5, 10.0);
        EXPECT_DOUBLE_EQ(dt, 0.01);
    }

} // namespace
