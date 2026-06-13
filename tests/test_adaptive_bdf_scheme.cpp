#include "time_scheme/adaptive_bdf_scheme.hpp"
#include "time_scheme/step_controller.hpp"
#include "time_scheme/time_scheme.hpp"
#include <gtest/gtest.h>

using namespace mhs::sim::time_scheme;

TEST(StepControllerTest, ErrorNormFindsMax)
{
    EXPECT_DOUBLE_EQ(StepController::error_norm({1.0, -2.0, 0.5}), 2.0);
    EXPECT_DOUBLE_EQ(StepController::error_norm({}), 0.0);
    EXPECT_DOUBLE_EQ(StepController::error_norm({-0.001, 0.001}), 0.001);
}

TEST(StepControllerTest, AcceptsWhenErrorBelowTolerance)
{
    TimeSchemeConfig cfg;
    cfg.abs_tol = 1e-3;
    StepController c(cfg);
    auto r = c.decide(/*dt=*/0.1, /*order=*/2, /*err=*/1e-4);
    EXPECT_TRUE(r.accepted);
    EXPECT_GT(r.next_dt, 0.1); // grew
}

TEST(StepControllerTest, RejectsWhenErrorAboveTolerance)
{
    TimeSchemeConfig cfg;
    cfg.abs_tol = 1e-6;
    StepController c(cfg);
    auto r = c.decide(/*dt=*/0.1, /*order=*/2, /*err=*/1e-2);
    EXPECT_FALSE(r.accepted);
    EXPECT_LT(r.next_dt, 0.1); // shrunk
}

TEST(StepControllerTest, ClampsToMinDt)
{
    TimeSchemeConfig cfg;
    cfg.abs_tol = 1e-6;
    cfg.min_dt  = 1e-9;
    cfg.max_dt  = 1.0;
    StepController c(cfg);
    auto r = c.decide(/*dt=*/1e-8, /*order=*/1, /*err=*/0.0);
    EXPECT_GE(r.next_dt, 1e-9);
}

TEST(StepControllerTest, ClampsToMaxDt)
{
    TimeSchemeConfig cfg;
    cfg.abs_tol = 1e-6;
    cfg.min_dt  = 1e-9;
    cfg.max_dt  = 1.0;
    StepController c(cfg);
    auto r = c.decide(/*dt=*/1.0, /*order=*/1, /*err=*/0.0);
    EXPECT_LE(r.next_dt, 1.0);
}

TEST(StepControllerTest, GrowsOnSmallError)
{
    TimeSchemeConfig cfg;
    cfg.abs_tol = 1e-3;
    cfg.safety  = 0.9;
    StepController c(cfg);
    auto r = c.decide(/*dt=*/0.01, /*order=*/1, /*err=*/1e-7);
    EXPECT_TRUE(r.accepted);
    EXPECT_GT(r.next_dt, 0.01);
}

TEST(StepControllerTest, ShrinksOnLargeError)
{
    TimeSchemeConfig cfg;
    cfg.abs_tol = 1e-3;
    cfg.safety  = 0.9;
    StepController c(cfg);
    auto r = c.decide(/*dt=*/0.1, /*order=*/1, /*err=*/1.0);
    EXPECT_FALSE(r.accepted);
    EXPECT_LT(r.next_dt, 0.1);
}

TEST(AdaptiveBdfSchemeTest, AcceptsSmallError)
{
    TimeSchemeConfig cfg;
    cfg.kind = TimeSchemeKind::AdaptiveBdf;
    cfg.abs_tol = 1e-3;
    AdaptiveBdfScheme s(cfg);
    mhs::core::TimeStepBuffer hist(1, 4);
    hist.reset({300.0});
    // Error below tolerance → Accept
    EXPECT_EQ(s.accept_or_reject(hist, {1.0, 2.0}, {1e-5, 1e-6}),
              AcceptDecision::Accept);
}

TEST(AdaptiveBdfSchemeTest, RejectsLargeError)
{
    TimeSchemeConfig cfg;
    cfg.kind = TimeSchemeKind::AdaptiveBdf;
    cfg.abs_tol = 1e-9;
    AdaptiveBdfScheme s(cfg);
    mhs::core::TimeStepBuffer hist(1, 4);
    hist.reset({300.0});
    // Error above tolerance → Reject
    EXPECT_EQ(s.accept_or_reject(hist, {1.0, 2.0}, {1.0, 1.0}),
              AcceptDecision::Reject);
}
