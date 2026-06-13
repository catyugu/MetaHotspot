#include "assembler/assembler.hpp"
#include "data/internal_model.hpp"
#include "data/time_step_buffer.hpp"
#include "time_scheme/bdf2_scheme.hpp"
#include <gtest/gtest.h>

using namespace mhs::sim;
using namespace mhs::sim::time_scheme;
using mhs::core::TimeStepBuffer;

namespace {
    StaticOpsResult make_static(double a, double b, double c)
    {
        StaticOpsResult sops;
        const int N = 3;
        std::vector<Eigen::Triplet<double>> trips;
        trips.emplace_back(0, 0, a);
        trips.emplace_back(1, 1, b);
        trips.emplace_back(2, 2, c);
        sops.K.resize(N, N);
        sops.K.setFromTriplets(trips.begin(), trips.end());
        sops.f_static = Eigen::VectorXd::Zero(N);
        return sops;
    }

    MassOpsResult make_mass(double m0, double m1, double m2)
    {
        MassOpsResult mops;
        mops.M_diag = Eigen::VectorXd(3);
        mops.M_diag << m0, m1, m2;
        return mops;
    }
}

TEST(Bdf2SchemeTest, CoefficientsFixedStep)
{
    // Fixed step δ=1: α0 = 3/(2h), α1 = -2/h, α2 = 1/(2h)
    StaticOpsResult sops = make_static(1.0, 2.0, 3.0);
    MassOpsResult mops = make_mass(0.1, 0.2, 0.3);

    TimeStepBuffer hist(3, 4);
    hist.reset({100.0, 200.0, 300.0});          // T_{n-2} at t=0
    hist.push({110.0, 220.0, 330.0}, 0.1);      // T_{n-1} at t=0.1
    hist.push({121.0, 242.0, 363.0}, 0.2);      // T_n   at t=0.2

    Bdf2Scheme s(TimeSchemeConfig{TimeSchemeKind::Bdf2, /*initial_dt=*/0.1});
    auto ls = s.build_system(sops, mops, hist, /*order=*/2, /*dt=*/0.1);

    // Fixed-step δ=1, h=0.1
    const double h = 0.1;
    const double a0 = 3.0 / (2.0 * h);
    const double a1 = -2.0 / h;
    const double a2 = 1.0 / (2.0 * h);
    EXPECT_NEAR(ls.A.coeff(0, 0), 1.0 + a0 * 0.1, 1e-9);
    EXPECT_NEAR(ls.A.coeff(1, 1), 2.0 + a0 * 0.2, 1e-9);
    EXPECT_NEAR(ls.A.coeff(2, 2), 3.0 + a0 * 0.3, 1e-9);

    // b(i) = a0*M(i)*T_n + a1*M(i)*T_{n-1} + a2*M(i)*T_{n-2}
    EXPECT_NEAR(ls.b(0), a0 * 0.1 * 121.0 + a1 * 0.1 * 110.0 + a2 * 0.1 * 100.0, 1e-9);
    EXPECT_NEAR(ls.b(1), a0 * 0.2 * 242.0 + a1 * 0.2 * 220.0 + a2 * 0.2 * 200.0, 1e-9);
    EXPECT_NEAR(ls.b(2), a0 * 0.3 * 363.0 + a1 * 0.3 * 330.0 + a2 * 0.3 * 300.0, 1e-9);
}

TEST(Bdf2SchemeTest, CoefficientsVariableStep)
{
    // BDF2 build_system semantics: history contains the state before the
    // current step is committed.  h_np1 = history.dt_to(1) is the most recent
    // committed step; h_n = the upcoming dt.  For δ=2: h_n=0.10, h_np1=0.05.
    //   reset at t=0, push at t=0.05, push at t=0.10
    //   dt_to(1) = 0.10 - 0.05 = 0.05 ✓
    //   upcoming dt = 0.10 ✓
    StaticOpsResult sops = make_static(1.0, 2.0, 3.0);
    MassOpsResult mops = make_mass(0.1, 0.2, 0.3);

    TimeStepBuffer hist(3, 4);
    hist.reset({100.0, 200.0, 300.0});           // t=0
    hist.push({105.0, 210.0, 315.0}, 0.05);      // t=0.05
    hist.push({110.0, 220.0, 330.0}, 0.10);      // t=0.10

    Bdf2Scheme s(TimeSchemeConfig{TimeSchemeKind::Bdf2, /*initial_dt=*/0.10});
    auto ls = s.build_system(sops, mops, hist, /*order=*/2, /*dt=*/0.10);

    const double h_n = 0.10;
    const double delta = 2.0;
    const double a0 = (1.0 + 2.0 * delta) / (h_n * (1.0 + delta));  // 5 / (0.1 * 3) ≈ 16.667
    const double a1 = -(1.0 + delta) / (h_n * delta);              // -3 / (0.1 * 2) = -15.0
    const double a2 = delta / (h_n * (1.0 + delta));               // 2 / (0.1 * 3) ≈ 6.667

    EXPECT_NEAR(ls.A.coeff(0, 0), 1.0 + a0 * 0.1, 1e-9);
    EXPECT_NEAR(ls.b(0), a0 * 0.1 * 110.0 + a1 * 0.1 * 105.0 + a2 * 0.1 * 100.0, 1e-9);
}

TEST(Bdf2SchemeTest, StartsAsOrder1)
{
    TimeStepBuffer hist(3, 4);
    hist.reset({300.0, 300.0, 300.0});
    // history.size() == 1 — first call must request order=1

    Bdf2Scheme s(TimeSchemeConfig{TimeSchemeKind::Bdf2, /*initial_dt=*/0.1});
    auto d = s.select_step(hist, 0.0);
    EXPECT_EQ(d.order, 1u);
}

TEST(Bdf2SchemeTest, PromoteToOrder2WhenHistoryBigEnough)
{
    TimeStepBuffer hist(3, 4);
    hist.reset({300.0, 300.0, 300.0});
    hist.push({301.0, 301.0, 301.0}, 0.1);
    // history.size() == 2 → BDF2 eligible

    Bdf2Scheme s(TimeSchemeConfig{TimeSchemeKind::Bdf2, /*initial_dt=*/0.1});
    auto d = s.select_step(hist, 0.1);
    EXPECT_EQ(d.order, 2u);
}

TEST(Bdf2SchemeTest, BuildSystemFallsBackToBdf1WithTinyHistory)
{
    // Even if order=2 is passed, if history.size() < 2 the implementation
    // must fall back to BDF1.
    StaticOpsResult sops = make_static(1.0, 2.0, 3.0);
    MassOpsResult mops = make_mass(0.1, 0.2, 0.3);
    TimeStepBuffer hist(3, 4);
    hist.reset({300.0, 300.0, 300.0});
    // size == 1

    Bdf2Scheme s(TimeSchemeConfig{TimeSchemeKind::Bdf2, /*initial_dt=*/0.1});
    auto ls = s.build_system(sops, mops, hist, /*order=*/2, /*dt=*/0.1);

    // BDF1 coefficients: A(c,c) = K(c,c) + M(c)/dt, b = M(c) * T_prev / dt
    EXPECT_NEAR(ls.A.coeff(0, 0), 1.0 + 0.1 / 0.1, 1e-9);
    EXPECT_NEAR(ls.b(0), 0.1 * 300.0 / 0.1, 1e-9);
}
