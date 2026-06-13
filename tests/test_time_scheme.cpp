#include "assembler/assembler.hpp"
#include "data/internal_model.hpp"
#include "data/time_step_buffer.hpp"
#include "time_scheme/time_scheme.hpp"
#include "time_scheme/bdf1_scheme.hpp"
#include "time_scheme/bdf2_scheme.hpp"
#include "time_scheme/adaptive_bdf_scheme.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace mhs::sim;
using namespace mhs::sim::time_scheme;
using mhs::core::GlobalState;
using mhs::core::TimeStepBuffer;

namespace {

    // Build a tiny K (3x3 diagonal) and M_diag (length 3) for unit testing
    // time-scheme build_system without going through the full assembler.
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

    TimeStepBuffer make_history(const std::vector<double>& T_initial, double /*t0*/ = 0.0)
    {
        TimeStepBuffer buf(T_initial.size(), 4);
        buf.reset(T_initial);
        return buf;
    }
}

TEST(TimeSchemeTest, Bdf1SchemeSelectStepReturnsInitialDt)
{
    Bdf1Scheme s(TimeSchemeConfig{TimeSchemeKind::Bdf1, /*initial_dt=*/0.5});
    auto d = s.select_step(make_history({300.0, 300.0, 300.0}), 0.0);
    EXPECT_DOUBLE_EQ(d.dt, 0.5);
    EXPECT_EQ(d.order, 1u);
}

TEST(TimeSchemeTest, Bdf1SchemeBuildSystemCoefficient)
{
    // K = diag(1, 2, 3), M_diag = (0.1, 0.2, 0.3), f_static = 0
    // dt = 0.1, T_prev = (300, 400, 500)
    //
    // A(c,c) = K(c,c) + M_diag(c) / dt
    // b(c)   = M_diag(c) * T_prev(c) / dt
    StaticOpsResult sops = make_static(1.0, 2.0, 3.0);
    MassOpsResult mops = make_mass(0.1, 0.2, 0.3);
    TimeStepBuffer hist = make_history({300.0, 400.0, 500.0});

    Bdf1Scheme s(TimeSchemeConfig{});
    auto ls = s.build_system(sops, mops, hist, /*order=*/1, /*dt=*/0.1);

    EXPECT_NEAR(ls.A.coeff(0, 0), 1.0 + 0.1 / 0.1, 1e-12);     // = 2.0
    EXPECT_NEAR(ls.A.coeff(1, 1), 2.0 + 0.2 / 0.1, 1e-12);     // = 4.0
    EXPECT_NEAR(ls.A.coeff(2, 2), 3.0 + 0.3 / 0.1, 1e-12);     // = 6.0
    EXPECT_NEAR(ls.b(0), 0.1 * 300.0 / 0.1, 1e-12);            // = 300
    EXPECT_NEAR(ls.b(1), 0.2 * 400.0 / 0.1, 1e-12);            // = 800
    EXPECT_NEAR(ls.b(2), 0.3 * 500.0 / 0.1, 1e-12);            // = 1500
}

TEST(TimeSchemeTest, Bdf1SchemeAcceptOrRejectAlwaysAccepts)
{
    Bdf1Scheme s(TimeSchemeConfig{});
    auto dec = s.accept_or_reject(make_history({300.0}), {301.0}, {1.0, 2.0});
    EXPECT_EQ(dec, AcceptDecision::Accept);
}

TEST(TimeSchemeTest, Bdf1SchemeFactoryCreatesBdf1)
{
    TimeSchemeConfig cfg;
    cfg.kind = TimeSchemeKind::Bdf1;
    auto p = create_scheme(cfg);
    EXPECT_NE(dynamic_cast<Bdf1Scheme*>(p.get()), nullptr);
}

TEST(TimeSchemeTest, Bdf2SchemeFactoryCreatesBdf2)
{
    TimeSchemeConfig cfg;
    cfg.kind = TimeSchemeKind::Bdf2;
    auto p = create_scheme(cfg);
    EXPECT_NE(dynamic_cast<Bdf2Scheme*>(p.get()), nullptr);
}

TEST(TimeSchemeTest, AdaptiveBdfFactoryCreatesAdaptive)
{
    TimeSchemeConfig cfg;
    cfg.kind = TimeSchemeKind::AdaptiveBdf;
    auto p = create_scheme(cfg);
    EXPECT_NE(dynamic_cast<AdaptiveBdfScheme*>(p.get()), nullptr);
}
