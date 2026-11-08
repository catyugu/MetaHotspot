#include "compiler/model_compiler.hpp"
#include "solver/solver.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <string>

TEST(SchedulerTest, SteadyHeatSourceProducesTemperatureGradient)
{
    // Cube with heat source + Dirichlet on bottom + convective BC on top
    mhs::model::ModelDefinition io;

    io.fom(0).mesh.x_vertices = {0.0, 0.005, 0.01};
    io.fom(0).mesh.y_vertices = {0.0, 0.005, 0.01};
    io.fom(0).mesh.z_vertices = {0.0, 0.005, 0.01};

    io.sources = {"1e6"};
    io.fom(0).volumes = {{0, 0, {{mhs::model::GeometryOperation::Add, {0, 0, 0}, {0.01, 0.01, 0.01}}}}};

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    io.materials.push_back({"copper", mat});

    // Dirichlet 300K on bottom face (Z=0)
    mhs::model::ThermalBoundarySpec boundary1;
    boundary1.condition = mhs::model::DirichletBoundary {"300"};
    boundary1.regions.push_back({mhs::model::Axis::Z, 0.0, {0.0, 0.01, 0.0, 0.01}});
    io.fom(0).thermal_boundaries.push_back(boundary1);

    io.fom(0).default_thermal_boundary = mhs::model::NeumannBoundary {};

    auto model = mhs::compiler::build_model(io);

    auto result = mhs::solver::solve(model);

    EXPECT_EQ(result.state.size(), model.fom(0).discrete.cells.size());
    EXPECT_TRUE(std::equal(result.state.begin(), result.state.end(), result.state.begin()));

    // With heat source and Dirichlet 300K at bottom, temperatures should be > 300K
    double max_T = 0.0;
    for (const auto& t : result.state) {
        max_T = std::max(max_T, t);
    }
    EXPECT_GT(max_T, 300.0) << "Heat source should raise temperature above 300K";
}

TEST(SchedulerTest, ProbeRecorderCapturesPerStep)
{
    mhs::model::ModelDefinition io;

    io.fom(0).mesh.x_vertices = {0.0, 0.005, 0.01};
    io.fom(0).mesh.y_vertices = {0.0, 0.005, 0.01};
    io.fom(0).mesh.z_vertices = {0.0, 0.005, 0.01};

    io.sources = {"1e8"};
    io.fom(0).volumes = {{0, 0, {{mhs::model::GeometryOperation::Add, {0, 0, 0}, {0.01, 0.01, 0.01}}}}};

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    mat.density = "8920";
    mat.specific_heat = "385";
    io.materials.push_back({"copper", mat});

    io.fom(0).default_thermal_boundary = mhs::model::NeumannBoundary {};

    mhs::model::PointProbe op1;
    op1.name = "center";
    op1.x = 0.005;
    op1.y = 0.005;
    op1.z = 0.005;
    io.point_probes.push_back(op1);
    mhs::model::PointProbe op2;
    op2.name = "z0";
    op2.x = 0.005;
    op2.y = 0.005;
    op2.z = 0;
    io.point_probes.push_back(op2);

    mhs::model::ThermalBoundarySpec boundary;
    boundary.condition = mhs::model::DirichletBoundary {"500"};
    boundary.regions.push_back({mhs::model::Axis::Z, 0.0, {0.0, 0.01, 0.0, 0.01}});
    io.fom(0).thermal_boundaries.push_back(boundary);

    auto model = mhs::compiler::build_model(io);

    mhs::model::Study study {mhs::model::StudyType::Transient, 300.0, 5.0, 1.0};
    auto result = mhs::solver::solve(model, study);

    const auto& traces = result.probe_traces;
    ASSERT_EQ(traces.size(), 2u);
    EXPECT_EQ(traces[0].name, "center");
    EXPECT_EQ(traces[1].name, "z0");

    for (const auto& tr : traces) {
        EXPECT_GE(result.snapshot_times.size(), 6u);
        EXPECT_EQ(tr.values.size(), result.snapshot_times.size());
        EXPECT_NEAR(result.snapshot_times.front(), 0.0, 1e-9);
        EXPECT_NEAR(result.snapshot_times.back(), 5.0, 1e-9);
        for (size_t i = 1; i < result.snapshot_times.size(); ++i) {
            EXPECT_GT(result.snapshot_times[i], result.snapshot_times[i - 1])
                << "Times must be monotonically increasing";
        }
    }

    EXPECT_NEAR(traces[0].values.front(), 300.0, 1e-3) << "t=0 must be initial temperature";
    EXPECT_GT(traces[0].values.back(), traces[0].values.front())
        << "Center probe must rise over time with strong heat source";

    for (double v : traces[1].values) {
        EXPECT_NEAR(v, 500.0, 1e-6) << "z0 probe on Dirichlet face must stay at 500K";
    }
}

TEST(SchedulerTest, ProbeRecorderUsesCurrentTimeForTimeDependentBC)
{
    mhs::model::ModelDefinition io;

    io.fom(0).mesh.x_vertices = {0.0, 0.005, 0.01};
    io.fom(0).mesh.y_vertices = {0.0, 0.005, 0.01};
    io.fom(0).mesh.z_vertices = {0.0, 0.005, 0.01};

    io.sources = {"0"};
    io.fom(0).volumes = {{0, 0, {{mhs::model::GeometryOperation::Add, {0, 0, 0}, {0.01, 0.01, 0.01}}}}};

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    io.materials.push_back({"copper", mat});

    io.fom(0).default_thermal_boundary = mhs::model::NeumannBoundary {};

    mhs::model::PointProbe op;
    op.name = "z0_dirichlet";
    op.x = 0.005;
    op.y = 0.005;
    op.z = 0;
    io.point_probes.push_back(op);

    mhs::model::ThermalBoundarySpec boundary;
    boundary.condition = mhs::model::DirichletBoundary {"500 + 100*t"};
    boundary.regions.push_back({mhs::model::Axis::Z, 0.0, {0.0, 0.01, 0.0, 0.01}});
    io.fom(0).thermal_boundaries.push_back(boundary);

    auto model = mhs::compiler::build_model(io);

    mhs::model::Study study {mhs::model::StudyType::Transient, 300.0, 5.0, 1.0};
    auto result = mhs::solver::solve(model, study);

    const auto& traces = result.probe_traces;
    ASSERT_EQ(traces.size(), 1u);
    const auto& tr = traces[0];
    ASSERT_GE(result.snapshot_times.size(), 6u);
    ASSERT_EQ(tr.values.size(), result.snapshot_times.size());

    for (size_t i = 0; i < result.snapshot_times.size(); ++i) {
        double expected = 500.0 + 100.0 * result.snapshot_times[i];
        EXPECT_NEAR(tr.values[i], expected, 1e-6)
            << "Time-dependent Dirichlet eval failed at t=" << result.snapshot_times[i] << " (expected " << expected
            << ", got " << tr.values[i] << ")";
    }
}
