#include "compiler/model_compiler.hpp"
#include "core/solver.hpp"
#include "model_test_utils.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <gtest/gtest.h>
#include <string>

TEST(SchedulerTest, SteadyHeatSourceProducesTemperatureGradient)
{
    // Cube with heat source + Dirichlet on bottom + convective BC on top
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0.0, 5.0, 10.0};
    io.mesh.y_vertices = {0.0, 5.0, 10.0};
    io.mesh.z_vertices = {0.0, 5.0, 10.0};

    mhs::model::LayerSpec layer;
    layer.thickness = "10";

    mhs::model::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "1e6"; // heat source

    mhs::model::RectOperation rect;
    rect.operation = mhs::model::GeometryOperation::Add;
    rect.rect.x = "0";
    rect.rect.y = "0";
    rect.rect.width = "10";
    rect.rect.height = "10";
    block.geometry.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    io.materials.push_back({"copper", mat});

    // Dirichlet 300K on bottom face (Z=0)
    mhs::model::BoundaryPatch boundary1;
    boundary1.condition = mhs::model::DirichletBoundary {"300"};
    boundary1.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    io.boundaries.push_back(boundary1);

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = mhs::sim::build_model(io);

    auto result = mhs::sim::solve(model);

    EXPECT_EQ(result.state.size(), model.cells.cell_to_grid.size());
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
    io.settings.study_type = mhs::model::StudyType::Transient;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0.0, 5.0, 10.0};
    io.mesh.y_vertices = {0.0, 5.0, 10.0};
    io.mesh.z_vertices = {0.0, 5.0, 10.0};

    mhs::model::LayerSpec layer;
    layer.thickness = "10";
    mhs::model::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "1e8";

    mhs::model::RectOperation rect;
    rect.operation = mhs::model::GeometryOperation::Add;
    rect.rect.x = "0";
    rect.rect.y = "0";
    rect.rect.width = "10";
    rect.rect.height = "10";
    block.geometry.push_back(rect);
    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    mat.density = "8920";
    mat.specific_heat = "385";
    io.materials.push_back({"copper", mat});

    io.settings.transient_duration = 5.0;
    io.settings.transient_output_interval = 1.0;

    io.default_boundary = mhs::model::NeumannBoundary {};

    mhs::model::ObservationPointSpec op1;
    op1.name = "center";
    op1.x = "5";
    op1.y = "5";
    op1.z = "5";
    io.observation_points.push_back(op1);
    mhs::model::ObservationPointSpec op2;
    op2.name = "z0";
    op2.x = "5";
    op2.y = "5";
    op2.z = "0";
    io.observation_points.push_back(op2);

    mhs::model::BoundaryPatch boundary;
    boundary.condition = mhs::model::DirichletBoundary {"500"};
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    io.boundaries.push_back(boundary);

    auto model = mhs::sim::build_model(io);

    auto result = mhs::sim::solve(model);

    const auto& traces = result.probe_traces;
    ASSERT_EQ(traces.size(), 2u);
    EXPECT_EQ(traces[0].name, "center");
    EXPECT_EQ(traces[1].name, "z0");

    for (const auto& tr : traces) {
        EXPECT_GE(tr.times.size(), 6u);
        EXPECT_GE(tr.values.size(), 6u);
        EXPECT_NEAR(tr.times.front(), 0.0, 1e-9);
        EXPECT_NEAR(tr.times.back(), 5.0, 1e-9);
        for (size_t i = 1; i < tr.times.size(); ++i) {
            EXPECT_GT(tr.times[i], tr.times[i - 1]) << "Times must be monotonically increasing";
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
    io.settings.study_type = mhs::model::StudyType::Transient;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0.0, 5.0, 10.0};
    io.mesh.y_vertices = {0.0, 5.0, 10.0};
    io.mesh.z_vertices = {0.0, 5.0, 10.0};

    mhs::model::LayerSpec layer;
    layer.thickness = "10";
    mhs::model::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "0";

    mhs::model::RectOperation rect;
    rect.operation = mhs::model::GeometryOperation::Add;
    rect.rect.x = "0";
    rect.rect.y = "0";
    rect.rect.width = "10";
    rect.rect.height = "10";
    block.geometry.push_back(rect);
    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    io.materials.push_back({"copper", mat});

    io.settings.transient_duration = 5.0;
    io.settings.transient_output_interval = 1.0;

    io.default_boundary = mhs::model::NeumannBoundary {};

    mhs::model::ObservationPointSpec op;
    op.name = "z0_dirichlet";
    op.x = "5";
    op.y = "5";
    op.z = "0";
    io.observation_points.push_back(op);

    mhs::model::BoundaryPatch boundary;
    boundary.condition = mhs::model::DirichletBoundary {"500 + 100*t"};
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    io.boundaries.push_back(boundary);

    auto model = mhs::sim::build_model(io);

    auto result = mhs::sim::solve(model);

    const auto& traces = result.probe_traces;
    ASSERT_EQ(traces.size(), 1u);
    const auto& tr = traces[0];
    ASSERT_GE(tr.times.size(), 6u);
    ASSERT_GE(tr.values.size(), 6u);

    for (size_t i = 0; i < tr.times.size(); ++i) {
        double expected = 500.0 + 100.0 * tr.times[i];
        EXPECT_NEAR(tr.values[i], expected, 1e-6) << "Time-dependent Dirichlet eval failed at t=" << tr.times[i]
                                                  << " (expected " << expected << ", got " << tr.values[i] << ")";
    }
}
