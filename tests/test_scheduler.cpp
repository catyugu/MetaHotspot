#include "preprocessor/preprocessor.hpp"
#include "scheduler/scheduler.hpp"
#include "solver/solver.hpp"
#include <gtest/gtest.h>

using namespace mhs;

// Helper: build a minimal IOStructure for a simple uniform cube
static IOStructure make_simple_cube_io()
{
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "test_block";
    block.material_name = "copper";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    Material mat;
    mat.name = "copper";
    mat.daore_xishu = "400";
    io.materials["copper"] = mat;

    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    return io;
}

TEST(SchedulerTest, ConstructWithConfig)
{
    SchedulerConfig config;
    config.is_steady = true;
    config.max_nonlinear_iterations = 10;
    Scheduler scheduler(config);
}

TEST(SchedulerTest, SetModelAndSolver)
{
    auto io = make_simple_cube_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    Scheduler scheduler;
    scheduler.setModel(model.get());
    scheduler.setSolver(Solver::create(SolverType::SparseLU));
}

TEST(SchedulerTest, SteadyRunProducesSolution)
{
    // Simple cube with Dirichlet BC on one face, Neumann(0) on others
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "test_block";
    block.material_name = "copper";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    Material mat;
    mat.name = "copper";
    mat.daore_xishu = "400";
    io.materials["copper"] = mat;

    // Dirichlet 500K on bottom face (Z=0)
    Boundary boundary;
    boundary.name = "bc1";
    boundary.bc_type = ThermalBCType::FirstType;
    boundary.first.temperature = "500";
    boundary.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary);

    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    SchedulerConfig config;
    config.is_steady = true;
    config.max_nonlinear_iterations = 50;

    Scheduler scheduler(config);
    scheduler.setModel(model.get());
    scheduler.setSolver(Solver::create(SolverType::Pardiso));

    scheduler.run();

    const auto& solution = scheduler.solution();
    EXPECT_EQ(solution.size(), model->cells.cell_count);

    // With Dirichlet 500K on bottom and Neumann(0) on all other faces,
    // and no heat source, steady state should be T=500K everywhere
    // (for a cube with uniform material and adiabatic sides)
    // Bottom cells should be ~500K (they have Dirichlet BC)
    // All cells should approach 500K since heat can only leave through bottom
    for (size_t i = 0; i < solution.size(); i++) {
        EXPECT_NEAR(solution[i], 500.0, 50.0) << "Cell " << i << " temperature";
    }
}

TEST(SchedulerTest, SteadyHeatSourceProducesTemperatureGradient)
{
    // Cube with heat source + Dirichlet on bottom + convective BC on top
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "test_block";
    block.material_name = "copper";
    block.ti_reyuan_expr = "1e6"; // heat source
    block.is_normal_material = true;

    Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    Material mat;
    mat.name = "copper";
    mat.daore_xishu = "400";
    io.materials["copper"] = mat;

    // Dirichlet 300K on bottom face (Z=0)
    Boundary boundary1;
    boundary1.name = "bc1";
    boundary1.bc_type = ThermalBCType::FirstType;
    boundary1.first.temperature = "300";
    boundary1.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary1);

    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    SchedulerConfig config;
    config.is_steady = true;
    config.max_nonlinear_iterations = 50;

    Scheduler scheduler(config);
    scheduler.setModel(model.get());
    scheduler.setSolver(Solver::create(SolverType::Pardiso));

    scheduler.run();

    const auto& solution = scheduler.solution();
    EXPECT_EQ(solution.size(), model->cells.cell_count);

    // With heat source and Dirichlet 300K at bottom, temperatures should be > 300K
    double max_T = 0.0;
    for (const auto& t : solution) {
        max_T = std::max(max_T, t);
    }
    EXPECT_GT(max_T, 300.0) << "Heat source should raise temperature above 300K";
}

TEST(SchedulerTest, SchedulerConfigDefaults)
{
    SchedulerConfig config;
    EXPECT_EQ(config.max_nonlinear_iterations, 50);
    EXPECT_NEAR(config.underrelaxation, 1.0, 1e-10);
    EXPECT_FALSE(config.is_steady);
}

TEST(SchedulerTest, SchedulerConfigCustom)
{
    SchedulerConfig config;
    config.is_steady = true;
    config.max_nonlinear_iterations = 20;
    config.underrelaxation = 0.7;

    EXPECT_TRUE(config.is_steady);
    EXPECT_EQ(config.max_nonlinear_iterations, 20);
    EXPECT_NEAR(config.underrelaxation, 0.7, 1e-10);
}

TEST(SchedulerTest, TransientStepCallbackFiresForEachStep)
{
    // Transient 5 steps with heat source; callback should fire 6 times
    // (t=0 initial + 5 step ends).
    IOStructure io;
    io.study_type = StudyType::Transient;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    Layer layer;
    layer.name = "l";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";
    Block block;
    block.name = "b";
    block.material_name = "copper";
    block.ti_reyuan_expr = "1e8"; // strong heat source to force T to rise
    block.is_normal_material = true;
    Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);
    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    Material mat;
    mat.name = "copper";
    mat.daore_xishu = "400";
    mat.midu = "8920";
    mat.bi_rerong = "385";
    io.materials["copper"] = mat;

    io.transient_duration = 5.0;
    io.transient_time_step = 1.0;

    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    SchedulerConfig config;
    config.transient_duration = 5.0;
    config.time_step = 1.0;
    config.max_nonlinear_iterations = 20;

    Scheduler scheduler(config);
    scheduler.setModel(model.get());
    scheduler.setSolver(Solver::create(SolverType::Pardiso));

    std::vector<double> times_seen;
    std::vector<double> temps_at_first_cell;
    StepCallback cb = [&times_seen, &temps_at_first_cell](double t, int /*step*/, const std::vector<double>& cell_T) {
        times_seen.push_back(t);
        if (!cell_T.empty())
            temps_at_first_cell.push_back(cell_T[0]);
    };
    scheduler.setCallback(std::move(cb));

    scheduler.run();

    // 6 fires expected: t=0, t=1, t=2, t=3, t=4, t=5
    EXPECT_EQ(times_seen.size(), 6u);
    EXPECT_NEAR(times_seen.front(), 0.0, 1e-9);
    EXPECT_NEAR(times_seen.back(), 5.0, 1e-9);
    for (size_t i = 1; i < times_seen.size(); ++i) {
        EXPECT_GT(times_seen[i], times_seen[i - 1]) << "Times must be monotonically increasing";
    }

    // With 1e8 W/m^3 heat source in 10mm copper cube, T should rise from 300K
    ASSERT_GE(temps_at_first_cell.size(), 2u);
    EXPECT_GT(temps_at_first_cell.back(), temps_at_first_cell.front())
        << "First cell temperature must rise over time with a strong heat source";
    EXPECT_NEAR(temps_at_first_cell.front(), 300.0, 1e-3) << "t=0 must be initial temperature";
}