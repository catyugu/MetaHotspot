#include "preprocessor/preprocessor.hpp"
#include "scheduler/scheduler.hpp"
#include "solver/solver.hpp"
#include <gtest/gtest.h>

using namespace mhs;
using namespace mhs::model;

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
    block.thickness_expr = "10";
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
    config.max_newton_iterations = 10;
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
    block.thickness_expr = "10";
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
    config.max_newton_iterations = 50;
    config.newton_tolerance = 1e-6;

    Scheduler scheduler(config);
    scheduler.setModel(model.get());
    scheduler.setSolver(Solver::create(SolverType::SparseLU));

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
    block.thickness_expr = "10";
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
    config.max_newton_iterations = 50;
    config.newton_tolerance = 1e-6;

    Scheduler scheduler(config);
    scheduler.setModel(model.get());
    scheduler.setSolver(Solver::create(SolverType::SparseLU));

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
    EXPECT_EQ(config.max_newton_iterations, 50);
    EXPECT_NEAR(config.newton_tolerance, 1e-6, 1e-10);
    EXPECT_NEAR(config.underrelaxation, 1.0, 1e-10);
    EXPECT_FALSE(config.is_steady);
}

TEST(SchedulerTest, SchedulerConfigCustom)
{
    SchedulerConfig config;
    config.is_steady = true;
    config.max_newton_iterations = 20;
    config.newton_tolerance = 1e-4;
    config.underrelaxation = 0.7;

    EXPECT_TRUE(config.is_steady);
    EXPECT_EQ(config.max_newton_iterations, 20);
    EXPECT_NEAR(config.newton_tolerance, 1e-4, 1e-10);
    EXPECT_NEAR(config.underrelaxation, 0.7, 1e-10);
}