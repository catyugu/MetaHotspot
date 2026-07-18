#include "data/model.hpp"
#include "model/model_definition.hpp"
#include "model_test_utils.hpp"
#include "postprocessor/postprocessor.hpp"
#include "preprocessor/preprocessor.hpp"
#include <gtest/gtest.h>

using namespace mhs::sim;

// Helper: build a minimal mhs::model::ModelDefinition with a uniform grid
static mhs::model::ModelDefinition make_simple_uniform_grid_io()
{
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

    io.default_boundary = mhs::model::NeumannBoundary {};

    return io;
}

TEST(PostprocessorTest, UniformGridInterpolationMatchesSimpleAverage)
{
    // On a uniform grid, distance-weighted interpolation should
    // produce the same result as a simple arithmetic mean.
    auto io = make_simple_uniform_grid_io();
    auto model = build_model(io);

    int N = static_cast<int>(model.cells.material_id.size());
    // Set cell temperatures to known values
    std::vector<double> cell_T(N);
    for (int i = 0; i < N; i++) {
        cell_T[i] = 300.0 + i * 10.0; // 300, 310, 320, ...
    }

    auto node_T = mhs::post::interpolate_cell_to_node(model, cell_T, 0.0);

    // On a uniform 5mm grid, the interior node at (vx=1, vy=1, vz=1)
    // is shared by all 8 cells. The node is at the geometric center
    // of the 8 cell centers, so all distances are equal,
    // and weighted average equals arithmetic mean.
    int node_ny = model.mesh.ny + 1;
    int node_nz = model.mesh.nz + 1;
    int interior_node = 1 * node_ny * node_nz + 1 * node_nz + 1;

    // The 8 cells are at (0,0,0),(1,0,0),(0,1,0),(1,1,0),
    //                    (0,0,1),(1,0,1),(0,1,1),(1,1,1)
    double expected = 0.0;
    for (int i = 0; i < 8; i++) {
        expected += cell_T[i];
    }
    expected /= 8.0;

    EXPECT_NEAR(node_T[interior_node], expected, 1e-10);
}

TEST(PostprocessorTest, DirichletBCOverridesMixedBoundaryAtCorner)
{
    // A corner node that touches both a Dirichlet face and a Neumann face
    // must have the Dirichlet temperature exclusively — never averaged
    // with the Neumann-computed temperature.
    //
    // Setup: 10x10x10mm cube with Dirichlet 500K on Z bottom face and
    // Neumann(0) on Y=0 face (adiabatic). The corner node at (0, 0, 0)
    // touches both. Without Dirichlet priority, the result would be an
    // average of ~500 and ~400 ≈ ~450-500 (depending on cell temps).
    // With Dirichlet priority, the result must be exactly 500K.
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

    mhs::model::MaterialSpec copper;
    copper.conductivity_x = copper.conductivity_y = copper.conductivity_z = "400";
    io.materials.push_back({"copper", copper});

    // Dirichlet BC on bottom Z face (Z=0) at 500K
    mhs::model::BoundaryPatch boundary_dirichlet;
    boundary_dirichlet.condition = mhs::model::DirichletBoundary {"500"};
    boundary_dirichlet.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    io.boundaries.push_back(boundary_dirichlet);

    // Neumann(0) for all other faces (adiabatic)
    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    // Set all cell temperatures to something different from 500
    int N = static_cast<int>(model.cells.material_id.size());
    std::vector<double> cell_T(N, 400.0);

    auto node_T = mhs::post::interpolate_cell_to_node(model, cell_T, 0.0);

    int node_ny = model.mesh.ny + 1;
    int node_nz = model.mesh.nz + 1;

    // Corner node (vx=0, vy=0, vz=0) touches Dirichlet Z-bottom AND
    // Neumann Y-bottom AND Neumann X-bottom. Dirichlet must win.
    int corner_idx = 0 * node_ny * node_nz + 0 * node_nz + 0;
    EXPECT_NEAR(node_T[corner_idx], 500.0, 1e-6)
        << "Corner node touching Dirichlet must have exactly 500K, not an average";

    // All nodes on Z=0 face should be 500K (they all touch Dirichlet)
    for (int vx = 0; vx < model.mesh.nx + 1; vx++) {
        for (int vy = 0; vy < model.mesh.ny + 1; vy++) {
            int idx = vx * node_ny * node_nz + vy * node_nz + 0;
            EXPECT_NEAR(node_T[idx], 500.0, 1e-6) << "Z=0 node at (vx=" << vx << ", vy=" << vy << ") must be 500K";
        }
    }
}

TEST(PostprocessorTest, DirichletEvalUsesProvidedTime)
{
    // Regression: interpolate_cell_to_node 之前把 FieldContext.t 硬编码成 0.0，
    // 导致时间依赖的 BC 表达式在任意时刻都被求值成 t=0 时的结果。
    // 本测试对 z=0 面上 Dirichlet 表达式 "500 + 100*t" 在 t=0 和 t=10
    // 两个时间点分别求值，验证两者差异严格为 1000K（旧实现会得到 0K 差异）。
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

    mhs::model::MaterialSpec copper;
    copper.conductivity_x = copper.conductivity_y = copper.conductivity_z = "400";
    io.materials.push_back({"copper", copper});

    // 时间依赖的 Dirichlet 表达式：T_bc = 500 + 100*t
    mhs::model::BoundaryPatch boundary;
    boundary.condition = mhs::model::DirichletBoundary {"500 + 100*t"};
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    io.boundaries.push_back(boundary);

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    int N = static_cast<int>(model.cells.material_id.size());
    std::vector<double> cell_T(N, 400.0);

    int node_ny = model.mesh.ny + 1;
    int node_nz = model.mesh.nz + 1;
    // z=0 面的中心节点 (vx=1, vy=1, vz=0)
    int node_idx = 1 * node_ny * node_nz + 1 * node_nz + 0;

    // t=0 时刻：500 + 100*0 = 500
    auto node_T_0 = mhs::post::interpolate_cell_to_node(model, cell_T, 0.0);
    EXPECT_NEAR(node_T_0[node_idx], 500.0, 1e-6) << "At t=0, time-dependent Dirichlet should evaluate to 500";

    // t=10 时刻：500 + 100*10 = 1500
    auto node_T_10 = mhs::post::interpolate_cell_to_node(model, cell_T, 10.0);
    EXPECT_NEAR(node_T_10[node_idx], 1500.0, 1e-6) << "At t=10, time-dependent Dirichlet should evaluate to 1500";

    // 旧实现下两个时刻结果相等；修复后差异必须严格为 1000
    EXPECT_NEAR(node_T_10[node_idx] - node_T_0[node_idx], 1000.0, 1e-6)
        << "Time must actually flow through the Dirichlet eval";
}
