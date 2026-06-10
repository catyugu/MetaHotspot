#include "common/internal_model.hpp"
#include "common/io_model.hpp"
#include "postprocessor/postprocessor.hpp"
#include "preprocessor/preprocessor.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace mhs::sim;

// Helper: build a minimal mhs::core::IOStructure with a uniform grid
static mhs::core::IOStructure make_simple_uniform_grid_io()
{
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    mhs::core::Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    mhs::core::Block block;
    block.name = "test_block";
    block.material_name = "copper";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    mhs::core::Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::core::Material mat;
    mat.name = "copper";
    mat.kx = mat.ky = mat.kz = "400";
    io.materials["copper"] = mat;

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    return io;
}

TEST(PostprocessorTest, UniformGridInterpolationMatchesSimpleAverage)
{
    // On a uniform grid, distance-weighted interpolation should
    // produce the same result as a simple arithmetic mean.
    auto io = make_simple_uniform_grid_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = static_cast<int>(model->cells.cell_bcs.size());
    // Set cell temperatures to known values
    std::vector<double> cell_T(N);
    for (int i = 0; i < N; i++) {
        cell_T[i] = 300.0 + i * 10.0; // 300, 310, 320, ...
    }

    auto node_T = mhs::post::interpolate_cell_to_node(*model, cell_T, 0.0);

    // On a uniform 5mm grid, the interior node at (vx=1, vy=1, vz=1)
    // is shared by all 8 cells. The node is at the geometric center
    // of the 8 cell centers, so all distances are equal,
    // and weighted average equals arithmetic mean.
    int node_ny = model->mesh.ny + 1;
    int node_nz = model->mesh.nz + 1;
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
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    mhs::core::Layer layer;
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    mhs::core::Block block;
    block.name = "b1";
    block.material_name = "copper";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    mhs::core::Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::core::Material copper;
    copper.name = "copper";
    copper.kx = copper.ky = copper.kz = "400";
    io.materials["copper"] = copper;

    // Dirichlet BC on bottom Z face (Z=0) at 500K
    mhs::core::Boundary boundary_dirichlet;
    boundary_dirichlet.name = "bc_dirichlet";
    boundary_dirichlet.bc_type = mhs::core::ThermalBCType::FirstType;
    boundary_dirichlet.first.temperature = "500";
    boundary_dirichlet.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary_dirichlet);

    // Neumann(0) for all other faces (adiabatic)
    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    // Set all cell temperatures to something different from 500
    int N = static_cast<int>(model->cells.cell_bcs.size());
    std::vector<double> cell_T(N, 400.0);

    auto node_T = mhs::post::interpolate_cell_to_node(*model, cell_T, 0.0);

    int node_ny = model->mesh.ny + 1;
    int node_nz = model->mesh.nz + 1;

    // Corner node (vx=0, vy=0, vz=0) touches Dirichlet Z-bottom AND
    // Neumann Y-bottom AND Neumann X-bottom. Dirichlet must win.
    int corner_idx = 0 * node_ny * node_nz + 0 * node_nz + 0;
    EXPECT_NEAR(node_T[corner_idx], 500.0, 1e-6)
        << "Corner node touching Dirichlet must have exactly 500K, not an average";

    // All nodes on Z=0 face should be 500K (they all touch Dirichlet)
    for (int vx = 0; vx < model->mesh.nx + 1; vx++) {
        for (int vy = 0; vy < model->mesh.ny + 1; vy++) {
            int idx = vx * node_ny * node_nz + vy * node_nz + 0;
            EXPECT_NEAR(node_T[idx], 500.0, 1e-6) << "Z=0 node at (vx=" << vx << ", vy=" << vy << ") must be 500K";
        }
    }
}

TEST(PostprocessorTest, DirichletBCOverridesBoundaryNodes)
{
    // A cube with Dirichlet BC on the bottom face (Z=0) at 500K.
    // All boundary nodes on the Z=0 face should have exactly 500K,
    // not the interpolated cell-center average.
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    mhs::core::Layer layer;
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    mhs::core::Block block;
    block.name = "b1";
    block.material_name = "copper";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    mhs::core::Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::core::Material copper;
    copper.name = "copper";
    copper.kx = copper.ky = copper.kz = "400";
    io.materials["copper"] = copper;

    // Dirichlet BC on bottom face (Z=0) at 500K
    mhs::core::Boundary boundary;
    boundary.name = "bc1";
    boundary.bc_type = mhs::core::ThermalBCType::FirstType;
    boundary.first.temperature = "500";
    boundary.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary);

    // Neumann(0) for all other faces (adiabatic)
    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = static_cast<int>(model->cells.cell_bcs.size());
    EXPECT_EQ(N, 8);

    // Solve for steady-state temperatures (use scheduler)
    // For a simple test, we'll just use uniform cell temperatures
    // and verify the boundary node override mechanism.
    // The Dirichlet BC nodes should always be 500K regardless of
    // what the cell temperatures are.
    std::vector<double> cell_T(N, 400.0); // All cells at 400K (doesn't matter for boundary test)

    auto node_T = mhs::post::interpolate_cell_to_node(*model, cell_T, 0.0);

    int node_ny = model->mesh.ny + 1;
    int node_nz = model->mesh.nz + 1;

    // All nodes with vz=0 lie on the Z bottom boundary face.
    // They should have Dirichlet value = 500K, not interpolated from cells.
    for (int vx = 0; vx < model->mesh.nx + 1; vx++) {
        for (int vy = 0; vy < model->mesh.ny + 1; vy++) {
            int node_idx = vx * node_ny * node_nz + vy * node_nz + 0;
            EXPECT_NEAR(node_T[node_idx], 500.0, 1e-6)
                << "Node at (vx=" << vx << ", vy=" << vy << ", vz=0) should be 500K (Dirichlet)";
        }
    }

    // Interior nodes (vz > 0) should NOT be 500K
    // They should be interpolated from cell temperatures
    for (int vx = 0; vx < model->mesh.nx + 1; vx++) {
        for (int vy = 0; vy < model->mesh.ny + 1; vy++) {
            for (int vz = 1; vz < model->mesh.nz + 1; vz++) {
                int node_idx = vx * node_ny * node_nz + vy * node_nz + vz;
                // Interior nodes should not equal Dirichlet value (unless coincidentally)
                // At least check they are not NaN
                EXPECT_FALSE(std::isnan(node_T[node_idx]))
                    << "Interior node at (vx=" << vx << ", vy=" << vy << ", vz=" << vz << ") should not be NaN";
            }
        }
    }
}

TEST(PostprocessorTest, DirichletEvalUsesProvidedTime)
{
    // Regression: interpolate_cell_to_node 之前把 FieldContext.t 硬编码成 0.0，
    // 导致时间依赖的 BC 表达式在任意时刻都被求值成 t=0 时的结果。
    // 本测试对 z=0 面上 Dirichlet 表达式 "500 + 100*t" 在 t=0 和 t=10
    // 两个时间点分别求值，验证两者差异严格为 1000K（旧实现会得到 0K 差异）。
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    mhs::core::Layer layer;
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    mhs::core::Block block;
    block.name = "b1";
    block.material_name = "copper";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    mhs::core::Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "10";
    rect.height_expr = "10";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::core::Material copper;
    copper.name = "copper";
    copper.kx = copper.ky = copper.kz = "400";
    io.materials["copper"] = copper;

    // 时间依赖的 Dirichlet 表达式：T_bc = 500 + 100*t
    mhs::core::Boundary boundary;
    boundary.name = "bc_time_dep";
    boundary.bc_type = mhs::core::ThermalBCType::FirstType;
    boundary.first.temperature = "500 + 100*t";
    boundary.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary);

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = static_cast<int>(model->cells.cell_bcs.size());
    std::vector<double> cell_T(N, 400.0);

    int node_ny = model->mesh.ny + 1;
    int node_nz = model->mesh.nz + 1;
    // z=0 面的中心节点 (vx=1, vy=1, vz=0)
    int node_idx = 1 * node_ny * node_nz + 1 * node_nz + 0;

    // t=0 时刻：500 + 100*0 = 500
    auto node_T_0 = mhs::post::interpolate_cell_to_node(*model, cell_T, 0.0);
    EXPECT_NEAR(node_T_0[node_idx], 500.0, 1e-6) << "At t=0, time-dependent Dirichlet should evaluate to 500";

    // t=10 时刻：500 + 100*10 = 1500
    auto node_T_10 = mhs::post::interpolate_cell_to_node(*model, cell_T, 10.0);
    EXPECT_NEAR(node_T_10[node_idx], 1500.0, 1e-6) << "At t=10, time-dependent Dirichlet should evaluate to 1500";

    // 旧实现下两个时刻结果相等；修复后差异必须严格为 1000
    EXPECT_NEAR(node_T_10[node_idx] - node_T_0[node_idx], 1000.0, 1e-6)
        << "Time must actually flow through the Dirichlet eval";
}
