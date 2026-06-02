#include "model/internal_model.hpp"
#include "model/io_model.hpp"
#include "postprocessor/postprocessor.hpp"
#include "preprocessor/preprocessor.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace mhs;
using namespace mhs::model;

// Helper: build a minimal IOStructure with a uniform grid
static IOStructure make_simple_uniform_grid_io()
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

TEST(PostprocessorTest, UniformGridInterpolationMatchesSimpleAverage)
{
    // On a uniform grid, distance-weighted interpolation should
    // produce the same result as a simple arithmetic mean.
    auto io = make_simple_uniform_grid_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    // Set cell temperatures to known values
    std::vector<double> cell_T(N);
    for (int i = 0; i < N; i++) {
        cell_T[i] = 300.0 + i * 10.0; // 300, 310, 320, ...
    }

    Postprocessor postprocessor;
    auto node_T = postprocessor.interpolate_cell_to_node(*model, cell_T);

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
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "b1";
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

    Material copper;
    copper.name = "copper";
    copper.daore_xishu = "400";
    io.materials["copper"] = copper;

    // Dirichlet BC on bottom Z face (Z=0) at 500K
    Boundary boundary_dirichlet;
    boundary_dirichlet.name = "bc_dirichlet";
    boundary_dirichlet.bc_type = ThermalBCType::FirstType;
    boundary_dirichlet.first.temperature = "500";
    boundary_dirichlet.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary_dirichlet);

    // Neumann(0) for all other faces (adiabatic)
    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    // Set all cell temperatures to something different from 500
    int N = model->cells.cell_count;
    std::vector<double> cell_T(N, 400.0);

    Postprocessor postprocessor;
    auto node_T = postprocessor.interpolate_cell_to_node(*model, cell_T);

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
            EXPECT_NEAR(node_T[idx], 500.0, 1e-6)
                << "Z=0 node at (vx=" << vx << ", vy=" << vy << ") must be 500K";
        }
    }
}

TEST(PostprocessorTest, DirichletBCOverridesBoundaryNodes)
{
    // A cube with Dirichlet BC on the bottom face (Z=0) at 500K.
    // All boundary nodes on the Z=0 face should have exactly 500K,
    // not the interpolated cell-center average.
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
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "b1";
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

    Material copper;
    copper.name = "copper";
    copper.daore_xishu = "400";
    io.materials["copper"] = copper;

    // Dirichlet BC on bottom face (Z=0) at 500K
    Boundary boundary;
    boundary.name = "bc1";
    boundary.bc_type = ThermalBCType::FirstType;
    boundary.first.temperature = "500";
    boundary.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary);

    // Neumann(0) for all other faces (adiabatic)
    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    EXPECT_EQ(N, 8);

    // Solve for steady-state temperatures (use scheduler)
    // For a simple test, we'll just use uniform cell temperatures
    // and verify the boundary node override mechanism.
    // The Dirichlet BC nodes should always be 500K regardless of
    // what the cell temperatures are.
    std::vector<double> cell_T(N, 400.0); // All cells at 400K (doesn't matter for boundary test)

    Postprocessor postprocessor;
    auto node_T = postprocessor.interpolate_cell_to_node(*model, cell_T);

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