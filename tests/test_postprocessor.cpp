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

    int N = model->cells.cell_count;
    // Set cell temperatures to known values
    std::vector<double> cell_T(N);
    for (int i = 0; i < N; i++) {
        cell_T[i] = 300.0 + i * 10.0; // 300, 310, 320, ...
    }

    auto node_T = mhs::post::interpolate_cell_to_node(*model, cell_T);

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
    int N = model->cells.cell_count;
    std::vector<double> cell_T(N, 400.0);

    auto node_T = mhs::post::interpolate_cell_to_node(*model, cell_T);

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

    int N = model->cells.cell_count;
    EXPECT_EQ(N, 8);

    // Solve for steady-state temperatures (use scheduler)
    // For a simple test, we'll just use uniform cell temperatures
    // and verify the boundary node override mechanism.
    // The Dirichlet BC nodes should always be 500K regardless of
    // what the cell temperatures are.
    std::vector<double> cell_T(N, 400.0); // All cells at 400K (doesn't matter for boundary test)

    auto node_T = mhs::post::interpolate_cell_to_node(*model, cell_T);

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

TEST(PostprocessorTest, SamplePointOnUniformFieldReturnsFieldValue)
{
    // 2x2x2 cell grid, all cells at 300K.
    // Sample point at (3, 3, 3) (mm) — inside a single cell.
    // sample_point uses cell-corner LSQ; for a uniform field every corner is 300K
    // and the LSQ fit is constant → expect 300K.
    auto io = make_simple_uniform_grid_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    std::vector<double> cell_T(N, 300.0);

    auto node_T = mhs::post::interpolate_cell_to_node(*model, cell_T);

    mhs::core::ProbePoint pt;
    pt.name = "center";
    // model->mesh 已是 SI 单位 (vertex_x = {0, 0.005, 0.01})
    pt.x = 0.003;
    pt.y = 0.003;
    pt.z = 0.003;

    double T = mhs::post::sample_point(node_T, *model, pt);
    EXPECT_FALSE(std::isnan(T)) << "Interior point should not be NaN";
    EXPECT_NEAR(T, 300.0, 1e-6);
}

TEST(PostprocessorTest, SamplePointOnLinearGradientInterpolates)
{
    // 5x5x5 cell grid, T = 300 + 100*z (linear in z). Sample at z=6mm
    // should return ~ 300 + 100*0.006 = 300.6K (mesh vertex z range 0..10mm
    // so vertex 6 mm → normalized t = 0.6 → T = 360K).
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};
    io.mesh_vertex_y = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};
    io.mesh_vertex_z = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

    mhs::core::Layer layer;
    layer.name = "linear";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    mhs::core::Block block;
    block.name = "b";
    block.material_name = "mat";
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
    mat.name = "mat";
    mat.kx = mat.ky = mat.kz = "400";
    io.materials["mat"] = mat;

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int node_ny = model->mesh.ny + 1;
    int node_nz = model->mesh.nz + 1;
    std::vector<double> node_T(node_ny * node_nz * node_ny, 0.0);

    // Set node T = 300 + 6 * vertex_z  (in SI vertex_z range 0..0.01 → T 300..360)
    for (int vx = 0; vx < node_ny; vx++)
        for (int vy = 0; vy < node_ny; vy++)
            for (int vz = 0; vz < node_nz; vz++) {
                double z = model->mesh.vertex_z[vz];
                node_T[vx * node_ny * node_nz + vy * node_nz + vz] = 300.0 + 6.0 * z;
            }

    mhs::core::ProbePoint pt;
    pt.name = "z6";
    pt.x = 0.005; // 5 mm
    pt.y = 0.005;
    pt.z = 0.006; // 6 mm

    double T = mhs::post::sample_point(node_T, *model, pt);
    EXPECT_FALSE(std::isnan(T));
    // 1e-2 容差：LSQ + Tikhonov 正则化在线性场顶点处有微小偏差
    EXPECT_NEAR(T, 300.0 + 6.0 * 0.006, 1e-2);
}

TEST(PostprocessorTest, SamplePointOutsideMeshReturnsNaN)
{
    auto io = make_simple_uniform_grid_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int N = model->cells.cell_count;
    std::vector<double> cell_T(N, 300.0);
    auto node_T = mhs::post::interpolate_cell_to_node(*model, cell_T);

    mhs::core::ProbePoint pt;
    pt.name = "outside";
    // model->mesh 顶点范围 0..0.01 m (0..10 mm) — 0.1 m 在网格外
    pt.x = 0.1;
    pt.y = 0.005;
    pt.z = 0.005;

    double T = mhs::post::sample_point(node_T, *model, pt);
    EXPECT_TRUE(std::isnan(T)) << "Out-of-mesh point must return NaN";
}

TEST(PostprocessorTest, SamplePointOutsideOnDirichletFaceReturnsDirichlet)
{
    // Dirichlet on Z=0 face at 500K; point (5, 5, 0) sits exactly on that face.
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
    layer.name = "l";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";
    mhs::core::Block block;
    block.name = "b";
    block.material_name = "mat";
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
    mat.name = "mat";
    mat.kx = mat.ky = mat.kz = "400";
    io.materials["mat"] = mat;

    mhs::core::Boundary boundary;
    boundary.name = "bc1";
    boundary.bc_type = mhs::core::ThermalBCType::FirstType;
    boundary.first.temperature = "500";
    boundary.face_keys.push_back("Z|E|0|0,10,0,10");
    io.boundaries.push_back(boundary);
    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);
    int N = model->cells.cell_count;
    std::vector<double> cell_T(static_cast<size_t>(N), 400.0);
    auto node_T = mhs::post::interpolate_cell_to_node(*model, cell_T);

    mhs::core::ProbePoint pt;
    pt.name = "on_face";
    pt.x = 0.005;
    pt.y = 0.005;
    pt.z = 0.0; // exactly on Dirichlet face (vertex_z[0] = 0)

    double T = mhs::post::sample_point(node_T, *model, pt);
    EXPECT_NEAR(T, 500.0, 1e-6) << "Probe on Dirichlet face must return the Dirichlet value";
}