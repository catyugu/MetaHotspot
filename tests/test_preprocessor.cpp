#include "common/internal_model.hpp"
#include "common/io_model.hpp"
#include "config.h"
#include "expr/expr.hpp"
#include "io/io.hpp"
#include "preprocessor/face_key_processor.hpp"
#include "preprocessor/preprocessor.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace mhs::core;
using namespace mhs::sim;
using namespace mhs::io;

// Helper: build a minimal mhs::core::IOStructure for testing
static mhs::core::IOStructure make_simple_io()
{
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    // Simple 10x10x10 mm cube, 2 cells each direction
    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    // One layer, one block covering the whole area
    mhs::core::Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    mhs::core::Block block;
    block.name = "test_block";
    block.material_name = "test_material";
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

    // mhs::core::Material
    mhs::core::Material mat;
    mat.name = "test_material";
    mat.kx = mat.ky = mat.kz = "400";
    mat.midu = "8920";
    mat.bi_rerong = "385";
    io.materials["test_material"] = mat;

    // No explicit boundaries - default other_bc applies
    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    return io;
}

// ---- mhs::core::MeshGeometry Tests ----

TEST(PreprocessorTest, MeshGeometryFromVertices)
{
    auto io = make_simple_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->mesh.nx, 2);
    EXPECT_EQ(model->mesh.ny, 2);
    EXPECT_EQ(model->mesh.nz, 2);

    // Check cell sizes (dx, dy, dz)
    EXPECT_EQ(model->mesh.dx.size(), 2);
    EXPECT_NEAR(model->mesh.dx[0], 5.0e-3, 1e-10); // 5mm -> 0.005m (SI)
    EXPECT_NEAR(model->mesh.dx[1], 5.0e-3, 1e-10);

    EXPECT_EQ(model->mesh.dy.size(), 2);
    EXPECT_NEAR(model->mesh.dy[0], 5.0e-3, 1e-10);
    EXPECT_NEAR(model->mesh.dy[1], 5.0e-3, 1e-10);

    EXPECT_EQ(model->mesh.dz.size(), 2);
    EXPECT_NEAR(model->mesh.dz[0], 5.0e-3, 1e-10);
    EXPECT_NEAR(model->mesh.dz[1], 5.0e-3, 1e-10);

    // Check cell centers (cx, cy, cz)
    EXPECT_NEAR(model->mesh.cx[0], 2.5e-3, 1e-10); // center of [0, 5mm] in m
    EXPECT_NEAR(model->mesh.cx[1], 7.5e-3, 1e-10);

    EXPECT_NEAR(model->mesh.cy[0], 2.5e-3, 1e-10);
    EXPECT_NEAR(model->mesh.cy[1], 7.5e-3, 1e-10);

    EXPECT_NEAR(model->mesh.cz[0], 2.5e-3, 1e-10);
    EXPECT_NEAR(model->mesh.cz[1], 7.5e-3, 1e-10);
}

// ---- Virtual Cell / LayerProcessor Tests ----

TEST(PreprocessorTest, AllCellsValidWhenSingleFullBlock)
{
    auto io = make_simple_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    // All 8 cells should be valid
    EXPECT_EQ(model->cells.cell_bcs.size(), 8u);
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(model->cells.valid_mask[i], 1);
        EXPECT_NE(model->cells.index_map[i], SIZE_MAX);
    }
}

TEST(PreprocessorTest, MaterialAssignment)
{
    auto io = make_simple_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    // All cells should have material_id = 0 (first material)
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(model->cells.material_id[i], 0);
    }

    // mhs::core::Material table should have one entry
    EXPECT_EQ(model->material_table.size(), 1);
    EXPECT_TRUE(model->material_table[0].kx.is_constant());
    EXPECT_NEAR(model->material_table[0].kx.constant_value(), 400.0, 1e-10);
}

TEST(PreprocessorTest, VirtualCellsFromSubRect)
{
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    // 100x100x30mm, with cells at x:0,50,100 y:0,50,100 z:0,2,4,6,8,10,15,20,25,30
    io.mesh_vertex_x = {0, 50, 100};
    io.mesh_vertex_y = {0, 50, 100};
    io.mesh_vertex_z = {0, 2, 4, 6, 8, 10, 15, 20, 25, 30};

    // mhs::core::Layer 1 (top): 2 blocks, with add/sub rects creating L-shape
    mhs::core::Layer layer1;
    layer1.name = "top";
    layer1.is_top_layer = true;
    layer1.thickness_expr = "20";

    // mhs::core::Block 1: L-shape via add rect (0,0,50,50) and (50,0,50,100)
    mhs::core::Block block1;
    block1.name = "b1";
    block1.material_name = "copper";
    block1.ti_reyuan_expr = "0";
    block1.is_normal_material = true;

    mhs::core::Rect r1;
    r1.add_sub = true;
    r1.x_expr = "0";
    r1.y_expr = "0";
    r1.width_expr = "50";
    r1.height_expr = "50";
    block1.all_rects.push_back(r1);

    mhs::core::Rect r2;
    r2.add_sub = true;
    r2.x_expr = "50";
    r2.y_expr = "0";
    r2.width_expr = "50";
    r2.height_expr = "100";
    block1.all_rects.push_back(r2);

    layer1.blocks.push_back(block1);
    io.layers.push_back(layer1);

    // mhs::core::Layer 2 (substrate): silicon with sub-rect hole at (0,0,50,50)
    mhs::core::Layer layer2;
    layer2.name = "substrate";
    layer2.is_top_layer = false;
    layer2.thickness_expr = "10";

    mhs::core::Block block2;
    block2.name = "b2";
    block2.material_name = "silicon";
    block2.ti_reyuan_expr = "0";
    block2.is_normal_material = true;

    mhs::core::Rect r3;
    r3.add_sub = true;
    r3.x_expr = "0";
    r3.y_expr = "0";
    r3.width_expr = "100";
    r3.height_expr = "100";
    block2.all_rects.push_back(r3);

    mhs::core::Rect r4;
    r4.add_sub = false; // subtract: removes (0,0,50,50) from the silicon
    r4.x_expr = "0";
    r4.y_expr = "0";
    r4.width_expr = "50";
    r4.height_expr = "50";
    block2.all_rects.push_back(r4);

    layer2.blocks.push_back(block2);
    io.layers.push_back(layer2);

    // Materials
    mhs::core::Material copper;
    copper.name = "copper";
    copper.kx = copper.ky = copper.kz = "400";
    io.materials["copper"] = copper;

    mhs::core::Material silicon;
    silicon.name = "silicon";
    silicon.kx = silicon.ky = silicon.kz = "130";
    io.materials["silicon"] = silicon;

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int nx = model->mesh.nx;
    int ny = model->mesh.ny;
    int nz = model->mesh.nz;
    EXPECT_EQ(nx, 2);
    EXPECT_EQ(ny, 2);
    EXPECT_EQ(nz, 9);

    // mhs::core::Layer 1 is top layer, covering z=10..30mm (layers[0] = top)
    // mhs::core::Layer 2 is substrate, covering z=0..10mm (layers[1])
    // In layer1 (z indices 4-8, covering z=10..30mm in SI):
    //   Block1 has add rects: (0,0,50,50) and (50,0,50,100)
    //   Cell (ix=0, iy=0) at cx=25mm=25e-3 is in rect1 (0,0,50,50) -> valid, copper
    //   Cell (ix=1, iy=0) at cx=75mm=75e-3 is in rect2 (50,0,50,100) -> valid, copper
    //   Cell (ix=0, iy=1) at cy=75mm is NOT in rect1 (cy=75 >= 0+50=50) -> virtual
    //   Cell (ix=1, iy=1) at cx=75,cy=75 is in rect2 (50,0,50,100) -> valid, copper

    // In layer2 (z indices 0-4, covering z=0..10mm):
    //   mhs::core::Block has (0,0,100,100) - subtract (0,0,50,50)
    //   Cell (ix=0, iy=0) -> subtracted -> virtual
    //   Cell (ix=1, iy=0) -> valid, silicon
    //   Cell (ix=0, iy=1) -> valid, silicon
    //   Cell (ix=1, iy=1) -> valid, silicon

    // Check cell (ix=0, iy=1, iz=0) -> layer2, not subtracted, in (0,0,100,100) -> valid
    // Actually cell at cy=75e-3 is NOT in sub-rect (0,0,50,50), so it IS valid
    int idx_01_0 = 0 * ny * nz + 1 * nz + 0;
    EXPECT_EQ(model->cells.valid_mask[idx_01_0], 1); // valid in layer2
    EXPECT_NE(model->cells.index_map[idx_01_0], SIZE_MAX);

    // Check cell (ix=0, iy=0, iz=5) -> layer0 (top), (ix=0, iy=0) cx=25mm, cy=25mm in rect1 -> valid
    // iz=5 means cz=12.5mm, which is in top layer (z=10..30mm)
    int idx_00_5 = 0 * ny * nz + 0 * nz + 5;
    EXPECT_EQ(model->cells.valid_mask[idx_00_5], 1);

    // Cell (ix=0, iy=0, iz=4) -> layer1 (substrate), cx=25mm cy=25mm -> subtracted -> virtual
    // iz=4 means cz=9mm, in substrate layer (z=0..10mm)
    int idx_00_4 = 0 * ny * nz + 0 * nz + 4;
    EXPECT_EQ(model->cells.valid_mask[idx_00_4], 0);
}

// ---- FaceKey / BC Tests ----

TEST(PreprocessorTest, FaceKeyParsing_ZE_Dirichlet)
{
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0, 50, 100};
    io.mesh_vertex_y = {0, 50, 100};
    io.mesh_vertex_z = {0, 10, 20, 30};

    mhs::core::Layer layer;
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "30";

    mhs::core::Block block;
    block.name = "b1";
    block.material_name = "copper";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    mhs::core::Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "100";
    rect.height_expr = "100";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::core::Material copper;
    copper.name = "copper";
    copper.kx = copper.ky = copper.kz = "400";
    io.materials["copper"] = copper;

    // mhs::core::Boundary: Dirichlet 500K on Z bottom face
    mhs::core::Boundary boundary;
    boundary.name = "bc1";
    boundary.bc_type = mhs::core::ThermalBCType::FirstType;
    boundary.first.temperature = "500";
    boundary.face_keys.push_back("Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100");
    io.boundaries.push_back(boundary);

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    // Face key "Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100"
    // Rects: {x:0-50, y:50-100}, {x:50-100, y:0-50}, {x:50-100, y:50-100}
    // Cell (ix=0, iy=1, iz=0) has cx=25mm, cy=75mm -> in rect1 -> FirstType on ZM
    int ny_bc = model->mesh.ny;
    int nz_bc = model->mesh.nz;
    int idx_bc = 0 * ny_bc * nz_bc + 1 * nz_bc + 0;
    int compact = model->cells.index_map[idx_bc];
    ASSERT_NE(compact, SIZE_MAX);

    EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)mhs::core::FaceDir::ZM], mhs::core::BcType::FirstType);

    // mhs::core::BCParamTable should have dirichlet_T entries
    EXPECT_FALSE(model->bc_params.dirichlet_T.empty());
    EXPECT_TRUE(model->bc_params.dirichlet_T[0].is_constant());
    EXPECT_NEAR(model->bc_params.dirichlet_T[0].constant_value(), 500.0, 1e-10);
}

TEST(PreprocessorTest, OtherBCFallback)
{
    auto io = make_simple_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    // With no explicit boundaries and other_bc=SecondType(0),
    // all faces on domain boundaries should have SecondType BC
    // Interior faces should have None BC

    // Cell (0,0,0) - bottom-left-front cell:
    // XM (x=0 face): domain boundary -> SecondType
    // YM (y=0 face): domain boundary -> SecondType
    // ZM (z=0 face): domain boundary -> SecondType
    // XP, YP, ZP: interior or domain boundary
    int ny = model->mesh.ny;
    int nz = model->mesh.nz;
    int idx = 0 * ny * nz + 0 * nz + 0;
    int compact = model->cells.index_map[idx];
    ASSERT_NE(compact, SIZE_MAX);

    EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)mhs::core::FaceDir::XM], mhs::core::BcType::SecondType);
    EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)mhs::core::FaceDir::YM], mhs::core::BcType::SecondType);
    EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)mhs::core::FaceDir::ZM], mhs::core::BcType::SecondType);

    // Interior cell (1,1,1): XP, YP, ZP are domain boundaries
    // XM, YM, ZM are interior -> None
    int idx_inner = 1 * ny * nz + 1 * nz + 1;
    int compact_inner = model->cells.index_map[idx_inner];
    ASSERT_NE(compact_inner, SIZE_MAX);

    EXPECT_EQ(model->cells.cell_bcs[compact_inner].types[(size_t)mhs::core::FaceDir::XM], mhs::core::BcType::None);
    EXPECT_EQ(model->cells.cell_bcs[compact_inner].types[(size_t)mhs::core::FaceDir::YM], mhs::core::BcType::None);
    EXPECT_EQ(model->cells.cell_bcs[compact_inner].types[(size_t)mhs::core::FaceDir::ZM], mhs::core::BcType::None);
    EXPECT_EQ(
        model->cells.cell_bcs[compact_inner].types[(size_t)mhs::core::FaceDir::XP], mhs::core::BcType::SecondType);
    EXPECT_EQ(
        model->cells.cell_bcs[compact_inner].types[(size_t)mhs::core::FaceDir::YP], mhs::core::BcType::SecondType);
    EXPECT_EQ(
        model->cells.cell_bcs[compact_inner].types[(size_t)mhs::core::FaceDir::ZP], mhs::core::BcType::SecondType);
}

// ---- Full Case1 Integration Test ----

TEST(PreprocessorTest, Case1XMLLoad)
{
    std::string case_path = std::string(PROJECT_SOURCE_DIR) + "/cases/simple_steady_tests/case1.xml";
    if (!std::filesystem::exists(case_path)) {
        GTEST_SKIP() << "Case1 XML not found at " << case_path;
    }

    auto io_data = mhs::io::read_xml(case_path);
    Preprocessor preprocessor;
    auto model = preprocessor.load(io_data);
    ASSERT_NE(model, nullptr);

    // Case1: X=[0,10,...,100], Y=[0,10,...,100], Z=[0,2,4,6,8,10,15,20,25,30]
    EXPECT_EQ(model->mesh.nx, 20);
    EXPECT_EQ(model->mesh.ny, 20);
    EXPECT_EQ(model->mesh.nz, 9);
    // Should have some valid and some virtual cells
    EXPECT_GT(model->cells.cell_bcs.size(), 0u);
    EXPECT_LT(model->cells.cell_bcs.size(), 3600u);

    // mhs::core::Material table should have copper and silicon
    EXPECT_EQ(model->material_table.size(), 2);
}
// ---- Epsilon Tolerance Tests for find_block_for_cell ----

TEST(PreprocessorTest, CellsOnExactBoundaryEdgeAreNotMisclassified)
{
    // This test verifies that cells whose centers fall exactly on
    // a block rect boundary edge (due to floating-point alignment)
    // are correctly classified as valid, not incorrectly excluded.
    //
    // Setup: 100x100x30mm domain, 2x2x9 cells.
    // mhs::core::Block rect: x=25, width=50, y=0, height=100 -> covers [25mm, 75mm] in X
    // Cell ix=0: cx=25mm -> exactly at rx=25mm (lower bound inclusive)
    // Cell ix=1: cx=75mm -> exactly at rx+rw=75mm (upper bound must be inclusive with epsilon)
    // Without epsilon tolerance, strict `<` on upper bound excludes ix=1.

    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0, 50, 100};
    io.mesh_vertex_y = {0, 50, 100};
    io.mesh_vertex_z = {0, 10, 20, 30};

    mhs::core::Layer layer;
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "30";

    mhs::core::Block block;
    block.name = "b1";
    block.material_name = "copper";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    mhs::core::Rect rect;
    rect.add_sub = true;
    rect.x_expr = "25";
    rect.y_expr = "0";
    rect.width_expr = "50";
    rect.height_expr = "100";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::core::Material copper;
    copper.name = "copper";
    copper.kx = copper.ky = copper.kz = "400";
    io.materials["copper"] = copper;

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int ny = model->mesh.ny;
    int nz = model->mesh.nz;

    // Cell (ix=0, iy=0, iz=0): cx=25mm >= rx=25mm, should be valid
    int idx0 = 0 * ny * nz + 0 * nz + 0;
    EXPECT_EQ(model->cells.valid_mask[idx0], 1);

    // Cell (ix=1, iy=0, iz=0): cx=75mm exactly equals rx+rw=75mm
    // Without epsilon tolerance, this cell is incorrectly classified as virtual
    int idx1 = 1 * ny * nz + 0 * nz + 0;
    EXPECT_EQ(model->cells.valid_mask[idx1], 1);
}

TEST(PreprocessorTest, LaterBlockOverridesEarlierBlockInOverlap)
{
    // In CAD semantics, later blocks override earlier blocks in overlapping
    // regions. A chip (block2, silicon) overlaying a substrate (block1, copper)
    // should assign silicon material to cells in the overlap area.
    //
    // Before the fix: first-match logic gives block1 (copper) to overlap cells.
    // After the fix: last-match logic gives block2 (silicon) to overlap cells.

    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0, 50, 100};
    io.mesh_vertex_y = {0, 50, 100};
    io.mesh_vertex_z = {0, 10, 20, 30};

    mhs::core::Layer layer;
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "30";

    // mhs::core::Block 1: background substrate covering entire 100x100mm area (copper)
    mhs::core::Block block1;
    block1.name = "substrate";
    block1.material_name = "copper";
    block1.ti_reyuan_expr = "0";
    block1.is_normal_material = true;

    mhs::core::Rect rect1;
    rect1.add_sub = true;
    rect1.x_expr = "0";
    rect1.y_expr = "0";
    rect1.width_expr = "100";
    rect1.height_expr = "100";
    block1.all_rects.push_back(rect1);

    // mhs::core::Block 2: chip overlaying the first quadrant (0-50, 0-50) (silicon)
    mhs::core::Block block2;
    block2.name = "chip";
    block2.material_name = "silicon";
    block2.ti_reyuan_expr = "1e7";
    block2.is_normal_material = true;

    mhs::core::Rect rect2;
    rect2.add_sub = true;
    rect2.x_expr = "0";
    rect2.y_expr = "0";
    rect2.width_expr = "50";
    rect2.height_expr = "50";
    block2.all_rects.push_back(rect2);

    layer.blocks.push_back(block1);
    layer.blocks.push_back(block2);
    io.layers.push_back(layer);

    mhs::core::Material copper;
    copper.name = "copper";
    copper.kx = copper.ky = copper.kz = "400";
    io.materials["copper"] = copper;

    mhs::core::Material silicon;
    silicon.name = "silicon";
    silicon.kx = silicon.ky = silicon.kz = "130";
    io.materials["silicon"] = silicon;

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int ny = model->mesh.ny;
    int nz = model->mesh.nz;

    // Cell (ix=0, iy=0, iz=0): cx=25mm, cy=25mm — in overlap of both blocks.
    // Last block (block2 = silicon) should override first block (block1 = copper).
    int idx_overlap = 0 * ny * nz + 0 * nz + 0;
    EXPECT_EQ(model->cells.valid_mask[idx_overlap], 1);

    // mhs::core::Material should be silicon (block2), not copper (block1)
    // name_to_idx order: "copper" = 0, "silicon" = 1
    EXPECT_EQ(model->cells.material_id[idx_overlap], 1)
        << "Overlapping cell must get material from later block (silicon), not earlier (copper)";

    // Cell (ix=1, iy=0, iz=0): cx=75mm, cy=25mm — only in block1 (copper)
    int idx_only_block1 = 1 * ny * nz + 0 * nz + 0;
    EXPECT_EQ(model->cells.material_id[idx_only_block1], 0) << "Cell in only block1 must get copper material";
}

TEST(PreprocessorTest, ParseFaceKey_XFormatSevenParts)
{
    // case1 mhs::core::Boundary 5 (convection on X faces of the top die)
    mhs::sim::FaceKeyInfo fk = mhs::sim::parse_face_key("X|E|5|-7.5|7.5|26|29", 1.0);
    EXPECT_EQ(fk.axis, 'X');
    EXPECT_EQ(fk.side, 'E');
    EXPECT_NEAR(fk.coord_value, 5.0, 1e-12);
    ASSERT_EQ(fk.rects.size(), 1u);
    EXPECT_NEAR(fk.rects[0][0], -7.5, 1e-12);
    EXPECT_NEAR(fk.rects[0][1], 7.5, 1e-12);
    EXPECT_NEAR(fk.rects[0][2], 26.0, 1e-12);
    EXPECT_NEAR(fk.rects[0][3], 29.0, 1e-12);
}

TEST(PreprocessorTest, ResolveFaceKeys_AssignsYFormatToYPBoundary)
{
    // End-to-end: feed a Y-format face key, assert the correct Y+ face
    // of a cell adjacent to vertex_y = 7.5mm gets the assigned BC.
    //
    // Mesh: 2x2 cells over [0, 10]x[0, 10]x[0, 10] mm; one block filling all.
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0, 5, 10};
    io.mesh_vertex_y = {0, 5, 10};
    io.mesh_vertex_z = {0, 5, 10};

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

    // Convection on the upper half of the y=10mm Y+ face
    mhs::core::Boundary boundary;
    boundary.name = "bc_yp";
    boundary.bc_type = mhs::core::ThermalBCType::ThirdType;
    boundary.third.convection_coeff = "10";
    boundary.third.T_inf = "200";
    boundary.face_keys.push_back("Y|E|10|0|10|0|5"); // cx: 0-10, cz: 0-5
    io.boundaries.push_back(boundary);

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    // Cells adjacent to the Y+ face at y=10mm have iy=1 (vertex_y[2] = 10).
    // Their YP face is exposed. mhs::core::Rect: cx in [0,10], cz in [0,5].
    // Cells with iz=0 (cz=2.5) are inside the rect and get ThirdType.
    // Cells with iz=1 (cz=7.5) are outside the rect and fall through to
    // the other_bc (SecondType).
    int ny = model->mesh.ny;
    int nz = model->mesh.nz;
    for (int ix = 0; ix < 2; ++ix) {
        int idx = ix * ny * nz + 1 * nz + 0; // iz=0 -> cz=2.5 in [0,5]
        int compact = (int)model->cells.index_map[idx];
        EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)mhs::core::FaceDir::YP], mhs::core::BcType::ThirdType)
            << "Y-format face key should assign ThirdType to YP face at (" << ix << ",1,0)";
    }

    // Cells with iz=1 (cz=7.5) fall outside the rect and must NOT get this BC;
    // they should fall through to the other_bc (SecondType).
    int idx_outside = 0 * ny * nz + 1 * nz + 1; // ix=0, iy=1, iz=1 -> cz=7.5
    int compact_out = (int)model->cells.index_map[idx_outside];
    EXPECT_NE(model->cells.cell_bcs[compact_out].types[(size_t)mhs::core::FaceDir::YP], mhs::core::BcType::ThirdType)
        << "Cell outside rect must not get the ThirdType BC";
}

TEST(PreprocessorTest, ResolveFaceKeys_MultipleFaceKeysInOneBoundary)
{
    // A single boundary carrying many face_keys must apply each one.
    // Mirrors case1 mhs::core::Boundary 5: 4 X-keys + 2 Y-keys covering the side faces
    // of the top die. Without correct per-key iteration, some faces would
    // silently fall through to other_bc.
    mhs::core::IOStructure io;
    io.study_type = mhs::core::StudyType::Steady;
    io.dimension = mhs::core::Dimension::Dimension3D;
    io.length_unit = mhs::core::LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0, 5, 10};
    io.mesh_vertex_y = {0, 5, 10};
    io.mesh_vertex_z = {0, 5, 10};

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

    // One boundary, three face_keys targeting three different exposed faces
    mhs::core::Boundary boundary;
    boundary.name = "bc_multi";
    boundary.bc_type = mhs::core::ThermalBCType::ThirdType;
    boundary.third.convection_coeff = "10";
    boundary.third.T_inf = "200";
    boundary.face_keys.push_back("X|E|10|0|10|0|10"); // XP face, all cz
    boundary.face_keys.push_back("X|E|0|0|10|0|10"); // XM face, all cz
    boundary.face_keys.push_back("Y|E|10|0|10|0|10"); // YP face, all cz
    io.boundaries.push_back(boundary);

    io.other_bc_type = mhs::core::ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    int ny = model->mesh.ny;
    int nz = model->mesh.nz;
    int nx = model->mesh.nx;

    // Every exposed face of every cell on the domain boundary (ix=0 XP/XP, etc.)
    // and specifically the three faces the keys target must be ThirdType.
    auto check_face = [&](int ix, int iy, int iz, mhs::core::FaceDir dir) {
        int idx = ix * ny * nz + iy * nz + iz;
        int compact = (int)model->cells.index_map[idx];
        EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)dir], mhs::core::BcType::ThirdType)
            << "Face " << (int)dir << " of cell (" << ix << "," << iy << "," << iz
            << ") should be ThirdType from one of the boundary face_keys";
    };

    // X|E|10 -> XP face at x=10mm -> cells with ix=nx-1=1
    for (int iy = 0; iy < ny; ++iy)
        for (int iz = 0; iz < nz; ++iz)
            check_face(nx - 1, iy, iz, mhs::core::FaceDir::XP);

    // X|E|0 -> XM face at x=0mm -> cells with ix=0
    for (int iy = 0; iy < ny; ++iy)
        for (int iz = 0; iz < nz; ++iz)
            check_face(0, iy, iz, mhs::core::FaceDir::XM);

    // Y|E|10 -> YP face at y=10mm -> cells with iy=ny-1=1
    for (int ix = 0; ix < nx; ++ix)
        for (int iz = 0; iz < nz; ++iz)
            check_face(ix, ny - 1, iz, mhs::core::FaceDir::YP);
}
