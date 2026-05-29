#include "expr/expr.hpp"
#include "io/io.hpp"
#include "model/internal_model.hpp"
#include "model/io_model.hpp"
#include "preprocessor/preprocessor.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace mhs;
using namespace mhs::model;

// Helper: build a minimal IOStructure for testing
static IOStructure make_simple_io()
{
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    // Simple 10x10x10 mm cube, 2 cells each direction
    io.mesh_vertex_x = {0.0, 5.0, 10.0};
    io.mesh_vertex_y = {0.0, 5.0, 10.0};
    io.mesh_vertex_z = {0.0, 5.0, 10.0};

    // One layer, one block covering the whole area
    Layer layer;
    layer.name = "test_layer";
    layer.is_top_layer = true;
    layer.thickness_expr = "10";

    Block block;
    block.name = "test_block";
    block.material_name = "test_material";
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

    // Material
    Material mat;
    mat.name = "test_material";
    mat.daore_xishu = "400";
    mat.midu = "8920";
    mat.bi_rerong = "385";
    io.materials["test_material"] = mat;

    // No explicit boundaries - default other_bc applies
    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    return io;
}

// ---- MeshGeometry Tests ----

TEST(PreprocessorTest, MeshGeometryFromVertices)
{
    auto io = make_simple_io();
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->mesh.nx, 2);
    EXPECT_EQ(model->mesh.ny, 2);
    EXPECT_EQ(model->mesh.nz, 2);
    EXPECT_EQ(model->mesh.total_cell_count, 8);

    // Check vertex arrays
    EXPECT_EQ(model->mesh.vertex_x.size(), 3);
    EXPECT_EQ(model->mesh.vertex_y.size(), 3);
    EXPECT_EQ(model->mesh.vertex_z.size(), 3);

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
    EXPECT_EQ(model->cells.cell_count, 8);
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

    // Material table should have one entry
    EXPECT_EQ(model->material_table.size(), 1);
    EXPECT_TRUE(model->material_table[0].k.is_constant());
    EXPECT_NEAR(model->material_table[0].k.constant_value(), 400.0, 1e-10);
}

TEST(PreprocessorTest, VirtualCellsFromSubRect)
{
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    // 100x100x30mm, with cells at x:0,50,100 y:0,50,100 z:0,2,4,6,8,10,15,20,25,30
    io.mesh_vertex_x = {0, 50, 100};
    io.mesh_vertex_y = {0, 50, 100};
    io.mesh_vertex_z = {0, 2, 4, 6, 8, 10, 15, 20, 25, 30};

    // Layer 1 (top): 2 blocks, with add/sub rects creating L-shape
    Layer layer1;
    layer1.name = "top";
    layer1.is_top_layer = true;
    layer1.thickness_expr = "20";

    // Block 1: L-shape via add rect (0,0,50,50) and (50,0,50,100)
    Block block1;
    block1.name = "b1";
    block1.material_name = "copper";
    block1.thickness_expr = "20";
    block1.ti_reyuan_expr = "0";
    block1.is_normal_material = true;

    Rect r1;
    r1.add_sub = true;
    r1.x_expr = "0";
    r1.y_expr = "0";
    r1.width_expr = "50";
    r1.height_expr = "50";
    block1.all_rects.push_back(r1);

    Rect r2;
    r2.add_sub = true;
    r2.x_expr = "50";
    r2.y_expr = "0";
    r2.width_expr = "50";
    r2.height_expr = "100";
    block1.all_rects.push_back(r2);

    layer1.blocks.push_back(block1);
    io.layers.push_back(layer1);

    // Layer 2 (substrate): silicon with sub-rect hole at (0,0,50,50)
    Layer layer2;
    layer2.name = "substrate";
    layer2.is_top_layer = false;
    layer2.thickness_expr = "10";

    Block block2;
    block2.name = "b2";
    block2.material_name = "silicon";
    block2.thickness_expr = "10";
    block2.ti_reyuan_expr = "0";
    block2.is_normal_material = true;

    Rect r3;
    r3.add_sub = true;
    r3.x_expr = "0";
    r3.y_expr = "0";
    r3.width_expr = "100";
    r3.height_expr = "100";
    block2.all_rects.push_back(r3);

    Rect r4;
    r4.add_sub = false; // subtract: removes (0,0,50,50) from the silicon
    r4.x_expr = "0";
    r4.y_expr = "0";
    r4.width_expr = "50";
    r4.height_expr = "50";
    block2.all_rects.push_back(r4);

    layer2.blocks.push_back(block2);
    io.layers.push_back(layer2);

    // Materials
    Material copper;
    copper.name = "copper";
    copper.daore_xishu = "400";
    io.materials["copper"] = copper;

    Material silicon;
    silicon.name = "silicon";
    silicon.daore_xishu = "130";
    io.materials["silicon"] = silicon;

    io.other_bc_type = ThermalBCType::SecondType;
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

    // Layer 1 is top layer, covering z=10..30mm (layers[0] = top)
    // Layer 2 is substrate, covering z=0..10mm (layers[1])
    // In layer1 (z indices 4-8, covering z=10..30mm in SI):
    //   Block1 has add rects: (0,0,50,50) and (50,0,50,100)
    //   Cell (ix=0, iy=0) at cx=25mm=25e-3 is in rect1 (0,0,50,50) -> valid, copper
    //   Cell (ix=1, iy=0) at cx=75mm=75e-3 is in rect2 (50,0,50,100) -> valid, copper
    //   Cell (ix=0, iy=1) at cy=75mm is NOT in rect1 (cy=75 >= 0+50=50) -> virtual
    //   Cell (ix=1, iy=1) at cx=75,cy=75 is in rect2 (50,0,50,100) -> valid, copper

    // In layer2 (z indices 0-4, covering z=0..10mm):
    //   Block has (0,0,100,100) - subtract (0,0,50,50)
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
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0, 50, 100};
    io.mesh_vertex_y = {0, 50, 100};
    io.mesh_vertex_z = {0, 10, 20, 30};

    Layer layer;
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "30";

    Block block;
    block.name = "b1";
    block.material_name = "copper";
    block.thickness_expr = "30";
    block.ti_reyuan_expr = "0";
    block.is_normal_material = true;

    Rect rect;
    rect.add_sub = true;
    rect.x_expr = "0";
    rect.y_expr = "0";
    rect.width_expr = "100";
    rect.height_expr = "100";
    block.all_rects.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    Material copper;
    copper.name = "copper";
    copper.daore_xishu = "400";
    io.materials["copper"] = copper;

    // Boundary: Dirichlet 500K on Z bottom face
    Boundary boundary;
    boundary.name = "bc1";
    boundary.bc_type = ThermalBCType::FirstType;
    boundary.first.temperature = "500";
    boundary.face_keys.push_back("Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100");
    io.boundaries.push_back(boundary);

    io.other_bc_type = ThermalBCType::SecondType;
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

    EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)FaceDir::ZM], BcType::FirstType);

    // BCParamTable should have dirichlet_T entries
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

    EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)FaceDir::XM], BcType::SecondType);
    EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)FaceDir::YM], BcType::SecondType);
    EXPECT_EQ(model->cells.cell_bcs[compact].types[(size_t)FaceDir::ZM], BcType::SecondType);

    // Interior cell (1,1,1): XP, YP, ZP are domain boundaries
    // XM, YM, ZM are interior -> None
    int idx_inner = 1 * ny * nz + 1 * nz + 1;
    int compact_inner = model->cells.index_map[idx_inner];
    ASSERT_NE(compact_inner, SIZE_MAX);

    EXPECT_EQ(model->cells.cell_bcs[compact_inner].types[(size_t)FaceDir::XM], BcType::None);
    EXPECT_EQ(model->cells.cell_bcs[compact_inner].types[(size_t)FaceDir::YM], BcType::None);
    EXPECT_EQ(model->cells.cell_bcs[compact_inner].types[(size_t)FaceDir::ZM], BcType::None);
    EXPECT_EQ(model->cells.cell_bcs[compact_inner].types[(size_t)FaceDir::XP], BcType::SecondType);
    EXPECT_EQ(model->cells.cell_bcs[compact_inner].types[(size_t)FaceDir::YP], BcType::SecondType);
    EXPECT_EQ(model->cells.cell_bcs[compact_inner].types[(size_t)FaceDir::ZP], BcType::SecondType);
}

// ---- Expression Compilation Tests ----

TEST(PreprocessorTest, HeatSourceCompilation)
{
    IOStructure io;
    io.study_type = StudyType::Steady;
    io.dimension = Dimension::Dimension3D;
    io.length_unit = LengthUnit::Mm;
    io.initial_temperature = 300.0;
    io.ambient_temperature = 300.0;

    io.mesh_vertex_x = {0, 50, 100};
    io.mesh_vertex_y = {0, 50, 100};
    io.mesh_vertex_z = {0, 15, 30};

    Layer layer;
    layer.name = "test";
    layer.is_top_layer = true;
    layer.thickness_expr = "30";

    Block block1;
    block1.name = "b1";
    block1.material_name = "copper";
    block1.thickness_expr = "30";
    block1.ti_reyuan_expr = "0";
    block1.is_normal_material = true;

    Rect rect1;
    rect1.add_sub = true;
    rect1.x_expr = "0";
    rect1.y_expr = "0";
    rect1.width_expr = "50";
    rect1.height_expr = "50";
    block1.all_rects.push_back(rect1);

    Block block2;
    block2.name = "b2";
    block2.material_name = "copper";
    block2.thickness_expr = "30";
    block2.ti_reyuan_expr = "1e8"; // heat source
    block2.is_normal_material = true;

    Rect rect2;
    rect2.add_sub = true;
    rect2.x_expr = "50";
    rect2.y_expr = "0";
    rect2.width_expr = "50";
    rect2.height_expr = "50";
    block2.all_rects.push_back(rect2);

    layer.blocks.push_back(block1);
    layer.blocks.push_back(block2);
    io.layers.push_back(layer);

    Material copper;
    copper.name = "copper";
    copper.daore_xishu = "400";
    io.materials["copper"] = copper;

    io.other_bc_type = ThermalBCType::SecondType;
    io.other_bc_second.heat_flux = "0";

    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    // Cell at (0,0,*) belongs to block1 -> heat source = 0
    // Cell at (1,0,*) belongs to block2 -> heat source = 1e8

    int idx_block1 = 0 * 2 * 2 + 0 * 2 + 0;
    int compact1 = model->cells.index_map[idx_block1];
    EXPECT_TRUE(model->cells.heat_source[compact1].is_constant());
    EXPECT_NEAR(model->cells.heat_source[compact1].constant_value(), 0.0, 1e-10);

    int idx_block2 = 1 * 2 * 2 + 0 * 2 + 0;
    int compact2 = model->cells.index_map[idx_block2];
    EXPECT_TRUE(model->cells.heat_source[compact2].is_constant());
    EXPECT_NEAR(model->cells.heat_source[compact2].constant_value(), 1e8, 1e-6);
}

// ---- Full Case1 Integration Test ----

TEST(PreprocessorTest, Case1XMLLoad)
{
    std::string case_path = "cases/original_steady_tests/case1.xml";
    if (!std::filesystem::exists(case_path)) {
        GTEST_SKIP() << "Case1 XML not found at " << case_path;
    }

    auto io = io::read_xml(case_path);
    Preprocessor preprocessor;
    auto model = preprocessor.load(io);
    ASSERT_NE(model, nullptr);

    // Case1: X=[0,10,...,100], Y=[0,10,...,100], Z=[0,2,4,6,8,10,15,20,25,30]
    EXPECT_EQ(model->mesh.nx, 10);
    EXPECT_EQ(model->mesh.ny, 10);
    EXPECT_EQ(model->mesh.nz, 9);
    EXPECT_EQ(model->mesh.total_cell_count, 900);

    // Should have some valid and some virtual cells
    EXPECT_GT(model->cells.cell_count, 0);
    EXPECT_LT(model->cells.cell_count, 900);

    // Material table should have copper and silicon
    EXPECT_EQ(model->material_table.size(), 2);
}