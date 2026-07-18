#include "data/model.hpp"
#include "model/model_definition.hpp"
#include "model_test_utils.hpp"
#include "expr/expr.hpp"
#include "preprocessor/preprocessor.hpp"
#include <gtest/gtest.h>

using namespace mhs::core;
using namespace mhs::sim;

// Helpers for testing patch-based BC assignment.

/// Find a FaceBC on `cell_idx` with direction `dir`.
/// Returns pointer or nullptr if that face has BcType::None.
static const mhs::core::FaceBC* find_face_bc(const Model& model, uint32_t cell_idx, FaceDir dir)
{
    auto& fb = model.face_bcs[cell_idx * FACE_COUNT + (size_t)dir];
    return fb.type != BcType::None ? &fb : nullptr;
}

/// Return the BcType on cell_idx for direction dir, or None if not found.
static BcType get_bc_type(const Model& model, uint32_t cell_idx, FaceDir dir)
{
    auto* p = find_face_bc(model, cell_idx, dir);
    return p ? p->type : BcType::None;
}

/// Return the param_idx on cell_idx for direction dir, or 0 if not found.
static uint16_t get_bc_param(const Model& model, uint32_t cell_idx, FaceDir dir)
{
    auto* p = find_face_bc(model, cell_idx, dir);
    return p ? p->param_idx : 0;
}

// Helper: build a minimal mhs::model::ModelDefinition for testing
static mhs::model::ModelDefinition make_simple_io()
{
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    // Simple 10x10x10 mm cube, 2 cells each direction
    io.mesh.x_vertices = {0.0, 5.0, 10.0};
    io.mesh.y_vertices = {0.0, 5.0, 10.0};
    io.mesh.z_vertices = {0.0, 5.0, 10.0};

    // One layer, one block covering the whole area
    mhs::model::LayerSpec layer;
    layer.thickness = "10";

    mhs::model::BlockSpec block;
    block.material = "test_material";
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

    // mhs::model::MaterialSpec
    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "400";
    mat.density = "8920";
    mat.specific_heat = "385";
    io.materials.push_back({"test_material", mat});

    // No explicit boundaries: the default boundary applies.
    io.default_boundary = mhs::model::NeumannBoundary {};

    return io;
}

// ---- mhs::core::MeshGeometry Tests ----

TEST(PreprocessorTest, MeshGeometryFromVertices)
{
    auto io = make_simple_io();
    auto model = build_model(io);

    EXPECT_EQ(model.mesh.nx, 2);
    EXPECT_EQ(model.mesh.ny, 2);
    EXPECT_EQ(model.mesh.nz, 2);

    // Check cell sizes (dx, dy, dz)
    EXPECT_EQ(model.mesh.dx.size(), 2);
    EXPECT_NEAR(model.mesh.dx[0], 5.0e-3, 1e-10); // 5mm -> 0.005m (SI)
    EXPECT_NEAR(model.mesh.dx[1], 5.0e-3, 1e-10);

    EXPECT_EQ(model.mesh.dy.size(), 2);
    EXPECT_NEAR(model.mesh.dy[0], 5.0e-3, 1e-10);
    EXPECT_NEAR(model.mesh.dy[1], 5.0e-3, 1e-10);

    EXPECT_EQ(model.mesh.dz.size(), 2);
    EXPECT_NEAR(model.mesh.dz[0], 5.0e-3, 1e-10);
    EXPECT_NEAR(model.mesh.dz[1], 5.0e-3, 1e-10);

    // Check cell centers (cx, cy, cz)
    EXPECT_NEAR(model.mesh.cx[0], 2.5e-3, 1e-10); // center of [0, 5mm] in m
    EXPECT_NEAR(model.mesh.cx[1], 7.5e-3, 1e-10);

    EXPECT_NEAR(model.mesh.cy[0], 2.5e-3, 1e-10);
    EXPECT_NEAR(model.mesh.cy[1], 7.5e-3, 1e-10);

    EXPECT_NEAR(model.mesh.cz[0], 2.5e-3, 1e-10);
    EXPECT_NEAR(model.mesh.cz[1], 7.5e-3, 1e-10);
}

// ---- Virtual Cell / LayerProcessor Tests ----

TEST(PreprocessorTest, CellMappingsAreExactInverses)
{
    auto definition = make_simple_io();

    mhs::model::RectOperation cutout;
    cutout.operation = mhs::model::GeometryOperation::Subtract;
    cutout.rect.x = "0";
    cutout.rect.y = "0";
    cutout.rect.width = "5";
    cutout.rect.height = "5";
    definition.layers[0].blocks[0].geometry.push_back(cutout);

    const auto model = build_model(definition);
    const auto& cells = model.cells;

    EXPECT_EQ(cells.grid_to_cell.size(), 8u);
    EXPECT_EQ(cells.cell_to_grid.size(), 6u);
    EXPECT_EQ(cells.cell_to_grid.size(), cells.material_id.size());

    std::size_t active_count = 0;
    for (mhs::Index grid = 0; grid < cells.grid_to_cell.size(); ++grid) {
        const mhs::Index cell = cells.grid_to_cell[grid];
        if (cell == mhs::invalidIndex)
            continue;

        ++active_count;
        ASSERT_LT(cell, cells.cell_to_grid.size());
        EXPECT_EQ(cells.cell_to_grid[cell], grid);
    }

    EXPECT_EQ(active_count, cells.cell_to_grid.size());
    for (mhs::Index cell = 0; cell < cells.cell_to_grid.size(); ++cell) {
        const mhs::Index grid = cells.cell_to_grid[cell];
        ASSERT_LT(grid, cells.grid_to_cell.size());
        EXPECT_EQ(cells.grid_to_cell[grid], cell);
    }
}

TEST(PreprocessorTest, RectOperationsFollowAppendOrder)
{
    auto definition = make_simple_io();
    definition.mesh.x_vertices = {0.0, 5.0, 10.0, 15.0};
    definition.mesh.y_vertices = {0.0, 5.0};
    definition.mesh.z_vertices = {0.0, 5.0};
    definition.layers[0].thickness = "5";

    auto& rects = definition.layers[0].blocks[0].geometry;
    rects[0].rect.width = "15";
    rects[0].rect.height = "5";

    mhs::model::RectOperation subtract;
    subtract.operation = mhs::model::GeometryOperation::Subtract;
    subtract.rect.x = "5";
    subtract.rect.y = "0";
    subtract.rect.width = "10";
    subtract.rect.height = "5";
    rects.push_back(subtract);

    mhs::model::RectOperation add_back;
    add_back.operation = mhs::model::GeometryOperation::Add;
    add_back.rect.x = "10";
    add_back.rect.y = "0";
    add_back.rect.width = "5";
    add_back.rect.height = "5";
    rects.push_back(add_back);

    const auto model = build_model(definition);

    EXPECT_NE(model.cells.grid_to_cell[0], mhs::invalidIndex) << "The initial Add must keep the first cell";
    EXPECT_EQ(model.cells.grid_to_cell[1], mhs::invalidIndex) << "The later Subtract must remove the middle cell";
    EXPECT_NE(model.cells.grid_to_cell[2], mhs::invalidIndex) << "The final Add must restore the last cell";
}

TEST(PreprocessorTest, MaterialAssignment)
{
    auto io = make_simple_io();
    auto model = build_model(io);

    // All cells should have material_id = 0 (first material)
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(model.cells.material_id[i], 0);
    }

    // mhs::model::MaterialSpec table should have one entry
    EXPECT_EQ(model.material_table.size(), 1);
    EXPECT_TRUE(model.material_table[0].kx.is_constant());
    EXPECT_NEAR(model.material_table[0].kx.constant_value(), 400.0, 1e-10);
}

TEST(PreprocessorTest, VirtualCellsFromSubRect)
{
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    // 100x100x30mm, with cells at x:0,50,100 y:0,50,100 z:0,2,4,6,8,10,15,20,25,30
    io.mesh.x_vertices = {0, 50, 100};
    io.mesh.y_vertices = {0, 50, 100};
    io.mesh.z_vertices = {0, 2, 4, 6, 8, 10, 15, 20, 25, 30};

    // mhs::model::LayerSpec 1 (top): 2 blocks, with add/sub rects creating L-shape
    mhs::model::LayerSpec layer1;
    layer1.thickness = "20";

    // mhs::model::BlockSpec 1: L-shape via add rect (0,0,50,50) and (50,0,50,100)
    mhs::model::BlockSpec block1;
    block1.material = "copper";
    block1.volumetric_heat_source = "0";

    mhs::model::RectOperation r1;
    r1.operation = mhs::model::GeometryOperation::Add;
    r1.rect.x = "0";
    r1.rect.y = "0";
    r1.rect.width = "50";
    r1.rect.height = "50";
    block1.geometry.push_back(r1);

    mhs::model::RectOperation r2;
    r2.operation = mhs::model::GeometryOperation::Add;
    r2.rect.x = "50";
    r2.rect.y = "0";
    r2.rect.width = "50";
    r2.rect.height = "100";
    block1.geometry.push_back(r2);

    layer1.blocks.push_back(block1);
    io.layers.push_back(layer1);

    // mhs::model::LayerSpec 2 (substrate): silicon with sub-rect hole at (0,0,50,50)
    mhs::model::LayerSpec layer2;
    layer2.thickness = "10";

    mhs::model::BlockSpec block2;
    block2.material = "silicon";
    block2.volumetric_heat_source = "0";

    mhs::model::RectOperation r3;
    r3.operation = mhs::model::GeometryOperation::Add;
    r3.rect.x = "0";
    r3.rect.y = "0";
    r3.rect.width = "100";
    r3.rect.height = "100";
    block2.geometry.push_back(r3);

    mhs::model::RectOperation r4;
    r4.operation = mhs::model::GeometryOperation::Subtract; // subtract: removes (0,0,50,50) from the silicon
    r4.rect.x = "0";
    r4.rect.y = "0";
    r4.rect.width = "50";
    r4.rect.height = "50";
    block2.geometry.push_back(r4);

    layer2.blocks.push_back(block2);
    io.layers.push_back(layer2);

    // Materials
    mhs::model::MaterialSpec copper;
    copper.conductivity_x = copper.conductivity_y = copper.conductivity_z = "400";
    io.materials.push_back({"copper", copper});

    mhs::model::MaterialSpec silicon;
    silicon.conductivity_x = silicon.conductivity_y = silicon.conductivity_z = "130";
    io.materials.push_back({"silicon", silicon});

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    int nx = model.mesh.nx;
    int ny = model.mesh.ny;
    int nz = model.mesh.nz;
    EXPECT_EQ(nx, 2);
    EXPECT_EQ(ny, 2);
    EXPECT_EQ(nz, 9);

    // mhs::model::LayerSpec 1 is top layer, covering z=10..30mm (layers[0] = top)
    // mhs::model::LayerSpec 2 is substrate, covering z=0..10mm (layers[1])
    // In layer1 (z indices 4-8, covering z=10..30mm in SI):
    //   Block1 has add rects: (0,0,50,50) and (50,0,50,100)
    //   Cell (ix=0, iy=0) at cx=25mm=25e-3 is in rect1 (0,0,50,50) -> valid, copper
    //   Cell (ix=1, iy=0) at cx=75mm=75e-3 is in rect2 (50,0,50,100) -> valid, copper
    //   Cell (ix=0, iy=1) at cy=75mm is NOT in rect1 (cy=75 >= 0+50=50) -> virtual
    //   Cell (ix=1, iy=1) at cx=75,cy=75 is in rect2 (50,0,50,100) -> valid, copper

    // In layer2 (z indices 0-4, covering z=0..10mm):
    //   mhs::model::BlockSpec has (0,0,100,100) - subtract (0,0,50,50)
    //   Cell (ix=0, iy=0) -> subtracted -> virtual
    //   Cell (ix=1, iy=0) -> valid, silicon
    //   Cell (ix=0, iy=1) -> valid, silicon
    //   Cell (ix=1, iy=1) -> valid, silicon

    // Check cell (ix=0, iy=1, iz=0) -> layer2, not subtracted, in (0,0,100,100) -> valid
    // Actually cell at cy=75e-3 is NOT in sub-rect (0,0,50,50), so it IS valid
    int idx_01_0 = 0 * ny * nz + 1 * nz + 0;
    EXPECT_NE(model.cells.grid_to_cell[idx_01_0], mhs::invalidIndex); // valid in layer2

    // Check cell (ix=0, iy=0, iz=5) -> layer0 (top), (ix=0, iy=0) cx=25mm, cy=25mm in rect1 -> valid
    // iz=5 means cz=12.5mm, which is in top layer (z=10..30mm)
    int idx_00_5 = 0 * ny * nz + 0 * nz + 5;
    EXPECT_NE(model.cells.grid_to_cell[idx_00_5], mhs::invalidIndex);

    // Cell (ix=0, iy=0, iz=4) -> layer1 (substrate), cx=25mm cy=25mm -> subtracted -> virtual
    // iz=4 means cz=9mm, in substrate layer (z=0..10mm)
    int idx_00_4 = 0 * ny * nz + 0 * nz + 4;
    EXPECT_EQ(model.cells.grid_to_cell[idx_00_4], mhs::invalidIndex);
}

// ---- Structured boundary-region tests ----

TEST(PreprocessorTest, StructuredZRegionAssignsDirichletBoundary)
{
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0, 50, 100};
    io.mesh.y_vertices = {0, 50, 100};
    io.mesh.z_vertices = {0, 10, 20, 30};

    mhs::model::LayerSpec layer;
    layer.thickness = "30";

    mhs::model::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "0";

    mhs::model::RectOperation rect;
    rect.operation = mhs::model::GeometryOperation::Add;
    rect.rect.x = "0";
    rect.rect.y = "0";
    rect.rect.width = "100";
    rect.rect.height = "100";
    block.geometry.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec copper;
    copper.conductivity_x = copper.conductivity_y = copper.conductivity_z = "400";
    io.materials.push_back({"copper", copper});

    // mhs::model::BoundaryPatch: Dirichlet 500K on Z bottom face
    mhs::model::BoundaryPatch boundary;
    boundary.condition = mhs::model::DirichletBoundary {"500"};
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 50.0, 50.0, 100.0}, {50.0, 100.0, 0.0, 50.0}, {50.0, 100.0, 50.0, 100.0}}));
    io.boundaries.push_back(boundary);

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    // Face key "Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100"
    // Rects: {x:0-50, y:50-100}, {x:50-100, y:0-50}, {x:50-100, y:50-100}
    // Cell (ix=0, iy=1, iz=0) has cx=25mm, cy=75mm -> in rect1 -> FirstType on ZM
    int ny_bc = model.mesh.ny;
    int nz_bc = model.mesh.nz;
    int idx_bc = 0 * ny_bc * nz_bc + 1 * nz_bc + 0;
    size_t compact = model.cells.grid_to_cell[idx_bc];
    ASSERT_NE(compact, mhs::invalidIndex);

    EXPECT_EQ(get_bc_type(model, (uint32_t)compact, mhs::core::FaceDir::ZM), mhs::core::BcType::FirstType);

    // mhs::core::BCParamTable should have dirichlet_T entries
    EXPECT_FALSE(model.bc_params.dirichlet_T.empty());
    EXPECT_TRUE(model.bc_params.dirichlet_T[0].is_constant());
    EXPECT_NEAR(model.bc_params.dirichlet_T[0].constant_value(), 500.0, 1e-10);
}

TEST(PreprocessorTest, DefaultBoundaryFallback)
{
    auto io = make_simple_io();
    auto model = build_model(io);

    // With no explicit boundaries and a zero-flux default,
    // all faces on domain boundaries should have SecondType BC
    // Interior faces should have None BC (no patch entry)

    // Cell (0,0,0) - bottom-left-front cell:
    // XM (x=0 face): domain boundary -> SecondType
    // YM (y=0 face): domain boundary -> SecondType
    // ZM (z=0 face): domain boundary -> SecondType
    // XP, YP, ZP: interior or domain boundary
    int ny = model.mesh.ny;
    int nz = model.mesh.nz;
    int idx = 0 * ny * nz + 0 * nz + 0;
    size_t compact = model.cells.grid_to_cell[idx];
    ASSERT_NE(compact, mhs::invalidIndex);

    EXPECT_EQ(get_bc_type(model, (uint32_t)compact, mhs::core::FaceDir::XM), mhs::core::BcType::SecondType);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact, mhs::core::FaceDir::YM), mhs::core::BcType::SecondType);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact, mhs::core::FaceDir::ZM), mhs::core::BcType::SecondType);

    // Interior cell (1,1,1): XP, YP, ZP are domain boundaries
    // XM, YM, ZM are interior -> None (no patch)
    int idx_inner = 1 * ny * nz + 1 * nz + 1;
    size_t compact_inner = model.cells.grid_to_cell[idx_inner];
    ASSERT_NE(compact_inner, mhs::invalidIndex);

    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::XM), mhs::core::BcType::None);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::YM), mhs::core::BcType::None);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::ZM), mhs::core::BcType::None);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::XP), mhs::core::BcType::SecondType);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::YP), mhs::core::BcType::SecondType);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::ZP), mhs::core::BcType::SecondType);
}

// ---- Full Case1 Integration Test ----

TEST(PreprocessorTest, CellsOnExactBoundaryEdgeAreNotMisclassified)
{
    // This test verifies that cells whose centers fall exactly on
    // a block rect boundary edge (due to floating-point alignment)
    // are correctly classified as valid, not incorrectly excluded.
    //
    // Setup: 100x100x30mm domain, 2x2x9 cells.
    // mhs::model::BlockSpec rect: x=25, width=50, y=0, height=100 -> covers [25mm, 75mm] in X
    // Cell ix=0: cx=25mm -> exactly at rx=25mm (lower bound inclusive)
    // Cell ix=1: cx=75mm -> exactly at rx+rw=75mm (upper bound must be inclusive with epsilon)
    // Without epsilon tolerance, strict `<` on upper bound excludes ix=1.

    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0, 50, 100};
    io.mesh.y_vertices = {0, 50, 100};
    io.mesh.z_vertices = {0, 10, 20, 30};

    mhs::model::LayerSpec layer;
    layer.thickness = "30";

    mhs::model::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "0";

    mhs::model::RectOperation rect;
    rect.operation = mhs::model::GeometryOperation::Add;
    rect.rect.x = "25";
    rect.rect.y = "0";
    rect.rect.width = "50";
    rect.rect.height = "100";
    block.geometry.push_back(rect);

    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec copper;
    copper.conductivity_x = copper.conductivity_y = copper.conductivity_z = "400";
    io.materials.push_back({"copper", copper});

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    int ny = model.mesh.ny;
    int nz = model.mesh.nz;

    // Cell (ix=0, iy=0, iz=0): cx=25mm >= rx=25mm, should be valid
    int idx0 = 0 * ny * nz + 0 * nz + 0;
    EXPECT_NE(model.cells.grid_to_cell[idx0], mhs::invalidIndex);

    // Cell (ix=1, iy=0, iz=0): cx=75mm exactly equals rx+rw=75mm
    // Without epsilon tolerance, this cell is incorrectly classified as virtual
    int idx1 = 1 * ny * nz + 0 * nz + 0;
    EXPECT_NE(model.cells.grid_to_cell[idx1], mhs::invalidIndex);
}

TEST(PreprocessorTest, LaterBlockOverridesEarlierBlockInOverlap)
{
    // In CAD semantics, later blocks override earlier blocks in overlapping
    // regions. A chip (block2, silicon) overlaying a substrate (block1, copper)
    // should assign silicon material to cells in the overlap area.
    //
    // Before the fix: first-match logic gives block1 (copper) to overlap cells.
    // After the fix: last-match logic gives block2 (silicon) to overlap cells.

    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0, 50, 100};
    io.mesh.y_vertices = {0, 50, 100};
    io.mesh.z_vertices = {0, 10, 20, 30};

    mhs::model::LayerSpec layer;
    layer.thickness = "30";

    // mhs::model::BlockSpec 1: background substrate covering entire 100x100mm area (copper)
    mhs::model::BlockSpec block1;
    block1.material = "copper";
    block1.volumetric_heat_source = "0";

    mhs::model::RectOperation rect1;
    rect1.operation = mhs::model::GeometryOperation::Add;
    rect1.rect.x = "0";
    rect1.rect.y = "0";
    rect1.rect.width = "100";
    rect1.rect.height = "100";
    block1.geometry.push_back(rect1);

    // mhs::model::BlockSpec 2: chip overlaying the first quadrant (0-50, 0-50) (silicon)
    mhs::model::BlockSpec block2;
    block2.material = "silicon";
    block2.volumetric_heat_source = "1e7";

    mhs::model::RectOperation rect2;
    rect2.operation = mhs::model::GeometryOperation::Add;
    rect2.rect.x = "0";
    rect2.rect.y = "0";
    rect2.rect.width = "50";
    rect2.rect.height = "50";
    block2.geometry.push_back(rect2);

    layer.blocks.push_back(block1);
    layer.blocks.push_back(block2);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec copper;
    copper.conductivity_x = copper.conductivity_y = copper.conductivity_z = "400";
    io.materials.push_back({"copper", copper});

    mhs::model::MaterialSpec silicon;
    silicon.conductivity_x = silicon.conductivity_y = silicon.conductivity_z = "130";
    io.materials.push_back({"silicon", silicon});

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    int ny = model.mesh.ny;
    int nz = model.mesh.nz;

    // Cell (ix=0, iy=0, iz=0): cx=25mm, cy=25mm — in overlap of both blocks.
    // Last block (block2 = silicon) should override first block (block1 = copper).
    int idx_overlap = 0 * ny * nz + 0 * nz + 0;
    EXPECT_NE(model.cells.grid_to_cell[idx_overlap], mhs::invalidIndex);
    int c_overlap = (int)model.cells.grid_to_cell[idx_overlap];

    const mhs::core::FieldContext ctx {0.025, 0.025, 0.005, 300.0, 0.0};
    const auto overlap_material = model.cells.material_id[c_overlap];
    const auto overlap_heat_source = model.cells.heat_source_idx[c_overlap];
    EXPECT_DOUBLE_EQ(model.material_table[overlap_material].kx.eval(ctx), 130.0)
        << "Overlapping cell must get material from the later block";
    EXPECT_DOUBLE_EQ(model.heat_source_table[overlap_heat_source].eval(ctx), 1e7)
        << "Overlapping cell must get heat source from the same later block";

    // Cell (ix=1, iy=0, iz=0): cx=75mm, cy=25mm — only in block1 (copper)
    int idx_only_block1 = 1 * ny * nz + 0 * nz + 0;
    int c_only_block1 = (int)model.cells.grid_to_cell[idx_only_block1];
    const auto background_material = model.cells.material_id[c_only_block1];
    const auto background_heat_source = model.cells.heat_source_idx[c_only_block1];
    EXPECT_DOUBLE_EQ(model.material_table[background_material].kx.eval(ctx), 400.0)
        << "Cell outside the overlap must retain the earlier block material";
    EXPECT_DOUBLE_EQ(model.heat_source_table[background_heat_source].eval(ctx), 0.0)
        << "Cell outside the overlap must retain the earlier block heat source";
}

TEST(PreprocessorTest, LaterBoundaryOverridesEarlierBoundary)
{
    auto definition = make_simple_io();

    mhs::model::BoundaryPatch earlier;
    earlier.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    earlier.condition = mhs::model::DirichletBoundary {"310"};
    definition.boundaries.push_back(earlier);

    mhs::model::BoundaryPatch later;
    later.regions.push_back(mhs::test::face_region(mhs::model::Axis::Z, 0.0, {{0.0, 10.0, 0.0, 10.0}}));
    later.condition = mhs::model::ConvectionBoundary {"42", "280"};
    definition.boundaries.push_back(later);

    const auto model = build_model(definition);
    const auto cell = model.cells.grid_to_cell[0];
    ASSERT_NE(cell, mhs::invalidIndex);
    ASSERT_EQ(get_bc_type(model, static_cast<uint32_t>(cell), mhs::core::FaceDir::ZM), mhs::core::BcType::ThirdType);

    const uint16_t param_idx = get_bc_param(model, static_cast<uint32_t>(cell), mhs::core::FaceDir::ZM);
    const mhs::core::FieldContext ctx {0.0025, 0.0025, 0.0, 300.0, 0.0};
    ASSERT_LT(param_idx, model.bc_params.cauchy_h.size());
    ASSERT_LT(param_idx, model.bc_params.cauchy_T_inf.size());
    EXPECT_DOUBLE_EQ(model.bc_params.cauchy_h[param_idx].eval(ctx), 42.0);
    EXPECT_DOUBLE_EQ(model.bc_params.cauchy_T_inf[param_idx].eval(ctx), 280.0);
}

TEST(PreprocessorTest, StructuredYRegionAssignsYPBoundary)
{
    // End-to-end: feed a structured Y region, assert the correct Y+ face
    // of a cell adjacent to vertex_y = 7.5mm gets the assigned BC.
    //
    // Mesh: 2x2 cells over [0, 10]x[0, 10]x[0, 10] mm; one block filling all.
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0, 5, 10};
    io.mesh.y_vertices = {0, 5, 10};
    io.mesh.z_vertices = {0, 5, 10};

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

    // Convection on the upper half of the y=10mm Y+ face
    mhs::model::BoundaryPatch boundary;
    boundary.condition = mhs::model::ConvectionBoundary {"10", "200"};
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::Y, 10.0, {{0.0, 10.0, 0.0, 5.0}})); // cx: 0-10, cz: 0-5
    io.boundaries.push_back(boundary);

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    // Cells adjacent to the Y+ face at y=10mm have iy=1 (vertex_y[2] = 10).
    // Their YP face is exposed. mhs::model::RectOperation: cx in [0,10], cz in [0,5].
    // Cells with iz=0 (cz=2.5) are inside the rect and get ThirdType.
    // Cells with iz=1 (cz=7.5) are outside the rect and fall through to
    // the default boundary (SecondType).
    int ny = model.mesh.ny;
    int nz = model.mesh.nz;
    for (int ix = 0; ix < 2; ++ix) {
        int idx = ix * ny * nz + 1 * nz + 0; // iz=0 -> cz=2.5 in [0,5]
        int compact = (int)model.cells.grid_to_cell[idx];
        EXPECT_EQ(get_bc_type(model, (uint32_t)compact, mhs::core::FaceDir::YP), mhs::core::BcType::ThirdType)
            << "Y region should assign ThirdType to YP face at (" << ix << ",1,0)";
    }

    // Cells with iz=1 (cz=7.5) fall outside the rect and must NOT get this BC;
    // they should fall through to the default boundary (SecondType).
    int idx_outside = 0 * ny * nz + 1 * nz + 1; // ix=0, iy=1, iz=1 -> cz=7.5
    int compact_out = (int)model.cells.grid_to_cell[idx_outside];
    EXPECT_NE(get_bc_type(model, (uint32_t)compact_out, mhs::core::FaceDir::YP), mhs::core::BcType::ThirdType)
        << "Cell outside rect must not get the ThirdType BC";
}

TEST(PreprocessorTest, AppliesMultipleRegionsFromOneBoundaryPatch)
{
    // A single boundary carrying many regions must apply each one.
    // Mirrors case1 mhs::model::BoundaryPatch 5: 4 X-keys + 2 Y-keys covering the side faces
    // of the top die. Without correct per-key iteration, some faces would
    // silently fall through to the default boundary.
    mhs::model::ModelDefinition io;
    io.settings.study_type = mhs::model::StudyType::Steady;
    io.settings.length_unit = mhs::model::LengthUnit::Millimeter;
    io.settings.initial_temperature = 300.0;

    io.mesh.x_vertices = {0, 5, 10};
    io.mesh.y_vertices = {0, 5, 10};
    io.mesh.z_vertices = {0, 5, 10};

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

    // One boundary, three regions targeting three different exposed faces
    mhs::model::BoundaryPatch boundary;
    boundary.condition = mhs::model::ConvectionBoundary {"10", "200"};
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::X, 10.0, {{0.0, 10.0, 0.0, 10.0}})); // XP face, all cz
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::X, 0.0, {{0.0, 10.0, 0.0, 10.0}})); // XM face, all cz
    boundary.regions.push_back(mhs::test::face_region(mhs::model::Axis::Y, 10.0, {{0.0, 10.0, 0.0, 10.0}})); // YP face, all cz
    io.boundaries.push_back(boundary);

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    int ny = model.mesh.ny;
    int nz = model.mesh.nz;
    int nx = model.mesh.nx;

    // Every exposed face of every cell on the domain boundary (ix=0 XP/XP, etc.)
    // and specifically the three faces the keys target must be ThirdType.
    auto check_face = [&](int ix, int iy, int iz, mhs::core::FaceDir dir) {
        int idx = ix * ny * nz + iy * nz + iz;
        int compact = (int)model.cells.grid_to_cell[idx];
        EXPECT_EQ(get_bc_type(model, (uint32_t)compact, dir), mhs::core::BcType::ThirdType)
            << "Face " << (int)dir << " of cell (" << ix << "," << iy << "," << iz
            << ") should be ThirdType from one of the boundary regions";
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
