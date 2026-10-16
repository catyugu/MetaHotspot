#include "compiler/model_compiler.hpp"
#include "core/model_definition.hpp"
#include "model_test_utils.hpp"
#include "numerics/expression/expr.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace mhs::core;
using namespace mhs::sim;

// Helpers for testing patch-based BC assignment.

/// Find a FaceBC on `cell_idx` with direction `dir`.
/// Returns pointer or nullptr if that face has BcType::None.
static const mhs::core::FaceBC* find_face_bc(const Model& model, uint32_t cell_idx, FaceDir dir)
{
    for (auto id : model.discrete.incident_faces(CellId {cell_idx})) {
        const auto& f = model.discrete.faces[id.value];
        if (!f.neighbor.valid() && f.axis == static_cast<int>(dir) / 2
            && f.sign == (static_cast<int>(dir) % 2 ? 1 : -1)) {
            for (const auto& term : boundary_terms_on_face(model, id))
                if (term.bc.type != BcType::None)
                    return &term.bc;
        }
    }
    return nullptr;
}

/// Return the BcType on cell_idx for direction dir, or None if not found.
static BcType get_bc_type(const Model& model, uint32_t cell_idx, FaceDir dir)
{
    auto* p = find_face_bc(model, cell_idx, dir);
    return p ? p->type : BcType::None;
}

/// Return the param_idx on cell_idx for direction dir, or 0 if not found.
static mhs::core::TableIndex get_bc_param(const Model& model, uint32_t cell_idx, FaceDir dir)
{
    auto* p = find_face_bc(model, cell_idx, dir);
    return p ? p->param_idx : 0;
}

static TableIndex source_id(const Model& model, Index cell)
{
    for (const auto& term : model.source_terms)
        if (term.cell.value == cell)
            return term.source;
    throw std::logic_error("missing source projection");
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

TEST(PreprocessorTest, CartesianAdapterPreservesSIBounds)
{
    const auto model = build_model(make_simple_io());
    ASSERT_EQ(model.discrete.cells.size(), 8u);
    const auto& first = model.discrete.cells.front();
    const auto& last = model.discrete.cells.back();
    for (int a = 0; a < 3; ++a) {
        EXPECT_NEAR(first.min[a], 0, 1e-12);
        EXPECT_NEAR(first.max[a], 0.005, 1e-12);
        EXPECT_NEAR(last.center()[a], 0.0075, 1e-12);
    }
}

// Test-only conversion of a supplied Cartesian fixture index to a physical query.
// Runtime consumers receive neither these indices nor these axes.
static Index input_cell(const Model& model, const mhs::model::MeshSpec& input, Index flat)
{
    const auto ny = input.y_vertices.size() - 1, nz = input.z_vertices.size() - 1;
    const auto x = flat / (ny * nz), y = (flat % (ny * nz)) / nz, z = flat % nz;
    return model.discrete
        .locate({(input.x_vertices[x] + input.x_vertices[x + 1]) * 0.0005,
            (input.y_vertices[y] + input.y_vertices[y + 1]) * 0.0005,
            (input.z_vertices[z] + input.z_vertices[z + 1]) * 0.0005})
        .value;
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

    EXPECT_NE(input_cell(model, definition.mesh, 0), mhs::core::invalidIndex)
        << "The initial Add must keep the first cell";
    EXPECT_EQ(input_cell(model, definition.mesh, 1), mhs::core::invalidIndex)
        << "The later Subtract must remove the middle cell";
    EXPECT_NE(input_cell(model, definition.mesh, 2), mhs::core::invalidIndex)
        << "The final Add must restore the last cell";
}

TEST(PreprocessorTest, LayeredCutoutsPreservePhysicalDomain)
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

    // Layer 0 occupies z=10..30 mm; the substrate occupies z=0..10 mm.
    EXPECT_TRUE(model.discrete.locate({0.025, 0.075, 0.001}).valid());
    EXPECT_TRUE(model.discrete.locate({0.025, 0.025, 0.0125}).valid());
    EXPECT_FALSE(model.discrete.locate({0.025, 0.025, 0.009}).valid());
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
    boundary.regions.push_back(mhs::test::face_region(
        mhs::model::Axis::Z, 0.0, {{0.0, 50.0, 50.0, 100.0}, {50.0, 100.0, 0.0, 50.0}, {50.0, 100.0, 50.0, 100.0}}));
    io.boundaries.push_back(boundary);

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    // Face key "Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100"
    // Rects: {x:0-50, y:50-100}, {x:50-100, y:0-50}, {x:50-100, y:50-100}
    // Cell (ix=0, iy=1, iz=0) has cx=25mm, cy=75mm -> in rect1 -> FirstType on ZM
    int ny_bc = (io.mesh.y_vertices.size() - 1);
    int nz_bc = (io.mesh.z_vertices.size() - 1);
    int idx_bc = 0 * ny_bc * nz_bc + 1 * nz_bc + 0;
    size_t compact = input_cell(model, io.mesh, idx_bc);
    ASSERT_NE(compact, mhs::core::invalidIndex);

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
    int ny = (io.mesh.y_vertices.size() - 1);
    int nz = (io.mesh.z_vertices.size() - 1);
    int idx = 0 * ny * nz + 0 * nz + 0;
    size_t compact = input_cell(model, io.mesh, idx);
    ASSERT_NE(compact, mhs::core::invalidIndex);

    EXPECT_EQ(get_bc_type(model, (uint32_t)compact, mhs::core::FaceDir::XM), mhs::core::BcType::SecondType);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact, mhs::core::FaceDir::YM), mhs::core::BcType::SecondType);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact, mhs::core::FaceDir::ZM), mhs::core::BcType::SecondType);

    // Interior cell (1,1,1): XP, YP, ZP are domain boundaries
    // XM, YM, ZM are interior -> None (no patch)
    int idx_inner = 1 * ny * nz + 1 * nz + 1;
    size_t compact_inner = input_cell(model, io.mesh, idx_inner);
    ASSERT_NE(compact_inner, mhs::core::invalidIndex);

    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::XM), mhs::core::BcType::None);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::YM), mhs::core::BcType::None);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::ZM), mhs::core::BcType::None);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::XP), mhs::core::BcType::SecondType);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::YP), mhs::core::BcType::SecondType);
    EXPECT_EQ(get_bc_type(model, (uint32_t)compact_inner, mhs::core::FaceDir::ZP), mhs::core::BcType::SecondType);
}

// ---- Full Case1 Integration Test ----

TEST(PreprocessorTest, RejectsControlVolumesCutByPhysicalDomainBoundary)
{
    // Center inclusion is insufficient: both old cells extend outside [25,75].
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

    EXPECT_THROW(build_model(io), std::invalid_argument);
    io.mesh.x_vertices = {25, 50, 75};
    const auto model = build_model(io);
    ASSERT_EQ(model.discrete.cells.size(), 12u);
    EXPECT_NEAR(model.discrete.cells.front().min[0], 0.025, 1e-12);
    EXPECT_NEAR(model.discrete.cells.back().max[0], 0.075, 1e-12);
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

    int ny = (io.mesh.y_vertices.size() - 1);
    int nz = (io.mesh.z_vertices.size() - 1);

    // Cell (ix=0, iy=0, iz=0): cx=25mm, cy=25mm — in overlap of both blocks.
    // Last block (block2 = silicon) should override first block (block1 = copper).
    int idx_overlap = 0 * ny * nz + 0 * nz + 0;
    EXPECT_NE(input_cell(model, io.mesh, idx_overlap), mhs::core::invalidIndex);
    int c_overlap = (int)input_cell(model, io.mesh, idx_overlap);

    const mhs::core::FieldContext ctx {0.025, 0.025, 0.005, 300.0, 0.0};
    const auto overlap_material = model.cell_material[c_overlap];
    const auto overlap_heat_source = source_id(model, c_overlap);
    EXPECT_DOUBLE_EQ(model.material_table[overlap_material].kx.eval(ctx), 130.0)
        << "Overlapping cell must get material from the later block";
    EXPECT_DOUBLE_EQ(model.heat_source_table[overlap_heat_source].eval(ctx), 1e7)
        << "Overlapping cell must get heat source from the same later block";

    // Cell (ix=1, iy=0, iz=0): cx=75mm, cy=25mm — only in block1 (copper)
    int idx_only_block1 = 1 * ny * nz + 0 * nz + 0;
    int c_only_block1 = (int)input_cell(model, io.mesh, idx_only_block1);
    const auto background_material = model.cell_material[c_only_block1];
    const auto background_heat_source = source_id(model, c_only_block1);
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
    const auto cell = input_cell(model, definition.mesh, 0);
    ASSERT_NE(cell, mhs::core::invalidIndex);
    ASSERT_EQ(get_bc_type(model, static_cast<uint32_t>(cell), mhs::core::FaceDir::ZM), mhs::core::BcType::ThirdType);

    const mhs::core::TableIndex param_idx = get_bc_param(model, static_cast<uint32_t>(cell), mhs::core::FaceDir::ZM);
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
    boundary.regions.push_back(
        mhs::test::face_region(mhs::model::Axis::Y, 10.0, {{0.0, 10.0, 0.0, 5.0}})); // cx: 0-10, cz: 0-5
    io.boundaries.push_back(boundary);

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    // Cells adjacent to the Y+ face at y=10mm have iy=1 (vertex_y[2] = 10).
    // Their YP face is exposed. mhs::model::RectOperation: cx in [0,10], cz in [0,5].
    // Cells with iz=0 (cz=2.5) are inside the rect and get ThirdType.
    // Cells with iz=1 (cz=7.5) are outside the rect and fall through to
    // the default boundary (SecondType).
    int ny = (io.mesh.y_vertices.size() - 1);
    int nz = (io.mesh.z_vertices.size() - 1);
    for (int ix = 0; ix < 2; ++ix) {
        int idx = ix * ny * nz + 1 * nz + 0; // iz=0 -> cz=2.5 in [0,5]
        int compact = (int)input_cell(model, io.mesh, idx);
        EXPECT_EQ(get_bc_type(model, (uint32_t)compact, mhs::core::FaceDir::YP), mhs::core::BcType::ThirdType)
            << "Y region should assign ThirdType to YP face at (" << ix << ",1,0)";
    }

    // Cells with iz=1 (cz=7.5) fall outside the rect and must NOT get this BC;
    // they should fall through to the default boundary (SecondType).
    int idx_outside = 0 * ny * nz + 1 * nz + 1; // ix=0, iy=1, iz=1 -> cz=7.5
    int compact_out = (int)input_cell(model, io.mesh, idx_outside);
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
    boundary.regions.push_back(
        mhs::test::face_region(mhs::model::Axis::X, 10.0, {{0.0, 10.0, 0.0, 10.0}})); // XP face, all cz
    boundary.regions.push_back(
        mhs::test::face_region(mhs::model::Axis::X, 0.0, {{0.0, 10.0, 0.0, 10.0}})); // XM face, all cz
    boundary.regions.push_back(
        mhs::test::face_region(mhs::model::Axis::Y, 10.0, {{0.0, 10.0, 0.0, 10.0}})); // YP face, all cz
    io.boundaries.push_back(boundary);

    io.default_boundary = mhs::model::NeumannBoundary {};

    auto model = build_model(io);

    int ny = (io.mesh.y_vertices.size() - 1);
    int nz = (io.mesh.z_vertices.size() - 1);
    int nx = (io.mesh.x_vertices.size() - 1);

    // Every exposed face of every cell on the domain boundary (ix=0 XP/XP, etc.)
    // and specifically the three faces the keys target must be ThirdType.
    auto check_face = [&](int ix, int iy, int iz, mhs::core::FaceDir dir) {
        int idx = ix * ny * nz + iy * nz + iz;
        int compact = (int)input_cell(model, io.mesh, idx);
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
