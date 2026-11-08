#include "expression/expression.hpp"
#include "model/model_definition.hpp"
#include "pta_test_utils.hpp"
#include "solver/thermal_reconstruction.hpp"
#include <gtest/gtest.h>

using namespace mhs::core;
using namespace mhs::model;
using namespace mhs::compiler;

enum class FaceDir { XM, XP, YM, YP, ZM, ZP };

// Helpers for testing patch-based BC assignment.

/// Find a boundary integration face on `cell_idx` with direction `dir`.
/// Returns nullptr for an internal face without a boundary face.
static const mhs::model::BoundaryFace* find_face_bc(const Model& model, std::size_t cell_idx, FaceDir dir)
{
    for (const auto& term : model.boundary_faces)
        if (term.cell == cell_idx && static_cast<int>(term.axis) == static_cast<int>(dir) / 2
            && term.sign == (static_cast<int>(dir) % 2 ? 1 : -1))
            return &term;
    return nullptr;
}

template <typename Condition>
static const Condition* find_condition(const Model& model, std::size_t cell_idx, FaceDir dir)
{
    const auto* term = find_face_bc(model, cell_idx, dir);
    return term ? std::get_if<Condition>(&model.boundary_conditions[term->condition]) : nullptr;
}

static SourceId source_id(const Model& model, std::size_t cell)
{
    for (const auto& term : model.source_terms)
        if (term.cell == cell)
            return term.source;
    throw std::logic_error("missing source projection");
}

// Helper: build a minimal mhs::app::pta::Definition for testing
static mhs::app::pta::Definition make_simple_io()
{
    mhs::app::pta::Definition io;
    io.model.settings.study_type = mhs::model::StudyType::Steady;
    io.length_unit = mhs::app::pta::LengthUnit::Millimeter;
    io.model.settings.initial_temperature = 300.0;

    // Simple 10x10x10 mm cube, 2 cells each direction
    io.model.mesh.x_vertices = {0.0, 5.0, 10.0};
    io.model.mesh.y_vertices = {0.0, 5.0, 10.0};
    io.model.mesh.z_vertices = {0.0, 5.0, 10.0};

    // One layer, one block covering the whole area
    mhs::app::pta::LayerSpec layer;
    layer.thickness = "10";

    mhs::app::pta::BlockSpec block;
    block.material = "test_material";
    block.volumetric_heat_source = "0";

    mhs::app::pta::RectOperation rect;
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
    io.model.materials.push_back({"test_material", mat});

    // No explicit boundaries: the default boundary applies.
    io.model.default_thermal_boundary = mhs::model::NeumannBoundary {};

    return io;
}

TEST(PreprocessorTest, CartesianAdapterPreservesSIBounds)
{
    const auto model = test_pta::build_model(make_simple_io());
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
static std::size_t input_cell(const Model& model, const mhs::model::MeshSpec& input, std::size_t flat)
{
    const auto ny = input.y_vertices.size() - 1, nz = input.z_vertices.size() - 1;
    const auto x = flat / (ny * nz), y = (flat % (ny * nz)) / nz, z = flat % nz;
    return mhs::solver::locate_cell(model.discrete,
        {(input.x_vertices[x] + input.x_vertices[x + 1]) * 0.0005,
            (input.y_vertices[y] + input.y_vertices[y + 1]) * 0.0005,
            (input.z_vertices[z] + input.z_vertices[z + 1]) * 0.0005});
}

TEST(PreprocessorTest, RectOperationsFollowAppendOrder)
{
    auto definition = make_simple_io();
    definition.model.mesh.x_vertices = {0.0, 5.0, 10.0, 15.0};
    definition.model.mesh.y_vertices = {0.0, 5.0};
    definition.model.mesh.z_vertices = {0.0, 5.0};
    definition.layers[0].thickness = "5";

    auto& rects = definition.layers[0].blocks[0].geometry;
    rects[0].rect.width = "15";
    rects[0].rect.height = "5";

    mhs::app::pta::RectOperation subtract;
    subtract.operation = mhs::model::GeometryOperation::Subtract;
    subtract.rect.x = "5";
    subtract.rect.y = "0";
    subtract.rect.width = "10";
    subtract.rect.height = "5";
    rects.push_back(subtract);

    mhs::app::pta::RectOperation add_back;
    add_back.operation = mhs::model::GeometryOperation::Add;
    add_back.rect.x = "10";
    add_back.rect.y = "0";
    add_back.rect.width = "5";
    add_back.rect.height = "5";
    rects.push_back(add_back);

    const auto model = test_pta::build_model(definition);

    EXPECT_NE(input_cell(model, definition.model.mesh, 0), mhs::core::invalid_id)
        << "The initial Add must keep the first cell";
    EXPECT_EQ(input_cell(model, definition.model.mesh, 1), mhs::core::invalid_id)
        << "The later Subtract must remove the middle cell";
    EXPECT_NE(input_cell(model, definition.model.mesh, 2), mhs::core::invalid_id)
        << "The final Add must restore the last cell";
}

TEST(PreprocessorTest, LayeredCutoutsPreservePhysicalDomain)
{
    mhs::app::pta::Definition io;
    io.model.settings.study_type = mhs::model::StudyType::Steady;
    io.length_unit = mhs::app::pta::LengthUnit::Millimeter;
    io.model.settings.initial_temperature = 300.0;

    // 100x100x30mm, with cells at x:0,50,100 y:0,50,100 z:0,2,4,6,8,10,15,20,25,30
    io.model.mesh.x_vertices = {0, 50, 100};
    io.model.mesh.y_vertices = {0, 50, 100};
    io.model.mesh.z_vertices = {0, 2, 4, 6, 8, 10, 15, 20, 25, 30};

    // mhs::app::pta::LayerSpec 1 (top): 2 blocks, with add/sub rects creating L-shape
    mhs::app::pta::LayerSpec layer1;
    layer1.thickness = "20";

    // mhs::app::pta::BlockSpec 1: L-shape via add rect (0,0,50,50) and (50,0,50,100)
    mhs::app::pta::BlockSpec block1;
    block1.material = "copper";
    block1.volumetric_heat_source = "0";

    mhs::app::pta::RectOperation r1;
    r1.operation = mhs::model::GeometryOperation::Add;
    r1.rect.x = "0";
    r1.rect.y = "0";
    r1.rect.width = "50";
    r1.rect.height = "50";
    block1.geometry.push_back(r1);

    mhs::app::pta::RectOperation r2;
    r2.operation = mhs::model::GeometryOperation::Add;
    r2.rect.x = "50";
    r2.rect.y = "0";
    r2.rect.width = "50";
    r2.rect.height = "100";
    block1.geometry.push_back(r2);

    layer1.blocks.push_back(block1);
    io.layers.push_back(layer1);

    // mhs::app::pta::LayerSpec 2 (substrate): silicon with sub-rect hole at (0,0,50,50)
    mhs::app::pta::LayerSpec layer2;
    layer2.thickness = "10";

    mhs::app::pta::BlockSpec block2;
    block2.material = "silicon";
    block2.volumetric_heat_source = "0";

    mhs::app::pta::RectOperation r3;
    r3.operation = mhs::model::GeometryOperation::Add;
    r3.rect.x = "0";
    r3.rect.y = "0";
    r3.rect.width = "100";
    r3.rect.height = "100";
    block2.geometry.push_back(r3);

    mhs::app::pta::RectOperation r4;
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
    io.model.materials.push_back({"copper", copper});

    mhs::model::MaterialSpec silicon;
    silicon.conductivity_x = silicon.conductivity_y = silicon.conductivity_z = "130";
    io.model.materials.push_back({"silicon", silicon});

    io.model.default_thermal_boundary = mhs::model::NeumannBoundary {};

    auto model = test_pta::build_model(io);

    // Layer 0 occupies z=10..30 mm; the substrate occupies z=0..10 mm.
    EXPECT_TRUE((mhs::solver::locate_cell(model.discrete, {0.025, 0.075, 0.001}) != mhs::core::invalid_id));
    EXPECT_TRUE((mhs::solver::locate_cell(model.discrete, {0.025, 0.025, 0.0125}) != mhs::core::invalid_id));
    EXPECT_FALSE((mhs::solver::locate_cell(model.discrete, {0.025, 0.025, 0.009}) != mhs::core::invalid_id));
}

// ---- Structured boundary-region tests ----

TEST(PreprocessorTest, StructuredZRegionAssignsDirichletBoundary)
{
    mhs::app::pta::Definition io;
    io.model.settings.study_type = mhs::model::StudyType::Steady;
    io.length_unit = mhs::app::pta::LengthUnit::Millimeter;
    io.model.settings.initial_temperature = 300.0;

    io.model.mesh.x_vertices = {0, 50, 100};
    io.model.mesh.y_vertices = {0, 50, 100};
    io.model.mesh.z_vertices = {0, 10, 20, 30};

    mhs::app::pta::LayerSpec layer;
    layer.thickness = "30";

    mhs::app::pta::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "0";

    mhs::app::pta::RectOperation rect;
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
    io.model.materials.push_back({"copper", copper});

    // mhs::model::ThermalBoundarySpec: Dirichlet 500K on Z bottom face
    mhs::model::ThermalBoundarySpec boundary;
    boundary.condition = mhs::model::DirichletBoundary {"500"};
    for (const auto& rectangle : {mhs::model::RegionRect {0.0, 50.0, 50.0, 100.0},
             mhs::model::RegionRect {50.0, 100.0, 0.0, 50.0}, mhs::model::RegionRect {50.0, 100.0, 50.0, 100.0}})
        boundary.regions.push_back({mhs::model::Axis::Z, 0.0, rectangle});
    io.model.thermal_boundaries.push_back(boundary);

    io.model.default_thermal_boundary = mhs::model::NeumannBoundary {};

    auto model = test_pta::build_model(io);

    // Face key "Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100"
    // Rects: {x:0-50, y:50-100}, {x:50-100, y:0-50}, {x:50-100, y:50-100}
    // Cell (ix=0, iy=1, iz=0) has cx=25mm, cy=75mm -> in rect1 -> Dirichlet on ZM
    const auto ny_bc = (io.model.mesh.y_vertices.size() - 1);
    const auto nz_bc = (io.model.mesh.z_vertices.size() - 1);
    const auto idx_bc = 0 * ny_bc * nz_bc + 1 * nz_bc + 0;
    const auto compact = input_cell(model, io.model.mesh, idx_bc);
    ASSERT_NE(compact, mhs::core::invalid_id);

    const auto* condition = find_condition<Dirichlet>(model, compact, FaceDir::ZM);
    ASSERT_NE(condition, nullptr);
    EXPECT_TRUE(condition->temperature.is_constant());
    EXPECT_NEAR(condition->temperature.constant_value(), 500.0, 1e-10);
}

TEST(PreprocessorTest, DefaultBoundaryFallback)
{
    auto io = make_simple_io();
    auto model = test_pta::build_model(io);

    // With no explicit boundaries and a zero-flux default,
    // all faces on domain boundaries should have Neumann BC
    // Interior faces should have no boundary term (no patch entry)

    // Cell (0,0,0) - bottom-left-front cell:
    // XM (x=0 face): domain boundary -> Neumann
    // YM (y=0 face): domain boundary -> Neumann
    // ZM (z=0 face): domain boundary -> Neumann
    // XP, YP, ZP: interior or domain boundary
    const auto ny = (io.model.mesh.y_vertices.size() - 1);
    const auto nz = (io.model.mesh.z_vertices.size() - 1);
    const auto idx = 0 * ny * nz + 0 * nz + 0;
    const auto compact = input_cell(model, io.model.mesh, idx);
    ASSERT_NE(compact, mhs::core::invalid_id);

    EXPECT_NE(find_condition<Neumann>(model, compact, FaceDir::XM), nullptr);
    EXPECT_NE(find_condition<Neumann>(model, compact, FaceDir::YM), nullptr);
    EXPECT_NE(find_condition<Neumann>(model, compact, FaceDir::ZM), nullptr);

    // Interior cell (1,1,1): XP, YP, ZP are domain boundaries
    // XM, YM, ZM are interior -> no boundary term
    const auto idx_inner = 1 * ny * nz + 1 * nz + 1;
    const auto compact_inner = input_cell(model, io.model.mesh, idx_inner);
    ASSERT_NE(compact_inner, mhs::core::invalid_id);

    EXPECT_EQ(find_face_bc(model, compact_inner, FaceDir::XM), nullptr);
    EXPECT_EQ(find_face_bc(model, compact_inner, FaceDir::YM), nullptr);
    EXPECT_EQ(find_face_bc(model, compact_inner, FaceDir::ZM), nullptr);
    EXPECT_NE(find_condition<Neumann>(model, compact_inner, FaceDir::XP), nullptr);
    EXPECT_NE(find_condition<Neumann>(model, compact_inner, FaceDir::YP), nullptr);
    EXPECT_NE(find_condition<Neumann>(model, compact_inner, FaceDir::ZP), nullptr);
}

// ---- Full Case1 Integration Test ----

TEST(PreprocessorTest, UsesSuppliedDomainAlignedControlVolumes)
{
    // The caller supplies control volumes aligned with the physical domain.
    mhs::app::pta::Definition io;
    io.model.settings.study_type = mhs::model::StudyType::Steady;
    io.length_unit = mhs::app::pta::LengthUnit::Millimeter;
    io.model.settings.initial_temperature = 300.0;

    io.model.mesh.x_vertices = {25, 50, 75};
    io.model.mesh.y_vertices = {0, 50, 100};
    io.model.mesh.z_vertices = {0, 10, 20, 30};

    mhs::app::pta::LayerSpec layer;
    layer.thickness = "30";

    mhs::app::pta::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "0";

    mhs::app::pta::RectOperation rect;
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
    io.model.materials.push_back({"copper", copper});

    io.model.default_thermal_boundary = mhs::model::NeumannBoundary {};

    const auto model = test_pta::build_model(io);
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

    mhs::app::pta::Definition io;
    io.model.settings.study_type = mhs::model::StudyType::Steady;
    io.length_unit = mhs::app::pta::LengthUnit::Millimeter;
    io.model.settings.initial_temperature = 300.0;

    io.model.mesh.x_vertices = {0, 50, 100};
    io.model.mesh.y_vertices = {0, 50, 100};
    io.model.mesh.z_vertices = {0, 10, 20, 30};

    mhs::app::pta::LayerSpec layer;
    layer.thickness = "30";

    // mhs::app::pta::BlockSpec 1: background substrate covering entire 100x100mm area (copper)
    mhs::app::pta::BlockSpec block1;
    block1.material = "copper";
    block1.volumetric_heat_source = "0";

    mhs::app::pta::RectOperation rect1;
    rect1.operation = mhs::model::GeometryOperation::Add;
    rect1.rect.x = "0";
    rect1.rect.y = "0";
    rect1.rect.width = "100";
    rect1.rect.height = "100";
    block1.geometry.push_back(rect1);

    // mhs::app::pta::BlockSpec 2: chip overlaying the first quadrant (0-50, 0-50) (silicon)
    mhs::app::pta::BlockSpec block2;
    block2.material = "silicon";
    block2.volumetric_heat_source = "1e7";

    mhs::app::pta::RectOperation rect2;
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
    io.model.materials.push_back({"copper", copper});

    mhs::model::MaterialSpec silicon;
    silicon.conductivity_x = silicon.conductivity_y = silicon.conductivity_z = "130";
    io.model.materials.push_back({"silicon", silicon});

    io.model.default_thermal_boundary = mhs::model::NeumannBoundary {};

    auto model = test_pta::build_model(io);

    const auto ny = (io.model.mesh.y_vertices.size() - 1);
    const auto nz = (io.model.mesh.z_vertices.size() - 1);

    // Cell (ix=0, iy=0, iz=0): cx=25mm, cy=25mm — in overlap of both blocks.
    // Last block (block2 = silicon) should override first block (block1 = copper).
    const auto idx_overlap = 0 * ny * nz + 0 * nz + 0;
    EXPECT_NE(input_cell(model, io.model.mesh, idx_overlap), mhs::core::invalid_id);
    const auto c_overlap = input_cell(model, io.model.mesh, idx_overlap);

    const mhs::expression::FieldContext ctx {0.025, 0.025, 0.005, 300.0, 0.0};
    const auto overlap_material = model.cell_material[c_overlap];
    const auto overlap_heat_source = source_id(model, c_overlap);
    EXPECT_DOUBLE_EQ(model.material_table[overlap_material].kx.eval(ctx), 130.0)
        << "Overlapping cell must get material from the later block";
    EXPECT_DOUBLE_EQ(model.heat_source_table[overlap_heat_source].eval(ctx), 1e7)
        << "Overlapping cell must get heat source from the same later block";

    // Cell (ix=1, iy=0, iz=0): cx=75mm, cy=25mm — only in block1 (copper)
    const auto idx_only_block1 = 1 * ny * nz + 0 * nz + 0;
    const auto c_only_block1 = input_cell(model, io.model.mesh, idx_only_block1);
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

    mhs::model::ThermalBoundarySpec earlier;
    earlier.regions.push_back(test_pta::face_region(mhs::model::Axis::Z, 0.0, {0.0, 10.0, 0.0, 10.0}));
    earlier.condition = mhs::model::DirichletBoundary {"310"};
    definition.model.thermal_boundaries.push_back(earlier);

    mhs::model::ThermalBoundarySpec later;
    later.regions.push_back(test_pta::face_region(mhs::model::Axis::Z, 0.0, {0.0, 10.0, 0.0, 10.0}));
    later.condition = mhs::model::ConvectionBoundary {"42", "280"};
    definition.model.thermal_boundaries.push_back(later);

    const auto model = test_pta::build_model(definition);
    const auto cell = input_cell(model, definition.model.mesh, 0);
    ASSERT_NE(cell, mhs::core::invalid_id);
    const auto* condition = find_condition<Convection>(model, cell, FaceDir::ZM);
    ASSERT_NE(condition, nullptr);
    const mhs::expression::FieldContext ctx {0.0025, 0.0025, 0.0, 300.0, 0.0};
    EXPECT_DOUBLE_EQ(condition->coefficient.eval(ctx), 42.0);
    EXPECT_DOUBLE_EQ(condition->ambient_temperature.eval(ctx), 280.0);
}

TEST(PreprocessorTest, StructuredYRegionAssignsYPBoundary)
{
    // End-to-end: feed a structured Y region, assert the correct Y+ face
    // of a cell adjacent to vertex_y = 7.5mm gets the assigned BC.
    //
    // Mesh: 2x2 cells over [0, 10]x[0, 10]x[0, 10] mm; one block filling all.
    mhs::app::pta::Definition io;
    io.model.settings.study_type = mhs::model::StudyType::Steady;
    io.length_unit = mhs::app::pta::LengthUnit::Millimeter;
    io.model.settings.initial_temperature = 300.0;

    io.model.mesh.x_vertices = {0, 5, 10};
    io.model.mesh.y_vertices = {0, 5, 10};
    io.model.mesh.z_vertices = {0, 5, 10};

    mhs::app::pta::LayerSpec layer;
    layer.thickness = "10";

    mhs::app::pta::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "0";

    mhs::app::pta::RectOperation rect;
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
    io.model.materials.push_back({"copper", copper});

    // Convection on the upper half of the y=10mm Y+ face
    mhs::model::ThermalBoundarySpec boundary;
    boundary.condition = mhs::model::ConvectionBoundary {"10", "200"};
    boundary.regions.push_back(
        test_pta::face_region(mhs::model::Axis::Y, 10.0, {0.0, 10.0, 0.0, 5.0})); // cx: 0-10, cz: 0-5
    io.model.thermal_boundaries.push_back(boundary);

    io.model.default_thermal_boundary = mhs::model::NeumannBoundary {};

    auto model = test_pta::build_model(io);

    // Cells adjacent to the Y+ face at y=10mm have iy=1 (vertex_y[2] = 10).
    // Their YP face is exposed. mhs::app::pta::RectOperation: cx in [0,10], cz in [0,5].
    // Cells with iz=0 (cz=2.5) are inside the rect and get Convection.
    // Cells with iz=1 (cz=7.5) are outside the rect and fall through to
    // the default boundary (Neumann).
    const auto ny = (io.model.mesh.y_vertices.size() - 1);
    const auto nz = (io.model.mesh.z_vertices.size() - 1);
    for (int ix = 0; ix < 2; ++ix) {
        const auto idx = ix * ny * nz + 1 * nz + 0; // iz=0 -> cz=2.5 in [0,5]
        const auto compact = input_cell(model, io.model.mesh, idx);
        EXPECT_NE(find_condition<Convection>(model, compact, FaceDir::YP), nullptr)
            << "Y region should assign Convection to YP face at (" << ix << ",1,0)";
    }

    // Cells with iz=1 (cz=7.5) fall outside the rect and must NOT get this BC;
    // they should fall through to the default boundary (Neumann).
    const auto idx_outside = 0 * ny * nz + 1 * nz + 1; // ix=0, iy=1, iz=1 -> cz=7.5
    const auto compact_out = input_cell(model, io.model.mesh, idx_outside);
    EXPECT_EQ(find_condition<Convection>(model, compact_out, FaceDir::YP), nullptr)
        << "Cell outside rect must not get the Convection BC";
}

TEST(PreprocessorTest, AppliesMultipleRegionsFromOneThermalBoundarySpec)
{
    // A single boundary carrying many regions must apply each one.
    // Mirrors case1 mhs::model::ThermalBoundarySpec 5: 4 X-keys + 2 Y-keys covering the side faces
    // of the top die. Without correct per-key iteration, some faces would
    // silently fall through to the default boundary.
    mhs::app::pta::Definition io;
    io.model.settings.study_type = mhs::model::StudyType::Steady;
    io.length_unit = mhs::app::pta::LengthUnit::Millimeter;
    io.model.settings.initial_temperature = 300.0;

    io.model.mesh.x_vertices = {0, 5, 10};
    io.model.mesh.y_vertices = {0, 5, 10};
    io.model.mesh.z_vertices = {0, 5, 10};

    mhs::app::pta::LayerSpec layer;
    layer.thickness = "10";

    mhs::app::pta::BlockSpec block;
    block.material = "copper";
    block.volumetric_heat_source = "0";

    mhs::app::pta::RectOperation rect;
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
    io.model.materials.push_back({"copper", copper});

    // One boundary, three regions targeting three different exposed faces
    mhs::model::ThermalBoundarySpec boundary;
    boundary.condition = mhs::model::ConvectionBoundary {"10", "200"};
    boundary.regions.push_back(
        test_pta::face_region(mhs::model::Axis::X, 10.0, {0.0, 10.0, 0.0, 10.0})); // XP face, all cz
    boundary.regions.push_back(
        test_pta::face_region(mhs::model::Axis::X, 0.0, {0.0, 10.0, 0.0, 10.0})); // XM face, all cz
    boundary.regions.push_back(
        test_pta::face_region(mhs::model::Axis::Y, 10.0, {0.0, 10.0, 0.0, 10.0})); // YP face, all cz
    io.model.thermal_boundaries.push_back(boundary);

    io.model.default_thermal_boundary = mhs::model::NeumannBoundary {};

    auto model = test_pta::build_model(io);

    const auto ny = (io.model.mesh.y_vertices.size() - 1);
    const auto nz = (io.model.mesh.z_vertices.size() - 1);
    const auto nx = (io.model.mesh.x_vertices.size() - 1);

    // Every exposed face of every cell on the domain boundary (ix=0 XP/XP, etc.)
    // and specifically the three faces the keys target must be Convection.
    auto check_face = [&](std::size_t ix, std::size_t iy, std::size_t iz, FaceDir dir) {
        const auto idx = ix * ny * nz + iy * nz + iz;
        const auto compact = input_cell(model, io.model.mesh, idx);
        EXPECT_NE(find_condition<Convection>(model, compact, dir), nullptr)
            << "Face " << (int)dir << " of cell (" << ix << "," << iy << "," << iz
            << ") should be Convection from one of the boundary regions";
    };

    // X|E|10 -> XP face at x=10mm -> cells with ix=nx-1=1
    for (int iy = 0; iy < ny; ++iy)
        for (int iz = 0; iz < nz; ++iz)
            check_face(nx - 1, iy, iz, FaceDir::XP);

    // X|E|0 -> XM face at x=0mm -> cells with ix=0
    for (int iy = 0; iy < ny; ++iy)
        for (int iz = 0; iz < nz; ++iz)
            check_face(0, iy, iz, FaceDir::XM);

    // Y|E|10 -> YP face at y=10mm -> cells with iy=ny-1=1
    for (int ix = 0; ix < nx; ++ix)
        for (int iz = 0; iz < nz; ++iz)
            check_face(ix, ny - 1, iz, FaceDir::YP);
}

// A block rectangle that cuts through a control volume splits that cell into
// pieces, and a piece can lie entirely outside every block (the mesh spans more
// than the block).  Such a piece contributes no source projection; it must not
// be attributed to a block, which previously indexed layers[-1] and crashed.
TEST(PreprocessorTest, PartialBlockCoverageSkipsPiecesOutsideEveryBlock)
{
    mhs::app::pta::Definition io;
    io.model.settings.study_type = mhs::model::StudyType::Steady;
    io.length_unit = mhs::app::pta::LengthUnit::Millimeter;
    io.model.settings.initial_temperature = 300.0;

    // 4 mm x 4 mm mesh, one cell per direction; the block covers 0..3 mm, so the
    // cell centre (2 mm) is inside a block and the cell is active, while the cut
    // at 3 mm leaves a 3..4 mm piece that belongs to no block.
    io.model.mesh.x_vertices = {0.0, 4.0};
    io.model.mesh.y_vertices = {0.0, 4.0};
    io.model.mesh.z_vertices = {0.0, 1.0};

    mhs::app::pta::LayerSpec layer;
    layer.thickness = "1";

    mhs::app::pta::BlockSpec block;
    block.material = "silicon";
    block.volumetric_heat_source = "1e7";

    mhs::app::pta::RectOperation rect;
    rect.operation = mhs::model::GeometryOperation::Add;
    rect.rect.x = "0";
    rect.rect.y = "0";
    rect.rect.width = "3";
    rect.rect.height = "3";
    block.geometry.push_back(rect);
    layer.blocks.push_back(block);
    io.layers.push_back(layer);

    mhs::model::MaterialSpec silicon;
    silicon.conductivity_x = silicon.conductivity_y = silicon.conductivity_z = "130";
    silicon.density = "2330";
    silicon.specific_heat = "700";
    io.model.materials.push_back({"silicon", silicon});

    io.model.default_thermal_boundary = mhs::model::NeumannBoundary {};

    const auto model = test_pta::build_model(io);
    ASSERT_EQ(model.discrete.cells.size(), 1u);

    // Only the covered piece (3 mm x 3 mm x 1 mm) carries the source.
    double total = 0.0;
    for (const auto& term : model.source_terms)
        total += term.weight;
    EXPECT_NEAR(total, 9e-9, 1e-18);
    EXPECT_EQ(model.source_terms.size(), 1u);
}
