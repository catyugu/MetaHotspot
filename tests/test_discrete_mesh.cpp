#include "compiler/model_compiler.hpp"
#include "core/discrete_mesh.hpp"
#include "solver/assembler.hpp"
#include "solver/interpolation.hpp"
#include "solver/postprocessor.hpp"
#include <gtest/gtest.h>
#include <numeric>

using namespace mhs::core;

static mhs::model::ModelDefinition cube_definition()
{
    mhs::model::ModelDefinition d;
    d.settings.length_unit = mhs::model::LengthUnit::Meter;
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 0.5, 1};
    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "1";
    mat.density = mat.specific_heat = "1";
    d.materials.push_back({"solid", mat});
    mhs::model::LayerSpec layer;
    layer.thickness = "1";
    mhs::model::BlockSpec block;
    block.material = "solid";
    block.volumetric_heat_source = "0";
    block.geometry.push_back({mhs::model::GeometryOperation::Add, {"0", "0", "1", "1"}});
    layer.blocks.push_back(block);
    d.layers.push_back(layer);
    d.default_boundary = mhs::model::NeumannBoundary {};
    return d;
}

TEST(DiscreteMesh, ImportedTopologyHasUniqueFacesAndClosedCellIncidence)
{
    auto definition = cube_definition();
    const auto supplied = mhs::sim::build_model(definition).discrete;
    definition.mesh = {};
    const auto model = mhs::sim::build_model(definition, supplied);
    const auto& mesh = model.discrete;
    ASSERT_EQ(mesh.cells.size(), 8u);
    ASSERT_EQ(mesh.faces.size(), 36u);
    ASSERT_EQ(model.cell_material.size(), 8u);
    Index internal = 0;
    for (const auto& face : mesh.faces) {
        EXPECT_GT(face.area, 0);
        if (face.neighbor.valid())
            ++internal;
    }
    EXPECT_EQ(internal, 12u);
    for (Index c = 0; c < mesh.cells.size(); ++c) {
        Vec3 closure {};
        double surface = 0;
        for (const auto id : mesh.incident_faces(CellId {c})) {
            const auto& face = mesh.faces[id.value];
            const double sign = face.owner.value == c ? face.sign : -face.sign;
            closure[face.axis] += sign * face.area;
            surface += face.area;
        }
        EXPECT_DOUBLE_EQ(mesh.cells[c].volume(), 0.125);
        EXPECT_DOUBLE_EQ(surface, 1.5);
        for (auto v : closure)
            EXPECT_NEAR(v, 0, 1e-14);
    }
    EXPECT_NO_THROW(mesh.validate());
}

TEST(DiscreteMesh, OperatorsAndSamplingRespectArbitraryCellOrdering)
{
    auto definition = cube_definition();
    auto model = mhs::sim::build_model(definition);
    const std::vector<Index> order {7, 5, 3, 1, 6, 4, 2, 0};
    auto mesh = model.discrete;
    std::vector<Index> inverse(order.size());
    for (Index i = 0; i < order.size(); ++i) {
        mesh.cells[i] = model.discrete.cells[order[i]];
        inverse[order[i]] = i;
    }
    for (auto& face : mesh.faces) {
        face.owner.value = inverse[face.owner.value];
        if (face.neighbor.valid())
            face.neighbor.value = inverse[face.neighbor.value];
    }
    auto imported = mhs::sim::build_model(definition, mesh);
    for (auto* m : {&model, &imported}) {
        m->material_table[0].kx = mhs::core::parse("1+T/1000", {});
        m->heat_source_table[0] = mhs::core::parse("x+2*y+3*z", {});
    }
    std::vector<double> original {300, 310, 320, 330, 340, 350, 360, 370};
    std::vector<double> reordered(8);
    for (Index i = 0; i < order.size(); ++i)
        reordered[i] = original[order[i]];
    const auto baseline = mhs::sim::assemble_thermal(model, original, 0);
    const auto result = mhs::sim::assemble_thermal(imported, reordered, 0);
    for (Index i = 0; i < 8; ++i) {
        EXPECT_NEAR(result.f[i], baseline.f[order[i]], 1e-14);
        for (Index j = 0; j < 8; ++j) {
            EXPECT_NEAR(result.K.coeff(i, j), baseline.K.coeff(order[i], order[j]), 1e-14);
            EXPECT_NEAR(result.C.coeff(i, j), baseline.C.coeff(order[i], order[j]), 1e-14);
        }
    }
    const auto vertices = mhs::post::interpolate_cell_to_node(model, original, 0);
    const auto nodes = mhs::post::interpolate_cell_to_node(imported, reordered, 0);
    for (Index v = 0; v < imported.output.points.size(); ++v) {
        auto found = std::find(model.output.points.begin(), model.output.points.end(), imported.output.points[v]);
        ASSERT_NE(found, model.output.points.end());
        EXPECT_NEAR(nodes[v], vertices[found - model.output.points.begin()], 1e-12);
    }
    const Vec3 point {0.2, 0.3, 0.4};
    EXPECT_NEAR(mhs::utils::sample_point(model, original, 0, model.discrete.locate(point), point),
        mhs::utils::sample_point(imported, reordered, 0, imported.discrete.locate(point), point), 1e-12);
}

TEST(DiscreteMesh, PartialBoundaryIntegratesActualOverlap)
{
    auto d = cube_definition();
    mhs::model::BoundaryPatch bc;
    bc.condition = mhs::model::NeumannBoundary {"10"};
    bc.regions.push_back({mhs::model::Axis::X, 1.0, {{0.1, 0.2, 0.1, 0.2}}});
    d.boundaries.push_back(bc);
    const auto base = mhs::sim::build_model(cube_definition()).discrete;
    const auto model = mhs::sim::build_model(d);
    ASSERT_EQ(model.discrete.faces.size(), base.faces.size());
    EXPECT_EQ(model.discrete.cell_face_offsets, base.cell_face_offsets);
    EXPECT_EQ(model.discrete.cell_face_ids, base.cell_face_ids);
    for (Index i = 0; i < base.faces.size(); ++i) {
        EXPECT_EQ(model.discrete.faces[i].min, base.faces[i].min);
        EXPECT_EQ(model.discrete.faces[i].max, base.faces[i].max);
    }
    const auto ops = mhs::sim::assemble_thermal(model, std::vector<double>(8, 300), 0);
    EXPECT_NEAR(ops.f.sum(), 0.1, 1e-14);
    EXPECT_NO_THROW(model.discrete.validate());
}

TEST(DiscreteMesh, ImportedMeshRejectsEmptyOrDuplicatedClosedShell)
{
    auto mesh = mhs::sim::build_model(cube_definition()).discrete;
    const auto original = mesh.faces;
    mesh.faces.clear();
    mesh.build_incidence();
    EXPECT_THROW(mesh.validate(), std::invalid_argument);
    mesh.faces = original;
    mesh.faces.insert(mesh.faces.end(), original.begin(), original.end());
    mesh.build_incidence();
    EXPECT_THROW(mesh.validate(), std::invalid_argument);
}

TEST(DiscreteMesh, SourcesIntegrateAcrossPhysicalRegionsWithinOneCell)
{
    auto d = cube_definition();
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
    auto& first = d.layers[0].blocks[0];
    first.geometry = {{mhs::model::GeometryOperation::Add, {"0", "0", "0.25", "1"}}};
    first.volumetric_heat_source = "4*z";
    auto second = first;
    second.geometry = {{mhs::model::GeometryOperation::Add, {"0.25", "0", "0.75", "1"}}};
    second.volumetric_heat_source = "8*z";
    d.layers[0].blocks.push_back(second);
    const auto model = mhs::sim::build_model(d);
    const auto ops = mhs::sim::assemble_thermal(model, std::vector<double> {300}, 0);
    EXPECT_NEAR(ops.f.sum(), 3.5, 1e-13);
    ASSERT_EQ(model.source_terms.size(), 2u);
    EXPECT_NEAR(model.source_terms[0].weight, 0.25, 1e-14);
    EXPECT_NEAR(model.source_terms[1].weight, 0.75, 1e-14);
    auto other = d.materials[0];
    other.name = "other";
    other.value.conductivity_x = "2";
    d.materials.push_back(other);
    d.layers[0].blocks[1].material = "other";
    EXPECT_THROW(mhs::sim::build_model(d), std::invalid_argument);
}

TEST(DiscreteMesh, ExternalImportRejectsOverlappingControlVolumes)
{
    auto definition = cube_definition();
    definition.mesh.x_vertices = definition.mesh.y_vertices = definition.mesh.z_vertices = {0, 1};
    auto mesh = mhs::sim::build_model(definition).discrete;
    const auto faces = mesh.faces;
    mesh.cells.push_back(mesh.cells.front());
    for (auto face : faces) {
        face.owner = CellId {1};
        mesh.faces.push_back(face);
    }
    mesh.build_incidence();
    ASSERT_NO_THROW(mesh.validate());
    EXPECT_THROW(mhs::sim::build_model(definition, mesh), std::invalid_argument);
}
