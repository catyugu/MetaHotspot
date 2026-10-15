#include "compiler/model_compiler.hpp"
#include "core/discrete_mesh.hpp"
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
    d.default_boundary = mhs::model::NeumannBoundary{};
    return d;
}

TEST(DiscreteMesh, CompilerEmitsUniqueFacesAndClosedCellIncidence)
{
    const auto model = mhs::sim::build_model(cube_definition());
    const auto& mesh = model.discrete;
    ASSERT_EQ(mesh.cells.size(), 8u);
    ASSERT_EQ(mesh.faces.size(), 36u);
    ASSERT_EQ(model.dofs.cell_to_dof.size(), 8u);
    Index internal = 0;
    for (const auto& face : mesh.faces) {
        EXPECT_GT(face.area, 0);
        if (face.neighbor.valid()) ++internal;
    }
    EXPECT_EQ(internal, 12u);
    for (Index c = 0; c < mesh.cells.size(); ++c) {
        Vec3 closure{};
        double surface = 0;
        for (const auto id : mesh.incident_faces(CellId{c})) {
            const auto& face = mesh.faces[id.value];
            const double sign = face.owner.value == c ? face.sign : -face.sign;
            closure[face.axis] += sign * face.area;
            surface += face.area;
        }
        EXPECT_DOUBLE_EQ(mesh.cells[c].volume(), 0.125);
        EXPECT_DOUBLE_EQ(surface, 1.5);
        for (auto v : closure) EXPECT_NEAR(v, 0, 1e-14);
    }
    EXPECT_NO_THROW(mesh.validate());
}

TEST(DiscreteMesh, RejectsBrokenIncidenceAndInvalidDofPermutation)
{
    auto model = mhs::sim::build_model(cube_definition());
    model.discrete.faces[0].area = -1;
    EXPECT_THROW(model.discrete.validate(), std::invalid_argument);
    EXPECT_THROW(model.dofs.set_permutation({0,0,2,3,4,5,6,7}), std::invalid_argument);
    model.dofs.set_permutation({7,6,5,4,3,2,1,0});
    EXPECT_EQ(model.dofs.temperature(CellId{0}).value, 7u);
    EXPECT_EQ(model.dofs.cell(DofId{7}).value, 0u);
}
