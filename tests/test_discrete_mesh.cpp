#include "compiler/model_compiler.hpp"
#include "core/discrete_mesh.hpp"
#include "solver/assembler.hpp"
#include "solver/postprocessor.hpp"
#include "solver/interpolation.hpp"
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

TEST(DiscreteMesh, RejectsBrokenIncidenceAndInvalidDofPermutation)
{
    auto model = mhs::sim::build_model(cube_definition());
    model.discrete.faces[0].area = -1;
    EXPECT_THROW(model.discrete.validate(), std::invalid_argument);
    EXPECT_THROW(model.dofs.set_permutation({0, 0, 2, 3, 4, 5, 6, 7}), std::invalid_argument);
    model.dofs.set_permutation({7, 6, 5, 4, 3, 2, 1, 0});
    EXPECT_EQ(model.dofs.temperature(CellId {0}).value, 7u);
    EXPECT_EQ(model.dofs.cell(DofId {7}).value, 0u);
}

TEST(DiscreteMesh, ThermalOperatorsRespectIndependentDofOrdering)
{
    auto model = mhs::sim::build_model(cube_definition());
    model.material_table[0].kx = mhs::core::parse("1+T/1000", {});
    model.heat_source_table[0] = mhs::core::parse("x+2*y+3*z", {});
    std::vector<double> original {300, 310, 320, 330, 340, 350, 360, 370};
    const auto baseline = mhs::sim::assemble_thermal(model, original, 0);
    const std::vector<Index> permutation {7, 5, 3, 1, 6, 4, 2, 0};
    model.dofs.set_permutation(permutation);
    std::vector<double> reordered(8);
    for (Index c = 0; c < 8; ++c)
        reordered[permutation[c]] = original[c];
    const auto result = mhs::sim::assemble_thermal(model, reordered, 0);
    for (Index i = 0; i < 8; ++i) {
        EXPECT_NEAR(result.f[permutation[i]], baseline.f[i], 1e-14);
        for (Index j = 0; j < 8; ++j) {
            EXPECT_NEAR(result.K.coeff(permutation[i], permutation[j]), baseline.K.coeff(i, j), 1e-14);
            EXPECT_NEAR(result.C.coeff(permutation[i], permutation[j]), baseline.C.coeff(i, j), 1e-14);
        }
    }
}

TEST(DiscreteMesh, PartialBoundaryIntegratesActualOverlap)
{
    auto d = cube_definition();
    mhs::model::BoundaryPatch bc;
    bc.condition = mhs::model::NeumannBoundary {"10"};
    bc.regions.push_back({mhs::model::Axis::X, 1.0, {{0.1, 0.2, 0.1, 0.2}}});
    d.boundaries.push_back(bc);
    const auto model = mhs::sim::build_model(d);
    const auto ops = mhs::sim::assemble_thermal(model, std::vector<double>(8, 300), 0);
    EXPECT_NEAR(ops.f.sum(), 0.1, 1e-14);
    EXPECT_NO_THROW(model.discrete.validate());
}

TEST(DiscreteMesh, ExternalMeshNeedsNoCartesianAxes)
{
    auto definition=cube_definition();
    auto original=mhs::sim::build_model(definition);
    definition.mesh={};
    const auto imported=mhs::sim::build_model(definition,original.discrete);
    EXPECT_EQ(imported.discrete.cells.size(),8u);
    EXPECT_EQ(imported.discrete.faces.size(),36u);
    EXPECT_EQ(imported.output.cell_vertices.size(),8u);
    EXPECT_EQ(imported.output.points.size(),27u);
    EXPECT_EQ(imported.dofs.size(),8u);
}

TEST(DiscreteMesh, PointAndVertexSamplingUseDofMap)
{
    auto model=mhs::sim::build_model(cube_definition());
    const std::vector<double> temperature{300,310,320,330,340,350,360,370};
    const auto vertices=mhs::post::interpolate_cell_to_node(model,temperature,0);
    const Vec3 point{0.2,0.3,0.4};
    const auto cell=model.discrete.locate(point);
    const auto sample=mhs::utils::sample_point(model,temperature,0,cell,point);
    const std::vector<Index> permutation{7,5,3,1,6,4,2,0};
    model.dofs.set_permutation(permutation);
    std::vector<double> reordered(8);
    for(Index c=0;c<8;++c) reordered[permutation[c]]=temperature[c];
    const auto result=mhs::post::interpolate_cell_to_node(model,reordered,0);
    for(Index v=0;v<vertices.size();++v) EXPECT_NEAR(result[v],vertices[v],1e-12);
    EXPECT_NEAR(mhs::utils::sample_point(model,reordered,0,cell,point),sample,1e-12);
}
