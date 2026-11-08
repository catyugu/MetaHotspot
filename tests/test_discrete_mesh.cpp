#include "compiler/geometry_compiler.hpp"
#include "compiler/model_compiler.hpp"
#include "core/vector.hpp"
#include "model/discrete_mesh.hpp"
#include "solver/assembler.hpp"
#include "solver/thermal_reconstruction.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

using namespace mhs::core;
using namespace mhs::model;

static mhs::model::ModelDefinition cube_definition()
{
    mhs::model::ModelDefinition d;
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 0.5, 1};
    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "1";
    mat.density = mat.specific_heat = "1";
    d.materials.push_back({"solid", mat});
    d.sources = {"0"};
    d.volumes.push_back({0, 0, {{mhs::model::GeometryOperation::Add, {0, 0, 0}, {1, 1, 1}}}});
    return d;
}

TEST(DiscreteMesh, CartesianTopologyHasUniqueFacesAndClosedCellIncidence)
{
    auto definition = cube_definition();
    const auto model = mhs::compiler::build_model(definition);
    const auto& mesh = model.discrete;
    ASSERT_EQ(mesh.cells.size(), 8u);
    ASSERT_EQ(mesh.faces.size(), 36u);
    ASSERT_EQ(model.cell_material.size(), 8u);
    std::size_t internal = 0;
    for (const auto& face : mesh.faces) {
        EXPECT_GT(face.area(), 0);
        if (!face.exterior())
            ++internal;
    }
    EXPECT_EQ(internal, 12u);
    for (std::size_t c = 0; c < mesh.cells.size(); ++c) {
        Vec3 closure {};
        double surface = 0;
        for (const auto id : mesh.incident_faces(static_cast<CellId>(c))) {
            const auto& face = mesh.faces[id];
            const double sign = (face.minus == c ? 1 : -1);
            closure[static_cast<std::size_t>(face.axis)] += sign * face.area();
            surface += face.area();
        }
        EXPECT_DOUBLE_EQ(mesh.cells[c].volume(), 0.125);
        EXPECT_DOUBLE_EQ(surface, 1.5);
        for (auto v : closure)
            EXPECT_NEAR(v, 0, 1e-14);
    }
}

TEST(DiscreteMesh, TemperatureDependentBoundaryTracesSolveTheSurfaceLaw)
{
    for (const auto& condition : {mhs::model::ThermalBoundary {mhs::model::ConvectionBoundary {"10+0.01*T", "300"}},
             mhs::model::ThermalBoundary {mhs::model::NeumannBoundary {"1+0.001*T"}}}) {
        auto d = cube_definition();
        d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
        d.default_thermal_boundary = condition;
        const auto model = mhs::compiler::build_model(d);
        ASSERT_EQ(model.boundary_faces.size(), 6u);
        // Boundary sampling looks up terms by cell.
        EXPECT_TRUE(std::ranges::is_sorted(model.boundary_faces, {}, [](const auto& patch) { return patch.cell; }));
        const mhs::core::Vector<double> state {350};
        for (const auto& patch : model.boundary_faces) {
            // Unit conductivity, unit area and half-cell distance give g = 2 W/K;
            // T-dependent laws are evaluated at the surface trace, not at the cell.
            const auto center = mhs::solver::boundary_center(model.discrete.cells[patch.cell], patch);
            const auto trace = mhs::solver::evaluate_boundary(model.boundary_conditions[patch.condition], center,
                ((patch.bounds[1] - patch.bounds[0]) * (patch.bounds[3] - patch.bounds[2])),
                (model.discrete.cells[patch.cell].size()[static_cast<std::size_t>(patch.axis)] / 2), state[0], 1.0, 0);
            const mhs::expression::FieldContext ctx {center[0], center[1], center[2], trace.temperature, 0};
            const double conductive_flux = 2 * (state[0] - trace.temperature);
            if (std::holds_alternative<mhs::model::Convection>(model.boundary_conditions[patch.condition])) {
                const auto& law = std::get<mhs::model::Convection>(model.boundary_conditions[patch.condition]);
                const double h = law.coefficient.eval(ctx);
                const double ambient = law.ambient_temperature.eval(ctx);
                EXPECT_NEAR(conductive_flux, h * (trace.temperature - ambient), 1e-4);
                const double condensed = 2 * h / (2 + h);
                EXPECT_NEAR(trace.conductance, condensed, 1e-9);
                EXPECT_NEAR(trace.load, condensed * ambient, 1e-9);
            }
            else {
                const auto& law = std::get<mhs::model::Neumann>(model.boundary_conditions[patch.condition]);
                // Positive q is inward: the trace conducts q out of the cell.
                EXPECT_NEAR(conductive_flux, -law.heat_flux.eval(ctx), 1e-4);
                EXPECT_DOUBLE_EQ(trace.conductance, 0.0);
                EXPECT_NEAR(trace.load, -conductive_flux, 1e-5);
            }
            // Sampling and assembly read the same surface trace.
            EXPECT_DOUBLE_EQ(
                mhs::solver::sample_point(model, state, 0, static_cast<CellId>(0), center), trace.temperature);
        }
    }
}

TEST(DiscreteMesh, TemperatureDependentDirichletTemperatureIsRejected)
{
    auto d = cube_definition();
    d.thermal_boundaries = {{{{mhs::model::Axis::X, 1, {0, 1, 0, 1}}}, mhs::model::DirichletBoundary {"300 + 0.1*T"}}};
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
}

TEST(DiscreteMesh, PartialBoundaryIntegratesActualOverlap)
{
    auto d = cube_definition();
    mhs::model::ThermalBoundarySpec bc;
    bc.condition = mhs::model::NeumannBoundary {"10"};
    bc.regions.push_back({mhs::model::Axis::X, 1.0, {0.1, 0.2, 0.1, 0.2}});
    d.thermal_boundaries.push_back(bc);
    const auto base = mhs::compiler::build_model(cube_definition()).discrete;
    const auto model = mhs::compiler::build_model(d);
    ASSERT_EQ(model.discrete.faces.size(), base.faces.size());
    EXPECT_EQ(model.discrete.cell_face_offsets, base.cell_face_offsets);
    EXPECT_EQ(model.discrete.cell_face_ids, base.cell_face_ids);
    for (std::size_t i = 0; i < base.faces.size(); ++i) {
        EXPECT_EQ(model.discrete.faces[i].minus, base.faces[i].minus);
        EXPECT_EQ(model.discrete.faces[i].plus, base.faces[i].plus);
        EXPECT_EQ(model.discrete.faces[i].bounds, base.faces[i].bounds);
    }
    const auto ops = mhs::solver::assemble_thermal(model, mhs::core::Vector<double>(8, 300), 0);
    EXPECT_NEAR(std::accumulate(ops.f.begin(), ops.f.end(), 0.0), 0.1, 1e-14);
}

TEST(DiscreteMesh, SourcesIntegrateAcrossPhysicalRegionsWithinOneCell)
{
    auto d = cube_definition();
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
    auto& first = d.volumes[0];
    first.geometry = {{mhs::model::GeometryOperation::Add, {0, 0, 0}, {0.25, 1, 1}}};
    d.sources[0] = "4*z";
    auto second = first;
    second.geometry = {{mhs::model::GeometryOperation::Add, {0.25, 0, 0}, {1, 1, 1}}};
    d.sources.push_back("8*z");
    second.source = 1;
    d.volumes.push_back(second);
    const auto model = mhs::compiler::build_model(d);
    const auto ops = mhs::solver::assemble_thermal(model, mhs::core::Vector<double> {300}, 0);
    EXPECT_NEAR(std::accumulate(ops.f.begin(), ops.f.end(), 0.0), 3.5, 1e-13);
    ASSERT_EQ(model.source_terms.size(), 2u);
    EXPECT_NEAR(model.source_terms[0].weight, 0.25, 1e-14);
    EXPECT_NEAR(model.source_terms[1].weight, 0.75, 1e-14);
}

TEST(DiscreteMesh, PartialDirichletUsesProjectedAreaForAssemblyAndSampling)
{
    auto d = cube_definition();
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
    mhs::model::ThermalBoundarySpec bc;
    bc.condition = mhs::model::DirichletBoundary {"400"};
    bc.regions.push_back({mhs::model::Axis::X, 1, {0.1, 0.2, 0.1, 0.2}});
    d.thermal_boundaries.push_back(bc);
    const auto model = mhs::compiler::build_model(d);
    const mhs::core::Vector<double> state {300};
    const auto ops = mhs::solver::assemble_thermal(model, state, 0);
    EXPECT_NEAR(ops.K.coeff(0, 0), 0.02, 1e-14);
    EXPECT_NEAR(ops.f[0], 8, 1e-12);
    // The projected Dirichlet area raises the reconstructed x-gradient to 1 K/m,
    // so an interior probe moves at exactly that rate along x.
    const auto left = mhs::solver::sample_point(model, state, 0, static_cast<CellId>(0), {0.2, 0.5, 0.5});
    const auto right = mhs::solver::sample_point(model, state, 0, static_cast<CellId>(0), {0.8, 0.5, 0.5});
    EXPECT_NEAR((right - left) / 0.6, 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(mhs::solver::sample_point(model, state, 0, static_cast<CellId>(0), {1, 0.15, 0.15}), 400);
    EXPECT_DOUBLE_EQ(mhs::solver::sample_point(model, state, 0, static_cast<CellId>(0), {1, 0.8, 0.8}), 300);
}

TEST(DiscreteMesh, MissingReferencesRemainInvalid)
{
    DiscreteMesh mesh;
    mesh.cells.push_back({{0, 0, 0}, {1, 1, 1}});
    FacePatch face;
    EXPECT_EQ(face.minus, invalid_id);
    EXPECT_EQ(face.plus, invalid_id);
    face.minus = 0;
    mesh.faces.push_back(face);
    mhs::compiler::build_cell_face_incidence(mesh);
    EXPECT_EQ(mesh.incident_faces(0).size(), 1u);
    EXPECT_EQ(mhs::solver::locate_cell(mesh, {2, 2, 2}), invalid_id);
    EXPECT_EQ(mhs::solver::locate_cell(mesh, {0.5, 0.5, 0.5}), 0u);
}
