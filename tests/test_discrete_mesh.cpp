#include "compiler/geometry_compiler.hpp"
#include "compiler/model_compiler.hpp"
#include "core/vector.hpp"
#include "model/discrete_mesh.hpp"
#include "solver/assembler.hpp"
#include "solver/thermal_physics.hpp"
#include "solver/thermal_reconstruction.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>

using namespace mhs::core;
using namespace mhs::model;

struct CubeDefinition : mhs::model::ModelDefinition {
    mhs::model::MeshSpec mesh;
};

static CubeDefinition cube_definition()
{
    CubeDefinition d;
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 0.5, 1};
    mhs::model::MaterialSpec mat;
    mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "1";
    mat.density = mat.specific_heat = "1";
    d.materials.push_back({"solid", mat});
    d.sources = {"0"};
    d.volumes.push_back({0, 0, {{mhs::model::GeometryOperation::Add, {0, 0, 0}, {1, 1, 1}}}});
    return d;
}

static mhs::model::Model build_fixture(const CubeDefinition& d) { return mhs::compiler::build_model(d, d.mesh); }

TEST(DiscreteMesh, CartesianTopologyHasUniqueFacesAndClosedCellIncidence)
{
    auto definition = cube_definition();
    const auto model = build_fixture(definition);
    const auto& mesh = model.discrete;
    ASSERT_EQ(mesh.cells.size(), 8u);
    ASSERT_EQ(mesh.facets.size(), 36u);
    ASSERT_EQ(model.cell_material.size(), 8u);
    std::size_t internal = 0;
    for (const auto& face : mesh.facets) {
        EXPECT_GT(face.area(), 0);
        if (!face.exterior())
            ++internal;
    }
    EXPECT_EQ(internal, 12u);
    for (std::size_t c = 0; c < mesh.cells.size(); ++c) {
        Vec3 closure {};
        double surface = 0;
        for (const auto id : mesh.incident_facets(static_cast<CellId>(c))) {
            const auto& face = mesh.facets[id];
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
        const auto model = build_fixture(d);
        ASSERT_EQ(model.discrete.facets.size(), 6u);
        const mhs::core::Vector<double> state {350};
        for (std::size_t id = 0; id < model.discrete.facets.size(); ++id) {
            const auto& facet = model.discrete.facets[id];
            const auto& condition = model.boundary_conditions[model.facet_thermal_condition[id]];
            const auto center = facet.center();
            const double d = model.discrete.cells[facet.owner()].size()[static_cast<std::size_t>(facet.axis)] / 2;
            const mhs::expression::FieldContext cell_context {center[0], center[1], center[2], state[0], 0};
            const auto trace = mhs::solver::evaluate_boundary(condition, cell_context, facet.area(), facet.area() / d);
            const mhs::expression::FieldContext ctx {center[0], center[1], center[2], trace.temperature, 0};
            const double conductive_flux = 2 * (state[0] - trace.temperature);
            if (std::holds_alternative<mhs::model::Convection>(condition)) {
                const auto& law = std::get<mhs::model::Convection>(condition);
                const double h = law.coefficient.eval(ctx);
                const double ambient = law.ambient_temperature.eval(ctx);
                EXPECT_NEAR(conductive_flux, h * (trace.temperature - ambient), 1e-4);
                const double condensed = 2 * h / (2 + h);
                EXPECT_NEAR(trace.conductance, condensed, 1e-9);
                EXPECT_NEAR(trace.load, condensed * ambient, 1e-9);
            }
            else {
                const auto& law = std::get<mhs::model::Neumann>(condition);
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
    EXPECT_THROW(build_fixture(d), std::invalid_argument);
}

TEST(DiscreteMesh, PartialBoundaryIntegratesActualOverlap)
{
    auto d = cube_definition();
    mhs::model::ThermalBoundarySpec bc;
    bc.condition = mhs::model::NeumannBoundary {"10"};
    bc.regions.push_back({mhs::model::Axis::X, 1.0, {0.1, 0.2, 0.1, 0.2}});
    d.thermal_boundaries.push_back(bc);
    const auto base = build_fixture(cube_definition()).discrete;
    const auto model = build_fixture(d);
    // Boundary-condition rectangles refine the canonical geometry itself.
    ASSERT_GT(model.discrete.facets.size(), base.facets.size());
    EXPECT_EQ(model.discrete.cells.size(), base.cells.size());
    auto exterior_area = [](const auto& mesh) {
        double total = 0.0;
        for (const auto& facet : mesh.facets)
            if (facet.exterior())
                total += facet.area();
        return total;
    };
    EXPECT_NEAR(exterior_area(model.discrete), exterior_area(base), 1e-14);
    mhs::core::Vector<int> incidence(model.discrete.facets.size(), 0);
    for (std::size_t c = 0; c < model.discrete.cells.size(); ++c)
        for (const auto id : model.discrete.incident_facets(static_cast<CellId>(c))) {
            ASSERT_LT(id, model.discrete.facets.size());
            const auto& facet = model.discrete.facets[id];
            EXPECT_TRUE(facet.minus == c || facet.plus == c);
            ++incidence[id];
        }
    for (std::size_t id = 0; id < incidence.size(); ++id)
        EXPECT_EQ(incidence[id], model.discrete.facets[id].exterior() ? 1 : 2);
    double assigned_area = 0;
    for (std::size_t id = 0; id < model.discrete.facets.size(); ++id) {
        const auto& facet = model.discrete.facets[id];
        if (!facet.exterior())
            continue;
        const auto& condition = model.boundary_conditions[model.facet_thermal_condition[id]];
        const auto& flux = std::get<mhs::model::Neumann>(condition).heat_flux;
        const auto center = facet.center();
        if (flux.eval({center[0], center[1], center[2], 300, 0}) == 10.0)
            assigned_area += facet.area();
    }
    EXPECT_NEAR(assigned_area, 0.01, 1e-14);
    const auto ops = mhs::solver::assemble_thermal(model, mhs::core::Vector<double>(8, 300), 0);
    EXPECT_NEAR(std::accumulate(ops.f.begin(), ops.f.end(), 0.0), 0.1, 1e-14);
}

TEST(DiscreteMesh, SourcesIntegrateAcrossConformalCells)
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

    // A source boundary inside one cell is rejected by the conformal compiler.
    EXPECT_THROW(build_fixture(d), std::invalid_argument);
    d.mesh.x_vertices = {0, 0.25, 1};
    const auto model = build_fixture(d);
    const auto ops = mhs::solver::assemble_thermal(model, mhs::core::Vector<double> {300, 300}, 0);
    EXPECT_NEAR(std::accumulate(ops.f.begin(), ops.f.end(), 0.0), 3.5, 1e-13);
    ASSERT_EQ(model.discrete.cells.size(), 2u);
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
    const auto model = build_fixture(d);
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

TEST(DiscreteMesh, MissingFacetReferencesRemainInvalid)
{
    DiscreteMesh mesh;
    mesh.cells.push_back({{0, 0, 0}, {1, 1, 1}});
    Facet face;
    EXPECT_EQ(face.minus, invalid_id);
    EXPECT_EQ(face.plus, invalid_id);
    face.minus = 0;
    mesh.facets.push_back(face);
    mhs::compiler::build_cell_facet_incidence(mesh);
    EXPECT_EQ(mesh.incident_facets(0).size(), 1u);
}

TEST(DiscreteMesh, InvalidSurfaceRegionsAreRejectedByCppCompiler)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const mhs::model::SurfaceRegion invalid[] {
        {Axis::X, nan, {0, 1, 0, 1}},
        {Axis::X, 0, {0, nan, 0, 1}},
        {Axis::X, 0, {1, 0, 0, 1}},
        // Deliberately exercise rejection of an unnamed value of this uint8_t enum.
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        {static_cast<Axis>(3), 0, {0, 1, 0, 1}},
    };
    for (const auto& region : invalid) {
        auto thermal = cube_definition();
        thermal.thermal_boundaries.push_back({{region}, mhs::model::NeumannBoundary {"0"}});
        EXPECT_THROW(build_fixture(thermal), std::invalid_argument);
        auto fluid = cube_definition();
        fluid.fluid_boundaries.push_back({{region}, mhs::model::PressureOutlet {0}});
        EXPECT_THROW(build_fixture(fluid), std::invalid_argument);
    }
}
