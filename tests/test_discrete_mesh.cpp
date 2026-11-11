#include "compiler/geometry_compiler.hpp"
#include "compiler/model_compiler.hpp"
#include "core/vector.hpp"
#include "model/mesh.hpp"
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
    mhs::model::ConformalMeshSpec mesh;
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

TEST(Mesh, ConformalTopologyHasImplicitConnectionsAndClosedCells)
{
    const auto model = build_fixture(cube_definition());
    const auto& mesh = std::get<ConformalMesh>(model.mesh);
    EXPECT_EQ(cell_count(mesh), 8u);
    EXPECT_EQ(model.boundary_facets.size(), 24u);
    EXPECT_TRUE(mesh.logical_to_cell.empty());
    EXPECT_TRUE(mesh.cell_to_logical.empty());
    Vector<Vec3> closure(8);
    Vector<double> surface(8, 0);
    std::size_t internal = 0;
    for_each_connection(mesh, [&](Connection c) {
        const auto a = cell_geometry(mesh, c.a), b = cell_geometry(mesh, c.b);
        const auto face = connection_geometry(a, b);
        const auto axis = static_cast<std::size_t>(face.axis);
        closure[c.a][axis] += face.area();
        closure[c.b][axis] -= face.area();
        surface[c.a] += face.area();
        surface[c.b] += face.area();
        ++internal;
    });
    EXPECT_EQ(internal, 12u);
    for (const auto& f : model.boundary_facets) {
        closure[f.cell][static_cast<std::size_t>(f.axis)] += f.side * f.area();
        surface[f.cell] += f.area();
    }
    for (std::size_t c = 0; c < 8; ++c) {
        EXPECT_DOUBLE_EQ(cell_geometry(mesh, static_cast<CellId>(c)).volume(), 0.125);
        EXPECT_DOUBLE_EQ(surface[c], 1.5);
        for (auto value : closure[c])
            EXPECT_NEAR(value, 0, 1e-14);
    }
}

TEST(Mesh, TemperatureDependentBoundaryTracesSolveTheSurfaceLaw)
{
    for (const auto& condition : {mhs::model::ThermalBoundary {mhs::model::ConvectionBoundary {"10+0.01*T", "300"}},
             mhs::model::ThermalBoundary {mhs::model::NeumannBoundary {"1+0.001*T"}}}) {
        auto d = cube_definition();
        d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
        d.default_thermal_boundary = condition;
        const auto model = build_fixture(d);
        ASSERT_EQ(model.boundary_facets.size(), 6u);
        const mhs::core::Vector<double> state {350};
        for (std::size_t id = 0; id < model.boundary_facets.size(); ++id) {
            const auto& facet = model.boundary_facets[id];
            const auto& condition = model.boundary_conditions[model.boundary_thermal_condition[id]];
            const auto center = facet.center();
            const double d = cell_geometry(model.mesh, facet.cell).size()[static_cast<std::size_t>(facet.axis)] / 2;
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

TEST(Mesh, TemperatureDependentDirichletTemperatureIsRejected)
{
    auto d = cube_definition();
    d.thermal_boundaries = {{{{mhs::model::Axis::X, 1, {0, 1, 0, 1}}}, mhs::model::DirichletBoundary {"300 + 0.1*T"}}};
    EXPECT_THROW(build_fixture(d), std::invalid_argument);
}

TEST(Mesh, PartialBoundaryIntegratesActualOverlap)
{
    auto d = cube_definition();
    mhs::model::ThermalBoundarySpec bc;
    bc.condition = mhs::model::NeumannBoundary {"10"};
    bc.regions.push_back({mhs::model::Axis::X, 1.0, {0.1, 0.2, 0.1, 0.2}});
    d.thermal_boundaries.push_back(bc);
    const auto base = build_fixture(cube_definition());
    const auto model = build_fixture(d);
    // Boundary-condition rectangles refine the canonical geometry itself.
    ASSERT_GT(model.boundary_facets.size(), base.boundary_facets.size());
    EXPECT_EQ(model.cells.size(), base.cells.size());
    auto exterior_area = [](const auto& mesh) {
        double total = 0.0;
        for (const auto& facet : mesh.boundary_facets)
            total += facet.area();
        return total;
    };
    EXPECT_NEAR(exterior_area(model), exterior_area(base), 1e-14);
    double assigned_area = 0;
    for (std::size_t id = 0; id < model.boundary_facets.size(); ++id) {
        const auto& facet = model.boundary_facets[id];
        const auto& condition = model.boundary_conditions[model.boundary_thermal_condition[id]];
        const auto& flux = std::get<mhs::model::Neumann>(condition).heat_flux;
        const auto center = facet.center();
        if (flux.eval({center[0], center[1], center[2], 300, 0}) == 10.0)
            assigned_area += facet.area();
    }
    EXPECT_NEAR(assigned_area, 0.01, 1e-14);
    const auto ops = mhs::solver::assemble_thermal(model, mhs::core::Vector<double>(8, 300), 0);
    EXPECT_NEAR(std::accumulate(ops.f.begin(), ops.f.end(), 0.0), 0.1, 1e-14);
}

TEST(Mesh, SourcesIntegrateAcrossConformalCells)
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

    // Source quadrature may partition a cell without creating a material interface.
    const auto coarse = build_fixture(d);
    const auto coarse_ops = mhs::solver::assemble_thermal(coarse, Vector<double> {300}, 0);
    EXPECT_NEAR(coarse_ops.f[0], 3.5, 1e-13);
    d.mesh.x_vertices = {0, 0.25, 1};
    const auto model = build_fixture(d);
    const auto ops = mhs::solver::assemble_thermal(model, mhs::core::Vector<double> {300, 300}, 0);
    EXPECT_NEAR(std::accumulate(ops.f.begin(), ops.f.end(), 0.0), 3.5, 1e-13);
    ASSERT_EQ(model.cells.size(), 2u);
    ASSERT_EQ(model.source_terms.size(), 2u);
    EXPECT_NEAR(model.source_terms[0].weight, 0.25, 1e-14);
    EXPECT_NEAR(model.source_terms[1].weight, 0.75, 1e-14);
}

TEST(Mesh, PartialDirichletUsesProjectedAreaForAssemblyAndSampling)
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

TEST(Mesh, InvalidSurfaceRegionsAreRejectedByCppCompiler)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const mhs::model::SurfaceRegion invalid[] {
        {Axis::X, nan, {0, 1, 0, 1}},
        {Axis::X, 0, {0, nan, 0, 1}},
        {Axis::X, 0, {1, 0, 0, 1}},
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

TEST(Mesh, EffectiveGeometryAllowsHiddenCutsAndRejectsRealMaterialCuts)
{
    auto d = cube_definition();
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
    auto hidden = d.volumes[0];
    hidden.geometry[0].max[0] = 0.3;
    d.volumes.insert(d.volumes.begin(), hidden);
    EXPECT_NO_THROW(build_fixture(d));
    auto material = d.materials[0];
    d.materials.push_back(material);
    auto overlay = d.volumes.back();
    overlay.material = 1;
    overlay.geometry[0].max[0] = 0.6;
    d.volumes.push_back(overlay);
    EXPECT_THROW(build_fixture(d), std::invalid_argument);
}

TEST(Mesh, HolesHaveCompactActivityMapsAndExposedBoundary)
{
    auto d = cube_definition();
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 0.25, 0.75, 1};
    d.volumes[0].geometry.push_back({GeometryOperation::Subtract, {0.25, 0.25, 0.25}, {0.75, 0.75, 0.75}});
    const auto model = build_fixture(d);
    const auto& mesh = std::get<ConformalMesh>(model.mesh);
    ASSERT_EQ(cell_count(mesh), 26u);
    EXPECT_EQ(mesh.logical_to_cell.size(), 27u);
    EXPECT_EQ(mesh.cell_to_logical.size(), 26u);
    EXPECT_EQ(mesh.logical_to_cell[13], invalid_id);
    double area = 0;
    for (const auto& f : model.boundary_facets)
        area += f.area();
    EXPECT_NEAR(area, 7.5, 1e-14);
    const auto ops = mhs::solver::assemble_thermal(model, Vector<double>(26, 300), 0);
    for (std::size_t i = 0; i < 26; ++i) {
        double sum = 0;
        for (std::size_t j = 0; j < 26; ++j)
            sum += ops.K.coeff(i, j);
        EXPECT_NEAR(sum, 0, 1e-14);
    }
}

static NonConformalMesh coarse_fine_mesh()
{ return {{{{0, 0, 0}, {1, 2, 1}}, {{1, 0, 0}, {2, 1, 1}}, {{1, 1, 0}, {2, 2, 1}}}, {{0, 1}, {2, 0}, {1, 2}}}; }

TEST(Mesh, NonConformalCoarseFineContactsMatchIndependentMatrix)
{
    auto d = cube_definition();
    d.volumes[0].geometry[0].max = {2, 2, 1};
    const auto model = mhs::compiler::build_model(d, coarse_fine_mesh());
    ASSERT_TRUE(std::holds_alternative<NonConformalMesh>(model.mesh));
    const auto ops = mhs::solver::assemble_thermal(model, Vector<double> {300, 310, 320}, 0);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(ops.K.coeff(i, j), i == j ? 2 : -1);
            EXPECT_DOUBLE_EQ(ops.C.coeff(i, j), i == j ? (i == 0 ? 2 : 1) : 0);
        }
    double area = 0;
    for (const auto& f : model.boundary_facets)
        area += f.area();
    EXPECT_DOUBLE_EQ(area, 16);
    const auto gradients = mhs::solver::reconstruct_gradients(model, Vector<double>(3, 300), 0);
    for (const auto& gradient : gradients)
        for (auto g : gradient)
            EXPECT_NEAR(g, 0, 1e-13);
}

TEST(Mesh, NonConformalRejectsInvalidExplicitTopology)
{
    auto d = cube_definition();
    d.volumes[0].geometry[0].max = {2, 2, 1};
    auto missing = coarse_fine_mesh();
    missing.connections.pop_back();
    EXPECT_THROW(mhs::compiler::build_model(d, missing), std::invalid_argument);
    auto duplicate = coarse_fine_mesh();
    duplicate.connections.push_back({1, 0});
    EXPECT_THROW(mhs::compiler::build_model(d, duplicate), std::invalid_argument);
    auto invalid = coarse_fine_mesh();
    invalid.connections.push_back({0, 9});
    EXPECT_THROW(mhs::compiler::build_model(d, invalid), std::invalid_argument);
    invalid.connections.back() = {0, 0};
    EXPECT_THROW(mhs::compiler::build_model(d, invalid), std::invalid_argument);
    auto overlapping = coarse_fine_mesh();
    overlapping.cells.push_back({{0.2, 0.2, 0.2}, {0.8, 0.8, 0.8}});
    EXPECT_THROW(mhs::compiler::build_model(d, overlapping), std::invalid_argument);
    NonConformalMesh edge {{{{0, 0, 0}, {1, 1, 1}}, {{1, 1, 0}, {2, 2, 1}}}, {{0, 1}}};
    EXPECT_THROW(mhs::compiler::build_model(d, edge), std::invalid_argument);
}
