#include "compiler/geometry_compiler.hpp"
#include "pta_test_utils.hpp"
#include "solver/assembler.hpp"
#include "solver/thermal_physics.hpp"
#include "solver/thermal_reconstruction.hpp"
#include <algorithm>
#include <gtest/gtest.h>
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

TEST(Mesh, OperatorsAndSamplingRespectArbitraryCellOrdering)
{
    auto definition = cube_definition();
    auto model = test_pta::build_model(definition, definition.mesh);
    const mhs::core::Vector<std::size_t> order {7, 5, 3, 1, 6, 4, 2, 0};
    NonConformalMesh mesh;
    mhs::core::Vector<CellId> inverse(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        mesh.cells.push_back(cell_geometry(model.mesh, static_cast<CellId>(order[i])));
        inverse[order[i]] = static_cast<CellId>(i);
    }
    for_each_connection(model.mesh, [&](Connection c) { mesh.connections.push_back({inverse[c.a], inverse[c.b]}); });
    auto imported = test_pta::build_model(definition, definition.mesh);
    static_cast<mhs::model::Model&>(imported) = mhs::compiler::build_model(definition, mesh);
    for (auto* m : {&model, &imported}) {
        m->material_table[0].kx = mhs::expression::parse("1+T/1000", {});
        m->heat_source_table[0] = mhs::expression::parse("x+2*y+3*z", {});
    }
    mhs::core::Vector<double> original {300, 310, 320, 330, 340, 350, 360, 370};
    mhs::core::Vector<double> reordered(8);
    for (std::size_t i = 0; i < order.size(); ++i)
        reordered[i] = original[order[i]];
    const auto baseline = mhs::solver::assemble_thermal(model, original, 0);
    const auto result = mhs::solver::assemble_thermal(imported, reordered, 0);
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(result.f[i], baseline.f[order[i]], 1e-14);
        for (std::size_t j = 0; j < 8; ++j) {
            EXPECT_NEAR(result.K.coeff(i, j), baseline.K.coeff(order[i], order[j]), 1e-14);
            EXPECT_NEAR(result.C.coeff(i, j), baseline.C.coeff(order[i], order[j]), 1e-14);
        }
    }
    const auto vertices = test_pta::interpolate_cell_to_node(model, original, 0);
    const auto nodes = test_pta::interpolate_cell_to_node(imported, reordered, 0);
    for (std::size_t v = 0; v < imported.output.points.size(); ++v) {
        auto found = std::find(model.output.points.begin(), model.output.points.end(), imported.output.points[v]);
        ASSERT_NE(found, model.output.points.end());
        EXPECT_NEAR(nodes[v], vertices[found - model.output.points.begin()], 1e-12);
    }
    const Vec3 point {0.2, 0.3, 0.4};
    EXPECT_NEAR(mhs::solver::sample_point(model, original, 0, static_cast<CellId>(0), point),
        mhs::solver::sample_point(imported, reordered, 0, static_cast<CellId>(inverse[0]), point), 1e-12);
}

TEST(Mesh, TypedBoundaryConditionsPreserveExpressionContextAndFlux)
{
    auto d = cube_definition();
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
    const std::array<mhs::model::ThermalBoundary, 3> conditions {mhs::model::DirichletBoundary {"280+10*t+y"},
        mhs::model::NeumannBoundary {"T/100+t+y"}, mhs::model::ConvectionBoundary {"T/100+t+y", "280+t+z"}};
    for (size_t kind = 0; kind < conditions.size(); ++kind) {
        SCOPED_TRACE(kind);
        d.thermal_boundaries = {{{{mhs::model::Axis::X, 1, {0, 1, 0, 1}}}, conditions[kind]}};
        const auto model = test_pta::build_model(d, d.mesh);
        // All other faces share the compiled zero-flux default condition.
        ASSERT_EQ(model.boundary_conditions.size(), 2u);
        ASSERT_EQ(model.boundary_facets.size(), 6u);
        for (std::size_t id = 0; id < model.boundary_facets.size(); ++id) {
            const auto& facet = model.boundary_facets[id];
            const auto& condition = model.boundary_conditions[model.boundary_thermal_condition[id]];
            if (facet.axis != Axis::X || facet.side != 1) {
                EXPECT_TRUE(std::holds_alternative<Neumann>(condition));
                continue;
            }
            for (double time : {2.0, 3.0}) {
                const double T = 280 + 10 * time;
                const mhs::core::Vector<double> state {T};
                const auto ops = mhs::solver::assemble_thermal(model, state, time);
                const auto center = facet.center();
                const mhs::expression::FieldContext context {center[0], center[1], center[2], T, time};
                const double d = cell_geometry(model.mesh, facet.cell).size()[0] / 2;
                const double g_cell = facet.area() / d;
                const auto condensed = mhs::solver::evaluate_boundary(condition, context, facet.area(), g_cell);
                double g, load;
                if (kind == 0) {
                    const auto& bc = std::get<Dirichlet>(condition);
                    const double prescribed = 280 + 10 * time + 0.5;
                    EXPECT_DOUBLE_EQ(bc.temperature.eval({1, 0.5, 0.5, T, time}), prescribed);
                    EXPECT_NEAR(condensed.temperature, prescribed, 1e-12);
                    g = 2;
                    load = g * prescribed;
                    const auto nodes = test_pta::interpolate_cell_to_node(model, state, time);
                    for (std::size_t v = 0; v < nodes.size(); ++v) {
                        const auto& p = model.output.points[v];
                        if (p[0] == 1)
                            EXPECT_NEAR(nodes[v], prescribed, 1e-10);
                    }
                    EXPECT_DOUBLE_EQ(
                        mhs::solver::sample_point(model, state, time, static_cast<CellId>(0), {1, 0.5, 0.5}),
                        prescribed);
                }
                else if (kind == 1) {
                    const auto& bc = std::get<Neumann>(condition);
                    const double q = (T / 100 + time + 0.5) / (1 - 0.5 / 100);
                    const double trace = T + q * 0.5;
                    EXPECT_NEAR(bc.heat_flux.eval({1, 0.5, 0.5, trace, time}), q, 1e-12);
                    // The local trace solve stops at 1e-9 relative trace width,
                    // far inside the global nonlinear tolerance but not exact.
                    EXPECT_NEAR(condensed.temperature, trace, 1e-6);
                    g = 0;
                    load = q;
                }
                else {
                    const auto& bc = std::get<Convection>(condition);
                    const double trace = condensed.temperature;
                    const double h = trace / 100 + time + 0.5, ambient = 280 + time + 0.5;
                    EXPECT_DOUBLE_EQ(bc.coefficient.eval({1, 0.5, 0.5, trace, time}), h);
                    EXPECT_DOUBLE_EQ(bc.ambient_temperature.eval({1, 0.5, 0.5, trace, time}), ambient);
                    EXPECT_NEAR(2 * (T - trace), h * (trace - ambient), 1e-5);
                    g = h / (1 + h * 0.5);
                    load = g * ambient;
                }
                EXPECT_NEAR(ops.K.coeff(0, 0), g, 1e-9);
                EXPECT_NEAR(ops.f[0], load, 1e-6);
                // Condensed coefficients and the assembled operator must agree.
                EXPECT_NEAR(condensed.conductance, g, 1e-9);
                EXPECT_NEAR(condensed.load, load, 1e-6);
            }
        }
    }
}
