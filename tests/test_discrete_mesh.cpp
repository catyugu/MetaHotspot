#include "compiler/model_compiler.hpp"
#include "core/vector.hpp"
#include "model/discrete_mesh.hpp"
#include "solver/assembler.hpp"
#include "solver/postprocessor.hpp"
#include "solver/thermal_reconstruction.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

using namespace mhs::core;
using namespace mhs::model;

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
    auto supplied = mhs::compiler::build_model(definition).discrete;
    for (auto& face : supplied.faces)
        face.sign = 0; // import derives orientation from geometry
    definition.mesh = {};
    const auto model = mhs::compiler::build_model(definition, supplied);
    const auto& mesh = model.discrete;
    ASSERT_EQ(mesh.cells.size(), 8u);
    ASSERT_EQ(mesh.faces.size(), 36u);
    ASSERT_EQ(model.cell_material.size(), 8u);
    std::size_t internal = 0;
    for (const auto& face : mesh.faces) {
        EXPECT_GT(face.area, 0);
        if (face.neighbor != mhs::core::invalid_id)
            ++internal;
    }
    EXPECT_EQ(internal, 12u);
    for (std::size_t c = 0; c < mesh.cells.size(); ++c) {
        Vec3 closure {};
        double surface = 0;
        for (const auto id : mesh.incident_faces(static_cast<CellId>(c))) {
            const auto& face = mesh.faces[id];
            const double sign = face.owner == c ? face.sign : -face.sign;
            closure[face.axis] += sign * face.area;
            surface += face.area;
        }
        EXPECT_DOUBLE_EQ(mesh.cells[c].volume(), 0.125);
        EXPECT_DOUBLE_EQ(surface, 1.5);
        for (auto v : closure)
            EXPECT_NEAR(v, 0, 1e-14);
    }
}

TEST(DiscreteMesh, OperatorsAndSamplingRespectArbitraryCellOrdering)
{
    auto definition = cube_definition();
    auto model = mhs::compiler::build_model(definition);
    const mhs::core::Vector<std::size_t> order {7, 5, 3, 1, 6, 4, 2, 0};
    auto mesh = model.discrete;
    mhs::core::Vector<std::size_t> inverse(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        mesh.cells[i] = model.discrete.cells[order[i]];
        inverse[order[i]] = i;
    }
    for (auto& face : mesh.faces) {
        face.owner = inverse[face.owner];
        if (face.neighbor != mhs::core::invalid_id)
            face.neighbor = inverse[face.neighbor];
    }
    auto imported = mhs::compiler::build_model(definition, mesh);
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
    const auto vertices = mhs::solver::interpolate_cell_to_node(model, original, 0);
    const auto nodes = mhs::solver::interpolate_cell_to_node(imported, reordered, 0);
    for (std::size_t v = 0; v < imported.output.points.size(); ++v) {
        auto found = std::find(model.output.points.begin(), model.output.points.end(), imported.output.points[v]);
        ASSERT_NE(found, model.output.points.end());
        EXPECT_NEAR(nodes[v], vertices[found - model.output.points.begin()], 1e-12);
    }
    const Vec3 point {0.2, 0.3, 0.4};
    EXPECT_NEAR(mhs::solver::sample_point(model, original, 0, model.discrete.locate(point), point),
        mhs::solver::sample_point(imported, reordered, 0, imported.discrete.locate(point), point), 1e-12);
}

TEST(DiscreteMesh, TemperatureDependentBoundaryTracesSolveTheSurfaceLaw)
{
    for (const auto& condition : {mhs::model::ThermalBoundary {mhs::model::ConvectionBoundary {"10+0.01*T", "300"}},
             mhs::model::ThermalBoundary {mhs::model::NeumannBoundary {"1+0.001*T"}}}) {
        auto d = cube_definition();
        d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
        d.default_boundary = condition;
        const auto model = mhs::compiler::build_model(d);
        ASSERT_EQ(model.boundary_faces.size(), 6u);
        // The solver's patch lookup binary-searches this order.
        EXPECT_TRUE(std::ranges::is_sorted(model.boundary_faces, {}, [](const auto& patch) { return patch.parent; }));
        const mhs::core::Vector<double> state {350};
        for (const auto& patch : model.boundary_faces) {
            const auto face = model.discrete.faces[patch.parent].project(patch.bounds);
            // Unit conductivity, unit area and half-cell distance give g = 2 W/K;
            // T-dependent laws are evaluated at the surface trace, not at the cell.
            const auto trace
                = mhs::solver::evaluate_boundary(model.boundary_conditions[patch.condition], face, state[0], 1.0, 0);
            const mhs::expression::FieldContext ctx {
                face.center()[0], face.center()[1], face.center()[2], trace.temperature, 0};
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
                mhs::solver::sample_point(model, state, 0, static_cast<CellId>(0), face.center()), trace.temperature);
        }
    }
}

TEST(DiscreteMesh, TemperatureDependentDirichletTemperatureIsRejected)
{
    auto d = cube_definition();
    d.boundaries = {{{{mhs::model::Axis::X, 1, {{0, 1, 0, 1}}}}, mhs::model::DirichletBoundary {"300 + 0.1*T"}}};
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
}

TEST(DiscreteMesh, PartialBoundaryIntegratesActualOverlap)
{
    auto d = cube_definition();
    mhs::model::BoundaryPatch bc;
    bc.condition = mhs::model::NeumannBoundary {"10"};
    bc.regions.push_back({mhs::model::Axis::X, 1.0, {{0.1, 0.2, 0.1, 0.2}}});
    d.boundaries.push_back(bc);
    const auto base = mhs::compiler::build_model(cube_definition()).discrete;
    const auto model = mhs::compiler::build_model(d);
    ASSERT_EQ(model.discrete.faces.size(), base.faces.size());
    EXPECT_EQ(model.discrete.cell_face_offsets, base.cell_face_offsets);
    EXPECT_EQ(model.discrete.cell_face_ids, base.cell_face_ids);
    for (std::size_t i = 0; i < base.faces.size(); ++i) {
        EXPECT_EQ(model.discrete.faces[i].min, base.faces[i].min);
        EXPECT_EQ(model.discrete.faces[i].max, base.faces[i].max);
    }
    const auto ops = mhs::solver::assemble_thermal(model, mhs::core::Vector<double>(8, 300), 0);
    EXPECT_NEAR(std::accumulate(ops.f.begin(), ops.f.end(), 0.0), 0.1, 1e-14);
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
    mhs::model::BoundaryPatch bc;
    bc.condition = mhs::model::DirichletBoundary {"400"};
    bc.regions.push_back({mhs::model::Axis::X, 1, {{0.1, 0.2, 0.1, 0.2}}});
    d.boundaries.push_back(bc);
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

TEST(DiscreteMesh, TypedBoundaryConditionsPreserveExpressionContextAndFlux)
{
    auto d = cube_definition();
    d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
    const std::array<mhs::model::ThermalBoundary, 3> conditions {mhs::model::DirichletBoundary {"280+10*t+y"},
        mhs::model::NeumannBoundary {"x/100+t+y"}, mhs::model::ConvectionBoundary {"x/100+t+y", "280+t+z"}};
    for (size_t kind = 0; kind < conditions.size(); ++kind) {
        SCOPED_TRACE(kind);
        d.boundaries = {{{{mhs::model::Axis::X, 1, {{0, 1, 0, 1}}}}, conditions[kind]}};
        const auto model = mhs::compiler::build_model(d);
        // All other faces share the compiled zero-flux default condition.
        ASSERT_EQ(model.boundary_conditions.size(), 2u);
        ASSERT_EQ(model.boundary_faces.size(), 6u);
        for (const auto& term : model.boundary_faces) {
            const auto face = model.discrete.faces[term.parent].project(term.bounds);
            if (face.axis != 0 || face.sign != 1) {
                EXPECT_TRUE(std::holds_alternative<Neumann>(model.boundary_conditions[term.condition]));
                continue;
            }
            for (double time : {2.0, 3.0}) {
                const double T = 280 + 10 * time;
                const mhs::core::Vector<double> state {T};
                const auto ops = mhs::solver::assemble_thermal(model, state, time);
                // The cell is a unit cube with unit conductivity, so the
                // condensed patch coefficients follow from the trace alone.
                const auto condensed
                    = mhs::solver::evaluate_boundary(model.boundary_conditions[term.condition], face, T, 1.0, time);
                double g, load;
                if (kind == 0) {
                    const auto& bc = std::get<Dirichlet>(model.boundary_conditions[term.condition]);
                    const double prescribed = 280 + 10 * time + 0.5;
                    EXPECT_DOUBLE_EQ(bc.temperature.eval({1, 0.5, 0.5, T, time}), prescribed);
                    EXPECT_NEAR(condensed.temperature, prescribed, 1e-12);
                    g = 2;
                    load = g * prescribed;
                    const auto nodes = mhs::solver::interpolate_cell_to_node(model, state, time);
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
                    const auto& bc = std::get<Neumann>(model.boundary_conditions[term.condition]);
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
                    const auto& bc = std::get<Convection>(model.boundary_conditions[term.condition]);
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

TEST(DiscreteMesh, MissingReferencesRemainInvalid)
{
    DiscreteMesh mesh;
    mesh.cells.push_back({{0, 0, 0}, {1, 1, 1}});
    FacePatch face;
    EXPECT_EQ(face.owner, invalid_id);
    EXPECT_EQ(face.neighbor, invalid_id);
    face.owner = 0;
    mesh.faces.push_back(face);
    mesh.build_incidence();
    EXPECT_EQ(mesh.incident_faces(0).size(), 1u);
    EXPECT_EQ(mesh.locate({2, 2, 2}), invalid_id);
    EXPECT_EQ(mesh.locate({0.5, 0.5, 0.5}), 0u);
}

TEST(DiscreteMesh, ImportedReferencesAreValidatedBeforeUse)
{
    auto definition = cube_definition();
    const auto valid = mhs::compiler::build_model(definition).discrete;
    auto mesh = valid;
    mesh.faces[0].owner = static_cast<CellId>(mesh.cells.size());
    EXPECT_THROW(mhs::compiler::build_model(definition, mesh), std::out_of_range);
    mesh = valid;
    mesh.faces[0].neighbor = static_cast<CellId>(mesh.cells.size());
    EXPECT_THROW(mesh.build_incidence(), std::out_of_range);
    mesh = valid;
    mesh.faces[0].axis = 3;
    EXPECT_THROW(mhs::compiler::build_model(definition, mesh), std::invalid_argument);
}
