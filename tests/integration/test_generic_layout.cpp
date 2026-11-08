#include "compiler/model_compiler.hpp"
#include "solver/assembler.hpp"
#include "solver/connection_flux.hpp"
#include "solver/frozen_flow.hpp"
#include "solver/solver.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

namespace {
    mhs::model::ModelDefinition channel()
    {
        using namespace mhs::model;
        ModelDefinition d;
        d.domains[0].mesh.x_vertices = {0, 1, 2};
        d.domains[0].mesh.y_vertices = d.domains[0].mesh.z_vertices = {0, 1};
        MaterialSpec liquid;
        liquid.conductivity_x = liquid.conductivity_y = liquid.conductivity_z = "1";
        liquid.density = liquid.specific_heat = "1";
        liquid.dynamic_viscosity = "T/300";
        d.materials.push_back({"liquid", liquid});
        d.domains[0].volumes = {{0, mhs::core::invalid_id, {{GeometryOperation::Add, {0, 0, 0}, {2, 1, 1}}}}};
        d.domains[0].fluid_boundaries = {
            {{{Axis::X, 0, {0, 1, 0, 1}}}, PressureInlet {1, 280}}, {{{Axis::X, 2, {0, 1, 0, 1}}}, PressureOutlet {0}}};
        return d;
    }
}

TEST(GenericLayout, RuntimeViscosityUsesInitialFieldAndCompiledSnapshotIsReusable)
{
    auto d = channel();
    const auto compiled = mhs::compiler::build_model(d);
    const auto& domain = compiled.domains.front();
    ASSERT_EQ(compiled.layout.state_count, 2u);
    ASSERT_EQ(compiled.layout.fields.size(), 1u);
    EXPECT_EQ(domain.fluid.fluid_to_global.size(), 2u);
    // Source definition mutation cannot affect compiled viscosity expressions.
    d.materials[0].value.dynamic_viscosity = "0";
    const auto cold = mhs::solver::initialize_frozen_flow(domain, mhs::core::Vector<double> {300, 300});
    const auto hot = mhs::solver::initialize_frozen_flow(domain, mhs::core::Vector<double> {600, 600});
    const auto again = mhs::solver::initialize_frozen_flow(domain, mhs::core::Vector<double> {300, 300});
    for (std::size_t i = 0; i < cold.face_mass_flow.size(); ++i) {
        EXPECT_NEAR(hot.face_mass_flow[i] * 2, cold.face_mass_flow[i], 1e-12);
        EXPECT_DOUBLE_EQ(again.face_mass_flow[i], cold.face_mass_flow[i]);
    }
    const auto a = mhs::solver::solve(compiled);
    const auto b = mhs::solver::solve(compiled, {}, mhs::core::Vector<double> {600, 600});
    EXPECT_TRUE(a.converged);
    EXPECT_TRUE(b.converged);
}

TEST(GenericLayout, MissingActualInflowTemperatureFailsAtRuntime)
{
    auto d = channel();
    std::get<mhs::model::PressureInlet>(d.domains[0].fluid_boundaries.front().condition).temperature.reset();
    const auto compiled = mhs::compiler::build_model(d);
    EXPECT_THROW(mhs::solver::solve(compiled), std::invalid_argument);
    std::get<mhs::model::PressureInlet>(d.domains[0].fluid_boundaries.front().condition).temperature = 280;
    std::get<mhs::model::PressureOutlet>(d.domains[0].fluid_boundaries.back().condition).pressure = 2;
    const auto reversed = mhs::compiler::build_model(d);
    EXPECT_THROW(mhs::solver::solve(reversed), std::invalid_argument);
}

TEST(GenericLayout, PermutedTemperatureDofsPreserveOperatorsAndPhysicalSolution)
{
    auto a = mhs::compiler::build_model(channel());
    auto b = a;
    b.layout.fields[0].dofs = {1, 0};
    const auto original = mhs::solver::assemble_thermal(a, mhs::core::Vector<double> {300, 400}, 0);
    const auto permuted = mhs::solver::assemble_thermal(b, mhs::core::Vector<double> {400, 300}, 0);
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(original.f[i], permuted.f[1 - i]);
        for (std::size_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(original.K.coeff(i, j), permuted.K.coeff(1 - i, 1 - j));
            EXPECT_DOUBLE_EQ(original.C.coeff(i, j), permuted.C.coeff(1 - i, 1 - j));
        }
    }
    const auto sa = mhs::solver::solve(a);
    const auto sb = mhs::solver::solve(b);
    EXPECT_NEAR(sa.state[0], sb.state[1], 1e-10);
    EXPECT_NEAR(sa.state[1], sb.state[0], 1e-10);
}

TEST(GenericLayout, NonuniformInitialDensityPreservesFrozenMassBalance)
{
    auto d = channel();
    d.materials[0].value.density = "T/300";
    const auto compiled = mhs::compiler::build_model(d);
    const auto& domain = compiled.domains.front();
    const auto flow = mhs::solver::initialize_frozen_flow(domain, mhs::core::Vector<double> {300, 600});
    mhs::core::Vector<double> balance(2, 0);
    for (std::size_t i = 0; i < domain.discrete.faces.size(); ++i) {
        const auto& face = domain.discrete.faces[i];
        if (!face.exterior()) {
            balance[face.minus] += flow.face_mass_flow[i];
            balance[face.plus] -= flow.face_mass_flow[i];
        }
    }
    for (const auto& term : flow.boundary_terms)
        std::visit(
            [&](const auto& value) {
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, mhs::solver::FluidInflow>)
                    balance[term.cell] -= value.mass_flow_rate;
                else
                    balance[term.cell] += value.mass_flow_rate;
            },
            term.condition);
    for (const auto residual : balance)
        EXPECT_NEAR(residual, 0, 1e-12);
}

namespace {
    mhs::model::ModelDefinition joined_blocks(std::size_t n, bool nonmatching)
    {
        using namespace mhs::model;
        ModelDefinition d;
        d.domains.resize(2);
        MaterialSpec solid;
        solid.conductivity_x = solid.conductivity_y = solid.conductivity_z = "1";
        solid.density = solid.specific_heat = "1";
        d.materials.push_back({"solid", solid});
        for (std::size_t side = 0; side < 2; ++side) {
            auto& domain = d.domains[side];
            const auto nx = side == 1 && nonmatching ? n / 2 : n;
            for (std::size_t i = 0; i <= nx; ++i) {
                domain.mesh.x_vertices.push_back(static_cast<double>(i) / static_cast<double>(nx));
                domain.mesh.y_vertices.push_back(static_cast<double>(i) / static_cast<double>(nx));
            }
            for (std::size_t i = 0; i <= n / 2; ++i)
                domain.mesh.z_vertices.push_back(
                    static_cast<double>(side) / 2 + static_cast<double>(i) / static_cast<double>(n));
            domain.volumes = {{0, mhs::core::invalid_id, {{GeometryOperation::Add, {0, 0, 0}, {1, 1, 1}}}}};
            domain.default_thermal_boundary = DirichletBoundary {"300+x+2*y+3*z"};
            domain.ports = {{"join", {Axis::Z, 0.5, {0, 1, 0, 1}}}};
        }
        d.connections = {{{0, 0}, {1, 0}}};
        return d;
    }
}

TEST(GenericLayout, NonmatchingConnectionCorrectsLinearTangentialOffsetAndConservesFlux)
{
    const auto compiled = mhs::compiler::build_model(joined_blocks(4, true));
    ASSERT_EQ(compiled.connections.size(), 1u);
    const auto& connection = compiled.connections[0];
    EXPECT_TRUE(connection.nonmatching);
    EXPECT_EQ(connection.patches.size(), 16u);
    mhs::core::Vector<double> exact(compiled.layout.state_count);
    for (std::size_t d = 0; d < compiled.domains.size(); ++d) {
        const auto& f = compiled.layout.fields[d];
        for (std::size_t i = 0; i < f.dofs.size(); ++i) {
            const auto p = compiled.domains[d].discrete.cells[f.entities[i]].center();
            exact[f.dofs[i]] = 300 + p[0] + 2 * p[1] + 3 * p[2];
        }
    }
    const auto corrected = mhs::solver::assemble_thermal(compiled, exact, 0);
    auto baseline = compiled;
    baseline.connections[0].nonmatching = false;
    const auto tpfa = mhs::solver::assemble_thermal(baseline, exact, 0);
    double max_corrected = 0, max_tpfa = 0, total_correction = 0;
    for (std::size_t row = 0; row < exact.size(); ++row) {
        double rc = -corrected.f[row], rb = -tpfa.f[row];
        for (std::size_t col = 0; col < exact.size(); ++col) {
            rc += corrected.K.coeff(row, col) * exact[col];
            rb += tpfa.K.coeff(row, col) * exact[col];
        }
        max_corrected = std::max(max_corrected, std::abs(rc));
        max_tpfa = std::max(max_tpfa, std::abs(rb));
        total_correction += corrected.f[row] - tpfa.f[row];
    }
    EXPECT_LT(max_corrected, 1e-11);
    EXPECT_GT(max_tpfa, 0.01);
    EXPECT_NEAR(total_correction, 0, 1e-12);
    // Reversing the declared orientation reverses each physical heat rate.
    const auto original_flux = mhs::solver::evaluate_connection_flux(compiled, connection, exact, 0);
    auto reversed = connection;
    std::swap(reversed.a, reversed.b);
    for (auto& patch : reversed.patches)
        std::swap(patch.a, patch.b);
    const auto reversed_flux = mhs::solver::evaluate_connection_flux(compiled, reversed, exact, 0);
    for (std::size_t i = 0; i < original_flux.size(); ++i) {
        EXPECT_NEAR(original_flux[i].outward_a, -reversed_flux[i].outward_a, 1e-12);
        EXPECT_NEAR(original_flux[i].interface_temperature, reversed_flux[i].interface_temperature, 1e-12);
    }
    // Permute DOFs across both domain blocks, independently of local CellIds.
    auto permuted = compiled;
    for (auto& field : permuted.layout.fields)
        for (auto& dof : field.dofs)
            dof = static_cast<mhs::model::DofId>(exact.size() - 1 - dof);
    auto reverse_state = exact;
    std::reverse(reverse_state.begin(), reverse_state.end());
    const auto mapped = mhs::solver::assemble_thermal(permuted, reverse_state, 0);
    for (std::size_t i = 0; i < exact.size(); ++i) {
        EXPECT_NEAR(corrected.f[i], mapped.f[exact.size() - 1 - i], 1e-12);
        for (std::size_t j = 0; j < exact.size(); ++j) {
            EXPECT_DOUBLE_EQ(corrected.K.coeff(i, j), mapped.K.coeff(exact.size() - 1 - i, exact.size() - 1 - j));
            EXPECT_DOUBLE_EQ(corrected.C.coeff(i, j), mapped.C.coeff(exact.size() - 1 - i, exact.size() - 1 - j));
        }
    }
    const auto solution = mhs::solver::solve(compiled);
    EXPECT_TRUE(solution.converged);
    for (std::size_t i = 0; i < exact.size(); ++i)
        EXPECT_NEAR(solution.state[i], exact[i], 2e-4);
}

TEST(GenericLayout, ConnectionRejectsMissingDuplicateOverlappingAndInvalidPorts)
{
    using namespace mhs::model;
    auto d = joined_blocks(4, false);
    d.connections.clear();
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
    d.connections = {{{0, 0}, {1, 0}}, {{0, 0}, {1, 0}}};
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
    d.connections.resize(1);
    d.domains[0].ports.push_back(d.domains[0].ports.front());
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
    d.domains[0].ports.pop_back();
    d.domains[1].ports[0].region.coordinate = 0.6;
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
    d = joined_blocks(4, false);
    d.domains[0].thermal_boundaries = {{{d.domains[0].ports[0].region}, NeumannBoundary {"0"}}};
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
}

TEST(GenericLayout, NccManufacturedQuadraticConvergesAgainstIndependentSolution)
{
    double previous = 1.0;
    for (const std::size_t n : {4u, 8u, 16u}) {
        auto definition = joined_blocks(n, true);
        definition.sources = {"-6"};
        for (auto& d : definition.domains) {
            d.default_thermal_boundary = mhs::model::DirichletBoundary {"300+x*x+y*y+z*z"};
            d.volumes.front().source = 0;
        }
        auto compiled = mhs::compiler::build_model(definition);
        mhs::solver::SolveOptions options;
        options.linear_tolerance = 1e-12;
        options.nonlinear_relative_tolerance = 1e-12;
        const auto corrected = mhs::solver::solve(compiled, compiled.study, {}, options);
        ASSERT_TRUE(corrected.converged);
        compiled.connections[0].nonmatching = false; // explicit zero-corrector control, tests only
        const auto uncorrected = mhs::solver::solve(compiled, compiled.study, {}, options);
        ASSERT_TRUE(uncorrected.converged);
        auto error = [&](const auto& solution) {
            double sum = 0;
            for (std::size_t d = 0; d < compiled.domains.size(); ++d)
                for (std::size_t c = 0; c < compiled.domains[d].discrete.cells.size(); ++c) {
                    const auto& cell = compiled.domains[d].discrete.cells[c];
                    const auto x = cell.center();
                    const double exact = 300 + x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                    const double difference = solution.state[compiled.layout.fields[d].dofs[c]] - exact;
                    sum += cell.volume() * difference * difference;
                }
            return std::sqrt(sum);
        };
        const double ncc = error(corrected), tpfa = error(uncorrected);
        RecordProperty("ncc_error_" + std::to_string(n), ncc);
        RecordProperty("tpfa_error_" + std::to_string(n), tpfa);
        EXPECT_LT(ncc, previous * 0.4); // at least 1.32 order, independent of a fitted reference
        EXPECT_LT(ncc, tpfa);
        previous = ncc;
    }
}

TEST(GenericLayout, PartialPortsPreserveExternalBoundarySubdivisions)
{
    auto definition = joined_blocks(4, true);
    for (auto& domain : definition.domains)
        domain.ports[0].region.rectangle = {0, 0.4, 0, 0.7};
    const auto compiled = mhs::compiler::build_model(definition);
    double interface_area = 0;
    for (const auto& patch : compiled.connections[0].patches)
        interface_area += (patch.bounds[1] - patch.bounds[0]) * (patch.bounds[3] - patch.bounds[2]);
    EXPECT_NEAR(interface_area, 0.28, 1e-14);
    for (std::size_t d = 0; d < compiled.domains.size(); ++d) {
        double remaining = 0;
        for (const auto& boundary : compiled.domains[d].boundary_faces)
            if (boundary.axis == mhs::model::Axis::Z && boundary.sign == (d == 0 ? 1 : -1))
                remaining += (boundary.bounds[1] - boundary.bounds[0]) * (boundary.bounds[3] - boundary.bounds[2]);
        EXPECT_NEAR(remaining, 0.72, 1e-14);
    }
}


TEST(GenericLayout, StaggeredStokesPressureAndVelocitySatisfyConservativeContinuity)
{
    using namespace mhs::model;
    auto d = channel();
    d.domains[0].flow_mode = FlowMode::Stokes;
    d.domains[0].fluid_boundaries[0].condition = PressureInlet {1.0, 300};
    d.domains[0].fluid_boundaries[1].condition = PressureOutlet {0.0, 300};
    const auto compiled = mhs::compiler::build_model(d);
    ASSERT_EQ(compiled.layout.fields.size(), 5u);
    ASSERT_EQ(compiled.layout.fields[1].quantity, FieldQuantity::Pressure);
    ASSERT_EQ(compiled.layout.fields[2].support, EntitySupport::Face);
    ASSERT_EQ(compiled.layout.fields[1].dofs.size(), 2u);
    const auto solution = mhs::solver::solve(compiled);
    ASSERT_TRUE(solution.converged);
    const auto& velocity = compiled.layout.fields[2];
    ASSERT_EQ(velocity.dofs.size(), 3u);
    for (const auto dof : velocity.dofs)
        EXPECT_GT(solution.state[dof], 0);
    const auto ops = mhs::solver::assemble_thermal(compiled, solution.state, 0.0);
    for (const auto dof : compiled.layout.fields[1].dofs) {
        double residual = -ops.f[dof];
        for (auto p = ops.K.row_ptr()[dof]; p < ops.K.row_ptr()[dof + 1]; ++p)
            residual += ops.K.values()[p] * solution.state[ops.K.col_idx()[p]];
        EXPECT_NEAR(residual, 0.0, 1e-9);
        EXPECT_DOUBLE_EQ(ops.C.coeff(dof, dof), 0);
    }
}

TEST(GenericLayout, StaggeredStokesThermalFeedbackUpdatesViscosityAndMassFlux)
{
    using namespace mhs::model;
    auto d = channel();
    d.domains[0].flow_mode = FlowMode::Stokes;
    d.domains[0].fluid_boundaries[0].condition = PressureInlet {1, 300};
    d.domains[0].fluid_boundaries[1].condition = PressureOutlet {0, 300};
    const auto cold = mhs::compiler::build_model(d);
    const auto first = mhs::solver::solve(cold);
    ASSERT_TRUE(first.converged);
    d.sources = {"5"};
    d.domains[0].volumes[0].source = 0;
    const auto heated = mhs::compiler::build_model(d);
    const auto second = mhs::solver::solve(heated);
    ASSERT_TRUE(second.converged);
    const auto& face = heated.layout.fields[2];
    ASSERT_EQ(face.dofs.size(), 3u);
    const double cold_flow = first.state[face.dofs[1]];
    const double hot_flow = second.state[face.dofs[1]];
    EXPECT_GT(second.state[heated.layout.fields[0].dofs[0]], first.state[cold.layout.fields[0].dofs[0]]);
    EXPECT_LT(hot_flow, cold_flow);
    // Neither nonlinear solve may write its changing material properties
    // or velocity/pressure state into the immutable compiled model.
    const auto repeated = mhs::solver::solve(cold);
    ASSERT_TRUE(repeated.converged);
    EXPECT_NEAR(repeated.state[face.dofs[1]], cold_flow, 1e-9);
}
