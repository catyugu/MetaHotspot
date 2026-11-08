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
        d.components.emplace_back();
        d.fom(0).mesh.x_vertices = {0, 1, 2};
        d.fom(0).mesh.y_vertices = d.fom(0).mesh.z_vertices = {0, 1};
        MaterialSpec liquid;
        liquid.conductivity_x = liquid.conductivity_y = liquid.conductivity_z = "1";
        liquid.density = liquid.specific_heat = "1";
        liquid.dynamic_viscosity = "T/300";
        d.materials.push_back({"liquid", liquid});
        d.fom(0).volumes = {{0, mhs::core::invalid_id, {{GeometryOperation::Add, {0, 0, 0}, {2, 1, 1}}}}};
        d.fom(0).fluid_boundaries = {
            {{{Axis::X, 0, {0, 1, 0, 1}}}, PressureInlet {1, 280}}, {{{Axis::X, 2, {0, 1, 0, 1}}}, PressureOutlet {0}}};
        return d;
    }
}

TEST(GenericLayout, RuntimeViscosityUsesInitialFieldAndCompiledSnapshotIsReusable)
{
    auto d = channel();
    const auto compiled = mhs::compiler::build_model(d);
    const auto& domain = compiled.fom(0);
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
    std::get<mhs::model::PressureInlet>(d.fom(0).fluid_boundaries.front().condition).temperature.reset();
    const auto compiled = mhs::compiler::build_model(d);
    EXPECT_THROW(mhs::solver::solve(compiled), std::invalid_argument);
    std::get<mhs::model::PressureInlet>(d.fom(0).fluid_boundaries.front().condition).temperature = 280;
    std::get<mhs::model::PressureOutlet>(d.fom(0).fluid_boundaries.back().condition).pressure = 2;
    const auto reversed = mhs::compiler::build_model(d);
    EXPECT_THROW(mhs::solver::solve(reversed), std::invalid_argument);
}

TEST(GenericLayout, NonuniformInitialDensityPreservesFrozenMassBalance)
{
    auto d = channel();
    d.materials[0].value.density = "T/300";
    const auto compiled = mhs::compiler::build_model(d);
    const auto& domain = compiled.fom(0);
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
        d.components.emplace_back();
        d.components.resize(2);
        MaterialSpec solid;
        solid.conductivity_x = solid.conductivity_y = solid.conductivity_z = "1";
        solid.density = solid.specific_heat = "1";
        d.materials.push_back({"solid", solid});
        for (std::size_t side = 0; side < 2; ++side) {
            auto& domain = d.fom(static_cast<DomainId>(side));
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
            d.component(static_cast<DomainId>(side)).ports = {{"join", {Axis::Z, 0.5, {0, 1, 0, 1}}}};
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
    for (std::size_t d = 0; d < compiled.components.size(); ++d) {
        const auto& f = compiled.layout.fields[d];
        for (std::size_t i = 0; i < f.count; ++i) {
            const auto p = compiled.fom(static_cast<mhs::model::DomainId>(d)).discrete.cells[i].center();
            exact[f.dof(i)] = 300 + p[0] + 2 * p[1] + 3 * p[2];
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
    d.component(0).ports.push_back(d.component(0).ports.front());
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
    d.component(0).ports.pop_back();
    d.component(1).ports[0].region.coordinate = 0.6;
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
    d = joined_blocks(4, false);
    d.component(0).thermal_boundaries = {{{d.component(0).ports[0].region}, NeumannBoundary {"0"}}};
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
}

TEST(GenericLayout, NccManufacturedQuadraticConvergesAgainstIndependentSolution)
{
    double previous = 1.0;
    for (const std::size_t n : {4u, 8u, 16u}) {
        auto definition = joined_blocks(n, true);
        definition.sources = {"-6"};
        for (auto& value : definition.components) {
            auto& domain = std::get<mhs::model::FomDomainDefinition>(value.physics);
            domain.default_thermal_boundary = mhs::model::DirichletBoundary {"300+x*x+y*y+z*z"};
            domain.volumes.front().source = 0;
        }
        auto compiled = mhs::compiler::build_model(definition);
        mhs::solver::SolveOptions options;
        options.linear_tolerance = 1e-12;
        options.nonlinear_relative_tolerance = 1e-12;
        const auto corrected = mhs::solver::solve(compiled, {}, {}, options);
        ASSERT_TRUE(corrected.converged);
        compiled.connections[0].nonmatching = false; // explicit zero-corrector control, tests only
        const auto uncorrected = mhs::solver::solve(compiled, {}, {}, options);
        ASSERT_TRUE(uncorrected.converged);
        auto error = [&](const auto& solution) {
            double sum = 0;
            for (std::size_t d = 0; d < compiled.components.size(); ++d) {
                const auto& domain = compiled.fom(static_cast<mhs::model::DomainId>(d));
                for (std::size_t c = 0; c < domain.discrete.cells.size(); ++c) {
                    const auto& cell = domain.discrete.cells[c];
                    const auto x = cell.center();
                    const double exact = 300 + x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                    const double difference = solution.state[compiled.layout.fields[d].dof(c)] - exact;
                    sum += cell.volume() * difference * difference;
                }
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
    for (auto& value : definition.components)
        value.ports[0].region.rectangle = {0, 0.4, 0, 0.7};
    const auto compiled = mhs::compiler::build_model(definition);
    double interface_area = 0;
    for (const auto& patch : compiled.connections[0].patches)
        interface_area += (patch.bounds[1] - patch.bounds[0]) * (patch.bounds[3] - patch.bounds[2]);
    EXPECT_NEAR(interface_area, 0.28, 1e-14);
    for (std::size_t d = 0; d < compiled.components.size(); ++d) {
        double remaining = 0;
        for (const auto& boundary : compiled.fom(static_cast<mhs::model::DomainId>(d)).boundary_faces)
            if (boundary.axis == mhs::model::Axis::Z && boundary.sign == (d == 0 ? 1 : -1))
                remaining += (boundary.bounds[1] - boundary.bounds[0]) * (boundary.bounds[3] - boundary.bounds[2]);
        EXPECT_NEAR(remaining, 0.72, 1e-14);
    }
}

TEST(GenericLayout, StaggeredStokesPressureAndVelocitySatisfyConservativeContinuity)
{
    using namespace mhs::model;
    auto d = channel();
    d.fom(0).flow_mode = FlowMode::Stokes;
    d.fom(0).fluid_boundaries[0].condition = PressureInlet {1.0, 300};
    d.fom(0).fluid_boundaries[1].condition = PressureOutlet {0.0, 300};
    const auto compiled = mhs::compiler::build_model(d);
    ASSERT_EQ(compiled.layout.fields.size(), 3u);
    ASSERT_EQ(compiled.layout.fields[1].quantity, FieldQuantity::Pressure);
    ASSERT_EQ(compiled.layout.fields[2].support(), EntitySupport::Face);
    ASSERT_EQ(compiled.layout.fields[1].count, 2u);
    const auto solution = mhs::solver::solve(compiled);
    ASSERT_TRUE(solution.converged);
    const auto& velocity = compiled.layout.fields[2];
    ASSERT_EQ(velocity.count, 3u);
    for (std::size_t i = 0; i < velocity.count; ++i)
        EXPECT_GT(solution.state[velocity.dof(i)], 0);
    const auto ops = mhs::solver::assemble_thermal(compiled, solution.state, 0.0);
    const auto& pressure = compiled.layout.fields[1];
    for (std::size_t i = 0; i < pressure.count; ++i) {
        const auto dof = pressure.dof(i);
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
    d.fom(0).flow_mode = FlowMode::Stokes;
    d.fom(0).fluid_boundaries[0].condition = PressureInlet {1, 300};
    d.fom(0).fluid_boundaries[1].condition = PressureOutlet {0, 300};
    const auto cold = mhs::compiler::build_model(d);
    const auto first = mhs::solver::solve(cold);
    ASSERT_TRUE(first.converged);
    d.sources = {"5"};
    d.fom(0).volumes[0].source = 0;
    const auto heated = mhs::compiler::build_model(d);
    const auto second = mhs::solver::solve(heated);
    ASSERT_TRUE(second.converged);
    const auto& face = heated.layout.fields[2];
    ASSERT_EQ(face.count, 3u);
    const double cold_flow = first.state[face.dof(1)];
    const double hot_flow = second.state[face.dof(1)];
    EXPECT_GT(second.state[heated.layout.fields[0].dof(0)], first.state[cold.layout.fields[0].dof(0)]);
    EXPECT_LT(hot_flow, cold_flow);
    // Neither nonlinear solve may write its changing material properties
    // or velocity/pressure state into the immutable compiled model.
    const auto repeated = mhs::solver::solve(cold);
    ASSERT_TRUE(repeated.converged);
    EXPECT_NEAR(repeated.state[face.dof(1)], cold_flow, 1e-9);
}

TEST(GenericLayout, TransientStokesKeepsPressureAlgebraicAndConservesMass)
{
    using namespace mhs::model;
    auto definition = channel();
    definition.fom(0).flow_mode = FlowMode::Stokes;
    definition.fom(0).fluid_boundaries[0].condition = PressureInlet {1, 300};
    definition.fom(0).fluid_boundaries[1].condition = PressureOutlet {0, 300};
    auto compiled = mhs::compiler::build_model(definition);
    Study study;
    study.type = StudyType::Transient;
    study.duration = 0.2;
    study.output_interval = 0.1;
    const auto solution = mhs::solver::solve(compiled, study);
    ASSERT_TRUE(solution.converged);
    ASSERT_EQ(solution.snapshot_times.size(), 3u);
    const auto n = compiled.layout.state_count;
    ASSERT_EQ(solution.snapshot_states.size(), 3 * n);
    for (std::size_t step = 0; step < solution.snapshot_times.size(); ++step) {
        const auto time = solution.snapshot_times[step];
        const std::span<const double> state(solution.snapshot_states.data() + step * n, n);
        const auto ops = mhs::solver::assemble_thermal(compiled, state, time);
        const auto& pressure = compiled.layout.fields[1];
        for (std::size_t i = 0; i < pressure.count; ++i) {
            const auto dof = pressure.dof(i);
            EXPECT_DOUBLE_EQ(ops.C.coeff(dof, dof), 0);
            double continuity = -ops.f[dof];
            for (auto entry = ops.K.row_ptr()[dof]; entry < ops.K.row_ptr()[dof + 1]; ++entry)
                continuity += ops.K.values()[entry] * state[ops.K.col_idx()[entry]];
            EXPECT_NEAR(continuity, 0.0, 1e-8);
        }
    }
}

TEST(GenericLayout, StaggeredStokesRectangularPoiseuilleConvergesToAnalyticDuctFlow)
{
    using namespace mhs::model;
    constexpr double pi = 3.14159265358979323846;
    double previous_error = 0;
    for (const unsigned n : {4u, 8u}) {
        auto definition = channel();
        auto& domain = definition.fom(0);
        domain.flow_mode = FlowMode::Stokes;
        domain.mesh.y_vertices.clear();
        domain.mesh.z_vertices.clear();
        for (unsigned i = 0; i <= n; ++i) {
            const double coordinate = static_cast<double>(i) / n;
            domain.mesh.y_vertices.push_back(coordinate);
            domain.mesh.z_vertices.push_back(coordinate);
        }
        definition.materials[0].value.dynamic_viscosity = "1";
        domain.fluid_boundaries[0].condition = PressureInlet {2, 300};
        domain.fluid_boundaries[1].condition = PressureOutlet {0, 300};
        const auto compiled = mhs::compiler::build_model(definition);
        const auto solved = mhs::solver::solve(compiled);
        ASSERT_TRUE(solved.converged);
        const auto& mesh = compiled.fom(0).discrete;
        const auto& velocity = compiled.layout.fields[2];
        double squared_error = 0;
        std::size_t samples = 0;
        for (std::size_t k = 0; k < velocity.count; ++k) {
            const auto& face = mesh.faces[compiled.fom(0).fluid.velocity_faces[k]];
            if (face.exterior() || face.axis != Axis::X)
                continue;
            const double y = (face.bounds[0] + face.bounds[1]) / 2;
            const double z = (face.bounds[2] + face.bounds[3]) / 2;
            // Independent Fourier-series solution for fully developed
            // laminar flow in a unit-square duct, dp/dx = -1, mu = 1.
            double reference = 0;
            for (int harmonic = 1; harmonic <= 41; harmonic += 2) {
                const double kappa = harmonic * pi;
                const double transverse = 1 - std::cosh(kappa * (z - 0.5)) / std::cosh(kappa / 2);
                reference += 4 * std::sin(kappa * y) * transverse / (kappa * kappa * kappa);
            }
            const double difference = solved.state[velocity.dof(k)] - reference;
            squared_error += difference * difference / (n * n);
            ++samples;
        }
        ASSERT_EQ(samples, static_cast<std::size_t>(n * n));
        const double rms_error = std::sqrt(squared_error);
        RecordProperty("duct_error_" + std::to_string(n), rms_error);
        if (previous_error > 0)
            EXPECT_LT(rms_error, previous_error * 0.8);
        previous_error = rms_error;
    }
}
