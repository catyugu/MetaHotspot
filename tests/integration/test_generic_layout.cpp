#include "compiler/model_compiler.hpp"
#include "solver/assembler.hpp"
#include "solver/frozen_flow.hpp"
#include "solver/solver.hpp"
#include <gtest/gtest.h>
#include <type_traits>

namespace {
    mhs::model::ModelDefinition channel()
    {
        using namespace mhs::model;
        ModelDefinition d;
        d.domain.mesh.x_vertices = {0, 1, 2};
        d.domain.mesh.y_vertices = d.domain.mesh.z_vertices = {0, 1};
        MaterialSpec liquid;
        liquid.conductivity_x = liquid.conductivity_y = liquid.conductivity_z = "1";
        liquid.density = liquid.specific_heat = "1";
        liquid.dynamic_viscosity = "T/300";
        d.materials.push_back({"liquid", liquid});
        d.domain.volumes = {{0, mhs::core::invalid_id, {{GeometryOperation::Add, {0, 0, 0}, {2, 1, 1}}}}};
        d.domain.fluid_boundaries = {
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
    std::get<mhs::model::PressureInlet>(d.domain.fluid_boundaries.front().condition).temperature.reset();
    const auto compiled = mhs::compiler::build_model(d);
    EXPECT_THROW(mhs::solver::solve(compiled), std::invalid_argument);
    std::get<mhs::model::PressureInlet>(d.domain.fluid_boundaries.front().condition).temperature = 280;
    std::get<mhs::model::PressureOutlet>(d.domain.fluid_boundaries.back().condition).pressure = 2;
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
