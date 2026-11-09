#include "compiler/model_compiler.hpp"
#include "library_test_utils.hpp"
#include "solver/assembler.hpp"
#include "solver/connection_flux.hpp"
#include "solver/solver.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
    using namespace mhs::model;
    constexpr double q = 0.7071067811865475244;

    ComponentDefinition block(
        std::initializer_list<double> x, bool first, const std::shared_ptr<LibraryDefinition>& library)
    {
        ComponentDefinition component;
        auto& d = std::get<FomDomainDefinition>(component.physics);
        d.mesh.x_vertices = x;
        d.mesh.y_vertices = d.mesh.z_vertices = {0.0, 1.0};
        d.volumes = {
            {library->material_id("solid"), mhs::core::invalid_id, {{GeometryOperation::Add, {0, 0, 0}, {3, 1, 1}}}}};
        d.default_thermal_boundary = library->add_thermal_boundary(NeumannBoundary {});
        component.thermal_boundaries.push_back({{{Axis::X, first ? 0.0 : 3.0, {0, 1, 0, 1}}},
            library->add_thermal_boundary(DirichletBoundary {first ? "300" : "280"})});
        component.ports.push_back({"interface", {Axis::X, 1.0, {0, 1, 0, 1}}});
        return component;
    }

    // Tiny independent dense solver for a single implicit-Euler update;
    // it avoids adaptive-step controller differences between FOM and ROM
    // coordinates when checking strict Galerkin dynamical equivalence.
    std::array<double, 3> implicit_euler_step(
        const mhs::solver::Operators& op, std::span<const double> initial, double dt)
    {
        std::array<std::array<double, 4>, 3> a {};
        for (std::size_t i = 0; i < 3; ++i) {
            a[i][3] = op.f[i];
            for (std::size_t j = 0; j < 3; ++j) {
                a[i][j] = op.C.coeff(i, j) / dt + op.K.coeff(i, j);
                a[i][3] += op.C.coeff(i, j) * initial[j] / dt;
            }
        }
        for (std::size_t k = 0; k < 3; ++k) {
            std::size_t pivot = k;
            for (std::size_t row = k + 1; row < 3; ++row)
                if (std::abs(a[row][k]) > std::abs(a[pivot][k]))
                    pivot = row;
            if (std::abs(a[pivot][k]) < 1e-14)
                throw std::runtime_error("singular implicit-Euler reference system");
            std::swap(a[k], a[pivot]);
            for (std::size_t row = k + 1; row < 3; ++row) {
                const double factor = a[row][k] / a[k][k];
                for (std::size_t col = k; col < 4; ++col)
                    a[row][col] -= factor * a[k][col];
            }
        }
        std::array<double, 3> result {};
        for (int i = 2; i >= 0; --i) {
            double rhs = a[i][3];
            for (std::size_t j = i + 1; j < 3; ++j)
                rhs -= a[i][j] * result[j];
            result[i] = rhs / a[i][i];
        }
        return result;
    }

    ModelDefinition blocks(bool reduced)
    {
        ModelDefinition model;
        auto library = model.library_handle;
        MaterialSpec mat;
        mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "1";
        mat.density = mat.specific_heat = "1";
        library->add_material({"solid", mat});
        model.components.push_back(block({0.0, 1.0}, true, library));
        if (!reduced) {
            model.components.push_back(block({1.0, 2.0, 3.0}, false, library));
        }
        else {
            RomDomainDefinition rom;
            rom.order = 2;
            // Galerkin projection of K_B = [[1,-1],[-1,3]],
            // C_B = I and f_B = [0,560] by V = [[q,-q],[q,q]].
            rom.K = {1, 1, 1, 3};
            rom.C = {1, 0, 0, 1};
            rom.f = {560 * q, 560 * q};
            rom.initial_state = {300 / q, 0};
            PortDefinition port;
            port.name = "interface";
            port.region = {Axis::X, 1.0, {0, 1, 0, 1}};
            port.outward_sign = -1;
            RomPortPatch facet;
            facet.rectangle = {0, 1, 0, 1};
            facet.half_resistance = 0.5;
            facet.center = {0.5, 0.5};
            facet.trace = {q, -q};
            port.patches.push_back(facet);
            model.components.push_back({rom, {port}, {}, {}});
        }
        model.component(1).observations.push_back(
            {{"top", 2.5, 0.5, 0.5}, reduced ? mhs::core::Vector<double> {q, q} : mhs::core::Vector<double> {}, 0});
        model.connections.push_back({{0, 0}, {1, 0}});
        return model;
    }

}

TEST(RomDomain, FullBasisGalerkinsOperatorsAndTransientMatchFom)
{
    const auto fom = mhs::compiler::build_model(blocks(false));
    const auto rom = mhs::compiler::build_model(blocks(true));
    EXPECT_EQ(fom.layout.state_count, 3u);
    EXPECT_EQ(rom.layout.state_count, 3u);
    ASSERT_EQ(rom.components.size(), 2u);
    EXPECT_EQ(rom.rom(1).f.size(), 2u);
    const auto init_fom = mhs::core::Vector<double> {300, 300, 300};
    const auto init_rom = mhs::core::Vector<double> {300, 300 / q, 0};
    const auto a = mhs::solver::assemble_thermal(fom, init_fom, 0);
    const auto b = mhs::solver::assemble_thermal(rom, init_rom, 0);
    const std::array<std::array<double, 3>, 3> basis {{{1, 0, 0}, {0, q, -q}, {0, q, q}}};
    for (std::size_t i = 0; i < 3; ++i) {
        double load = 0;
        for (std::size_t m = 0; m < 3; ++m)
            load += basis[m][i] * a.f[m];
        EXPECT_NEAR(b.f[i], load, 1e-10);
        for (std::size_t j = 0; j < 3; ++j) {
            double k = 0, c = 0;
            for (std::size_t m = 0; m < 3; ++m)
                for (std::size_t n = 0; n < 3; ++n) {
                    k += basis[m][i] * a.K.coeff(m, n) * basis[n][j];
                    c += basis[m][i] * a.C.coeff(m, n) * basis[n][j];
                }
            EXPECT_NEAR(b.K.coeff(i, j), k, 1e-12);
            EXPECT_NEAR(b.C.coeff(i, j), c, 1e-12);
        }
    }

    const mhs::model::Study steady {};
    const auto f = mhs::solver::solve(fom, steady);
    const auto z = mhs::solver::solve(rom, steady);
    ASSERT_TRUE(f.converged);
    ASSERT_TRUE(z.converged);
    EXPECT_NEAR(z.state[0], f.state[0], 1e-8);
    EXPECT_NEAR(q * (z.state[1] - z.state[2]), f.state[1], 1e-8);
    EXPECT_NEAR(q * (z.state[1] + z.state[2]), f.state[2], 1e-8);
    ASSERT_EQ(f.probe_traces.size(), 1u);
    ASSERT_EQ(z.probe_traces.size(), 1u);
    EXPECT_NEAR(f.probe_traces[0].values.back(), z.probe_traces[0].values.back(), 1e-8);

    const auto first_fom = implicit_euler_step(a, init_fom, 0.05);
    const auto first_rom = implicit_euler_step(b, init_rom, 0.05);
    EXPECT_NEAR(first_fom[0], first_rom[0], 1e-10);
    EXPECT_NEAR(first_fom[1], q * (first_rom[1] - first_rom[2]), 1e-10);
    EXPECT_NEAR(first_fom[2], q * (first_rom[1] + first_rom[2]), 1e-10);

    const mhs::model::Study transient {mhs::model::StudyType::Transient, 300, 0.5, 0.25};
    // The production adaptive controller uses a coordinate-dependent error
    // norm. Both representations must approximate this independent analytic
    // response within the solver's default integration tolerance, but they
    // need not select identical time steps.
    const auto ft = mhs::solver::solve(fom, transient);
    const auto zt = mhs::solver::solve(rom, transient);
    ASSERT_TRUE(ft.converged);
    ASSERT_TRUE(zt.converged);
    ASSERT_EQ(ft.snapshot_times.size(), zt.snapshot_times.size());
    for (std::size_t t = 0; t < ft.snapshot_times.size(); ++t) {
        EXPECT_NEAR(ft.snapshot_times[t], zt.snapshot_times[t], 1e-12);
        const auto offset = t * 3;
        const double time = ft.snapshot_times[t];
        const double slow = std::exp(-time), fast = std::exp(-4 * time);
        const double middle = std::exp(-3 * time);
        const std::array<double, 3> exact {
            (890 + 20 * slow + 10 * fast - 20 * middle) / 3.0,
            290 + (40 * slow - 10 * fast) / 3.0,
            (850 + 20 * slow + 10 * fast + 20 * middle) / 3.0,
        };
        for (std::size_t i = 0; i < 3; ++i)
            EXPECT_NEAR(ft.snapshot_states[offset + i], exact[i], 5e-2);
        EXPECT_NEAR(ft.snapshot_states[offset], zt.snapshot_states[offset], 5e-2);
        EXPECT_NEAR(ft.snapshot_states[offset + 1],
            q * (zt.snapshot_states[offset + 1] - zt.snapshot_states[offset + 2]), 5e-2);
        EXPECT_NEAR(ft.snapshot_states[offset + 2],
            q * (zt.snapshot_states[offset + 1] + zt.snapshot_states[offset + 2]), 5e-2);
        EXPECT_NEAR(ft.probe_traces[0].values[t], zt.probe_traces[0].values[t], 5e-2);
    }
}

TEST(RomDomain, RomToRomConnectionNeedsNoFakeFomMesh)
{
    mhs::model::ModelDefinition def;
    auto library = def.library_handle;
    for (std::size_t i = 0; i < 2; ++i) {
        mhs::model::RomDomainDefinition rom;
        rom.order = 1;
        rom.K = {2};
        rom.C = {1};
        rom.f = {i == 0 ? 600.0 : 560.0};
        rom.initial_state = {300};
        mhs::model::PortDefinition port;
        port.name = "join";
        port.region = {mhs::model::Axis::X, 1, {0, 1, 0, 1}};
        port.outward_sign = i == 0 ? 1 : -1;
        mhs::model::RomPortPatch patch;
        patch.rectangle = {0, 1, 0, 1};
        patch.half_resistance = 0.5;
        patch.center = {0.5, 0.5};
        patch.trace = {1};
        port.patches.push_back(patch);

        def.components.push_back({rom, {port}, {}, {}});
    }
    def.connections.push_back({{0, 0}, {1, 0}});
    const auto model = mhs::compiler::build_model(def);
    ASSERT_EQ(model.components.size(), 2u);
    EXPECT_TRUE(std::holds_alternative<mhs::model::RomDomain>(model.components[0].physics));
    EXPECT_TRUE(std::holds_alternative<mhs::model::RomDomain>(model.components[1].physics));
    EXPECT_EQ(model.layout.state_count, 2u);
    EXPECT_EQ(model.layout.fields[0].quantity, mhs::model::FieldQuantity::ReducedCoordinate);
    const mhs::core::Vector<double> initial {300, 300};
    const auto ops = mhs::solver::assemble_thermal(model, initial, 0);
    EXPECT_NEAR(ops.K.coeff(0, 0), 3, 1e-12);
    EXPECT_NEAR(ops.K.coeff(0, 1), -1, 1e-12);
    EXPECT_NEAR(ops.K.coeff(1, 0), -1, 1e-12);
    EXPECT_NEAR(ops.K.coeff(1, 1), 3, 1e-12);
    const auto flux = mhs::solver::evaluate_connection_flux(model, model.connections[0], initial, 0);
    ASSERT_EQ(flux.size(), 1u);
    EXPECT_NEAR(flux[0].outward_a, 0, 1e-12);
    const auto solution = mhs::solver::solve(model);
    ASSERT_TRUE(solution.converged);
    EXPECT_NEAR(solution.state[0], 295, 1e-8);
    EXPECT_NEAR(solution.state[1], 285, 1e-8);
    const auto solved_flux = mhs::solver::evaluate_connection_flux(model, model.connections[0], solution.state, 0);
    EXPECT_NEAR(solved_flux[0].outward_a, 10, 1e-8);
}

TEST(RomDomain, ExposedHtcPortUsesOrdinaryThermalBoundaryCoupling)
{
    using namespace mhs::model;
    ModelDefinition d;
    auto library = d.library_handle;
    RomDomainDefinition rom;
    rom.order = 1;
    rom.K = {2};
    rom.C = {1};
    rom.f = {0};
    rom.initial_state = {300};
    PortDefinition port;
    port.name = "exposed-htc-face";
    port.region = {Axis::Z, 1, {0, 0.1, 0, 0.1}};
    port.outward_sign = 1;
    RomPortPatch patch;
    patch.rectangle = {0, 0.1, 0, 0.1};
    patch.half_resistance = 0.005; // m^2 K/W, included in boundary condensation.
    patch.center = {0.05, 0.05};
    patch.trace = {1};
    port.patches.push_back(patch);

    SurfaceAssignment boundary;
    boundary.regions.push_back({Axis::Z, 1, {0, 0.1, 0, 0.1}});
    boundary.condition = library->add_thermal_boundary(ConvectionBoundary {"100", "300"});

    d.components.push_back({rom, {port}, {std::move(boundary)}, {}});
    const mhs::core::Vector<double> initial {300};

    const auto first = mhs::compiler::build_model(d);
    ASSERT_TRUE(first.connections.empty());
    ASSERT_EQ(first.rom_boundary_terms.size(), 1u);
    const auto first_ops = mhs::solver::assemble_thermal(first, initial, 0);
    const double first_g = 0.01 / (0.005 + 1.0 / 100.0);
    EXPECT_NEAR(first_ops.K.coeff(0, 0), 2 + first_g, 1e-12);
    EXPECT_NEAR(first_ops.f[0], first_g * 300, 1e-12);
    EXPECT_NEAR(mhs::solver::solve(first).state[0], 300 * first_g / (2 + first_g), 1e-8);

    // Reuse the same intrinsic ROM and physical port; only the ordinary
    // exterior boundary condition changes with the surrounding environment.
    d.component(0).thermal_boundaries[0].condition = library->add_thermal_boundary(ConvectionBoundary {"200", "300"});
    const auto second = mhs::compiler::build_model(d);
    const auto second_ops = mhs::solver::assemble_thermal(second, initial, 0);
    const double second_g = 0.01 / (0.005 + 1.0 / 200.0);
    EXPECT_NEAR(second_ops.K.coeff(0, 0), 2 + second_g, 1e-12);
    EXPECT_NEAR(second_ops.f[0], second_g * 300, 1e-12);
    EXPECT_NEAR(mhs::solver::solve(second).state[0], 300 * second_g / (2 + second_g), 1e-8);
    EXPECT_NEAR(mhs::solver::solve(first).state[0], 300 * first_g / (2 + first_g), 1e-8);

    // A connected face must never receive an additional external HTC law.
    auto invalid = d;
    RomDomainDefinition other = rom;
    auto other_port = port;

    other_port.outward_sign = -1;
    invalid.components.push_back({other, {other_port}, {}, {}});
    invalid.connections.push_back({{0, 0}, {1, 0}});
    EXPECT_THROW(mhs::compiler::build_model(invalid), std::invalid_argument);
}

TEST(RomDomain, NonmatchingFomRomNccRecoversLinearInterfaceTrace)
{
    using namespace mhs::model;
    ModelDefinition d;
    auto library = d.library_handle;
    d.add_component();
    MaterialSpec solid;
    solid.conductivity_x = solid.conductivity_y = solid.conductivity_z = "1";
    solid.density = solid.specific_heat = "1";
    library->add_material({"solid", solid});
    auto& fom = d.fom(0);
    fom.mesh.x_vertices = {0, 1};
    fom.mesh.y_vertices = {0, 0.5, 1};
    fom.mesh.z_vertices = {0, 1};
    fom.volumes
        = {{library->material_id("solid"), mhs::core::invalid_id, {{GeometryOperation::Add, {0, 0, 0}, {1, 1, 1}}}}};
    d.component(0).ports.push_back({"join", {Axis::X, 1, {0, 1, 0, 1}}});

    RomDomainDefinition rom;
    rom.order = 1;
    rom.K = {1};
    rom.C = {1};
    rom.f = {0};
    rom.initial_state = {1};
    PortDefinition port;
    port.name = "join";
    port.region = {Axis::X, 1, {0, 1, 0, 1}};
    port.outward_sign = -1;
    RomPortPatch patch;
    patch.rectangle = {0, 1, 0, 1};
    patch.half_resistance = 0.5;
    patch.center = {0.5, 0.5};
    // At z=1 the ROM center temperature is 305 K, with a physical
    // tangential gradient of 20 K/m. The FOM cells at y=.25 and y=.75
    // have temperatures 300 and 310 K, so the corrected flux is zero.
    patch.trace = {305};
    patch.tangential_gradient[0] = {20};
    patch.tangential_gradient[1] = {0};
    port.patches.push_back(patch);

    d.components.push_back({rom, {port}, {}, {}});
    d.connections = {{{0, 0}, {1, 0}}};

    const auto compiled = mhs::compiler::build_model(d);
    ASSERT_EQ(compiled.connections.size(), 1u);
    ASSERT_TRUE(compiled.connections.front().nonmatching);
    const mhs::core::Vector<double> state {300, 310, 1};
    const auto corrected = mhs::solver::evaluate_connection_flux(compiled, compiled.connections.front(), state, 0);
    ASSERT_EQ(corrected.size(), 2u);
    for (const auto& part : corrected)
        EXPECT_NEAR(part.outward_a, 0, 1e-12);

    auto uncorrected = d;
    uncorrected.component(1).ports[0].patches[0].tangential_gradient[0] = {0};
    const auto model_tpfa = mhs::compiler::build_model(uncorrected);
    const auto tpfa = mhs::solver::evaluate_connection_flux(model_tpfa, model_tpfa.connections.front(), state, 0);
    ASSERT_EQ(tpfa.size(), 2u);
    EXPECT_NEAR(tpfa[0].outward_a, -2.5, 1e-12);
    EXPECT_NEAR(tpfa[1].outward_a, 2.5, 1e-12);

    const auto ops = mhs::solver::assemble_thermal(compiled, state, 0);
    for (std::size_t i = 0; i < state.size(); ++i)
        for (std::size_t j = 0; j < state.size(); ++j)
            EXPECT_NEAR(ops.K.coeff(i, j), ops.K.coeff(j, i), 1e-12);
}

TEST(RomDomain, NonmatchingRomPortRequiresNccGradient)
{
    using namespace mhs::model;
    ModelDefinition d;
    auto library = d.library_handle;

    RomDomainDefinition left;
    left.order = 1;
    left.K = {1};
    left.C = {1};
    left.f = {300};
    PortDefinition port;
    port.name = "join";
    port.region = {Axis::X, 1, {0, 1, 0, 1}};
    port.outward_sign = 1;
    RomPortPatch patch;
    patch.rectangle = {0, 1, 0, 1};
    patch.half_resistance = 0.5;
    patch.center = {0.5, 0.5};
    patch.trace = {1};
    port.patches.push_back(patch);

    RomDomainDefinition right = left;
    auto right_port = port;
    right_port.outward_sign = -1;
    right_port.patches.clear();
    right_port.patches.push_back(patch);
    right_port.patches[0].rectangle = {0, 0.5, 0, 1};
    right_port.patches.push_back(patch);
    right_port.patches[1].rectangle = {0.5, 1, 0, 1};

    d.components.push_back({std::move(left), {port}, {}, {}});
    d.components.push_back({std::move(right), {right_port}, {}, {}});
    d.connections.push_back({{0, 0}, {1, 0}});
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
}
