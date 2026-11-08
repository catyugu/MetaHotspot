#include "compiler/model_compiler.hpp"
#include "solver/assembler.hpp"
#include "solver/connection_flux.hpp"
#include "solver/solver.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <cmath>
#include <gtest/gtest.h>

namespace {
    using namespace mhs::model;
    constexpr double q = 0.7071067811865475244;

    FomDomainDefinition block(std::initializer_list<double> x, bool first)
    {
        FomDomainDefinition d;
        d.mesh.x_vertices = x;
        d.mesh.y_vertices = d.mesh.z_vertices = {0.0, 1.0};
        d.volumes = {{0, mhs::core::invalid_id, {{GeometryOperation::Add, {0, 0, 0}, {3, 1, 1}}}}};
        d.default_thermal_boundary = NeumannBoundary {};
        d.thermal_boundaries.push_back(
            {{{Axis::X, first ? 0.0 : 3.0, {0, 1, 0, 1}}}, DirichletBoundary {first ? "300" : "280"}});
        d.ports.push_back({"interface", {Axis::X, 1.0, {0, 1, 0, 1}}});
        return d;
    }

    // Tiny independent dense solver for a single implicit-Euler update;
    // it avoids adaptive-step controller differences between FOM and ROM
    // coordinates when checking strict Galerkin dynamical equivalence.
    std::array<double, 3> implicit_euler_step(const mhs::solver::Operators& op,
        std::span<const double> initial, double dt)
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
        MaterialSpec mat;
        mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "1";
        mat.density = mat.specific_heat = "1";
        model.materials.push_back({"solid", mat});
        model.domains[0] = block({0.0, 1.0}, true);
        if (!reduced) {
            model.domains.push_back(block({1.0, 2.0, 3.0}, false));
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
            RomPort port;
            port.name = "interface";
            port.region = {Axis::X, 1.0, {0, 1, 0, 1}};
            port.outward_sign = -1;
            RomPortPatch facet;
            facet.rectangle = {0, 1, 0, 1};
            facet.half_resistance = 0.5;
            facet.center = {0.5, 0.5};
            facet.trace = {q, -q};
            port.patches.push_back(facet);
            rom.ports.push_back(port);
            rom.observations.push_back({{"top", 2.5, 0.5, 0.5}, {q, q}, 0});
            model.rom_domains.push_back(rom);
        }
        model.point_probes.push_back({"top", 2.5, 0.5, 0.5});
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
    ASSERT_EQ(rom.rom_domains.size(), 1u);
    EXPECT_EQ(rom.rom_domains[0].order, 2u);
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
    def.domains.clear();
    for (std::size_t i = 0; i < 2; ++i) {
        mhs::model::RomDomainDefinition rom;
        rom.order = 1;
        rom.K = {2};
        rom.C = {1};
        rom.f = {i == 0 ? 600.0 : 560.0};
        rom.initial_state = {300};
        mhs::model::RomPort port;
        port.name = "join";
        port.region = {mhs::model::Axis::X, 1, {0, 1, 0, 1}};
        port.outward_sign = i == 0 ? 1 : -1;
        mhs::model::RomPortPatch patch;
        patch.rectangle = {0, 1, 0, 1};
        patch.half_resistance = 0.5;
        patch.center = {0.5, 0.5};
        patch.trace = {1};
        port.patches.push_back(patch);
        rom.ports.push_back(port);
        def.rom_domains.push_back(rom);
    }
    def.connections.push_back({{0, 0}, {1, 0}});
    const auto model = mhs::compiler::build_model(def);
    EXPECT_TRUE(model.domains.empty());
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

TEST(RomDomain, AffineHtcTermsReuseOneReducedSystemForDifferentBoundaryValues)
{
    mhs::model::ModelDefinition d;
    d.domains.clear();
    mhs::model::RomDomainDefinition rom;
    rom.order = 1;
    rom.K = {2};
    rom.C = {1};
    rom.f = {0};
    rom.initial_state = {300};
    // A projected convection term: K_h = A, f_h = A*T_ambient,
    // with physical area A=0.01 m² and ambient temperature 300 K.
    rom.affine_terms.push_back({"h", 100, {0.01}, {3}});
    d.rom_domains.push_back(rom);
    const mhs::core::Vector<double> initial {300};
    const auto first = mhs::compiler::build_model(d);
    const auto first_ops = mhs::solver::assemble_thermal(first, initial, 0);
    EXPECT_NEAR(first_ops.K.coeff(0, 0), 3, 1e-12);
    EXPECT_NEAR(first_ops.f[0], 300, 1e-12);
    const auto first_solution = mhs::solver::solve(first);
    ASSERT_TRUE(first_solution.converged);
    EXPECT_NEAR(first_solution.state[0], 100, 1e-8);

    // No new basis or trace extraction: only change the parameter coefficient
    // and construct another immutable compiled snapshot.
    d.rom_domains[0].affine_terms[0].value = 200;
    const auto second = mhs::compiler::build_model(d);
    const auto second_ops = mhs::solver::assemble_thermal(second, initial, 0);
    EXPECT_NEAR(second_ops.K.coeff(0, 0), 4, 1e-12);
    EXPECT_NEAR(second_ops.f[0], 600, 1e-12);
    const auto second_solution = mhs::solver::solve(second);
    ASSERT_TRUE(second_solution.converged);
    EXPECT_NEAR(second_solution.state[0], 150, 1e-8);
    const auto first_again = mhs::solver::solve(first);
    EXPECT_NEAR(first_again.state[0], first_solution.state[0], 1e-12);
}

TEST(RomDomain, NonmatchingFomRomNccRecoversLinearInterfaceTrace)
{
    using namespace mhs::model;
    ModelDefinition d;
    MaterialSpec solid;
    solid.conductivity_x = solid.conductivity_y = solid.conductivity_z = "1";
    solid.density = solid.specific_heat = "1";
    d.materials.push_back({"solid", solid});
    auto& fom = d.domains[0];
    fom.mesh.x_vertices = {0, 1};
    fom.mesh.y_vertices = {0, 0.5, 1};
    fom.mesh.z_vertices = {0, 1};
    fom.volumes = {{0, mhs::core::invalid_id, {{GeometryOperation::Add, {0, 0, 0}, {1, 1, 1}}}}};
    fom.ports.push_back({"join", {Axis::X, 1, {0, 1, 0, 1}}});

    RomDomainDefinition rom;
    rom.order = 1;
    rom.K = {1};
    rom.C = {1};
    rom.f = {0};
    rom.initial_state = {1};
    RomPort port;
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
    rom.ports.push_back(port);
    d.rom_domains.push_back(rom);
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
    uncorrected.rom_domains[0].ports[0].patches[0].tangential_gradient[0] = {0};
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
    mhs::model::ModelDefinition d;
    d.domains.clear();
    for (int side = 0; side < 2; ++side) {
        mhs::model::RomDomainDefinition rom;
        rom.order = 1;
        rom.K = {1};
        rom.C = {1};
        rom.f = {300};
        mhs::model::RomPort port;
        port.name = "join";
        port.region = {mhs::model::Axis::X, 1, {0, 1, 0, 1}};
        port.outward_sign = side ? -1 : 1;
        for (int i = 0; i < (side ? 2 : 1); ++i) {
            mhs::model::RomPortPatch patch;
            patch.rectangle = {side ? 0.5 * i : 0, side ? 0.5 * (i + 1) : 1, 0, 1};
            patch.half_resistance = 0.5;
            patch.center = {0.5, 0.5};
            patch.trace = {1};
            port.patches.push_back(patch);
        }
        rom.ports.push_back(port);
        d.rom_domains.push_back(rom);
    }
    d.connections.push_back({{0, 0}, {1, 0}});
    EXPECT_THROW(mhs::compiler::build_model(d), std::invalid_argument);
}
