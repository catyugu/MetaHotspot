#include "compiler/model_compiler.hpp"
#include "solver/assembler.hpp"
#include "solver/connection_flux.hpp"
#include "solver/solver.hpp"
#include <array>
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
        d.volumes = {{0, mhs::core::invalid_id,
            {{GeometryOperation::Add, {0, 0, 0}, {3, 1, 1}}}}};
        d.default_thermal_boundary = NeumannBoundary {};
        d.thermal_boundaries.push_back(
            {{{Axis::X, first ? 0.0 : 3.0, {0, 1, 0, 1}}},
                DirichletBoundary {first ? "300" : "280"}});
        d.ports.push_back({"interface", {Axis::X, 1.0, {0, 1, 0, 1}}});
        return d;
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

    const mhs::model::Study transient {mhs::model::StudyType::Transient, 300, 0.5, 0.25};
    const auto ft = mhs::solver::solve(fom, transient);
    const auto zt = mhs::solver::solve(rom, transient);
    ASSERT_TRUE(ft.converged);
    ASSERT_TRUE(zt.converged);
    ASSERT_EQ(ft.snapshot_times.size(), zt.snapshot_times.size());
    for (std::size_t t = 0; t < ft.snapshot_times.size(); ++t) {
        EXPECT_NEAR(ft.snapshot_times[t], zt.snapshot_times[t], 1e-12);
        const auto offset = t * 3;
        EXPECT_NEAR(ft.snapshot_states[offset], zt.snapshot_states[offset], 2e-6);
        EXPECT_NEAR(ft.snapshot_states[offset + 1],
            q * (zt.snapshot_states[offset + 1] - zt.snapshot_states[offset + 2]), 2e-6);
        EXPECT_NEAR(ft.snapshot_states[offset + 2],
            q * (zt.snapshot_states[offset + 1] + zt.snapshot_states[offset + 2]), 2e-6);
        EXPECT_NEAR(ft.probe_traces[0].values[t], zt.probe_traces[0].values[t], 2e-6);
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
    const auto solved_flux
        = mhs::solver::evaluate_connection_flux(model, model.connections[0], solution.state, 0);
    EXPECT_NEAR(solved_flux[0].outward_a, 10, 1e-8);
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
