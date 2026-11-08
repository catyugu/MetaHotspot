#include "core/vector.hpp"
#include "core/vector_ops.hpp"
#include "pta/io.hpp"
#include "pta_test_utils.hpp"
#include "solver/assembler.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <type_traits>

namespace {

    test_pta::Model load_microfluid_case()
    {
        const std::string root = MHS_SOURCE_DIR;
        const std::string input = root + "/cases/microfluid_cases/steady_case1.xml";
        EXPECT_TRUE(std::filesystem::exists(input));
        return test_pta::build_model(mhs::app::pta::read_xml(input));
    }

    mhs::app::pta::Definition single_fluid_cell()
    {
        using namespace mhs::model;
        mhs::app::pta::Definition d;
        d.length_unit = mhs::app::pta::LengthUnit::Meter;
        auto& mesh = d.model.fom(0).mesh;
        mesh.x_vertices = mesh.y_vertices = mesh.z_vertices = {0, 1};
        MaterialSpec mat;
        mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "1";
        mat.density = mat.specific_heat = "1";
        mat.dynamic_viscosity = "1";
        d.model.library->materials.push_back({"fluid", mat});
        mhs::app::pta::LayerSpec layer;
        layer.thickness = "1";
        mhs::app::pta::BlockSpec block;
        block.material = "fluid";
        block.volumetric_heat_source = "0";
        block.geometry.push_back({GeometryOperation::Add, {"0", "0", "1", "1"}});
        layer.blocks.push_back(block);
        d.layers.push_back(layer);
        return d;
    }

} // namespace

TEST(FluidModuleTest, SolidOnlyModelKeepsFluidDomainEmpty)
{
    const std::string input = std::string(MHS_SOURCE_DIR) + "/cases/simple_steady_cases/simple_steady_case1.xml";
    auto model = test_pta::build_model(mhs::app::pta::read_xml(input));

    EXPECT_TRUE(model.fluid.fluid_to_global.empty());
    EXPECT_TRUE(model.fluid.global_to_fluid.empty());
    EXPECT_TRUE(model.flow.face_mass_flow.empty());
    EXPECT_TRUE(model.flow.interface_heat_transfer_factor.empty());
    EXPECT_TRUE(model.flow.boundary_terms.empty());
}

TEST(FluidModuleTest, FrozenFluxConservesMassAcrossCellsAndBoundaries)
{
    const auto model = load_microfluid_case();
    ASSERT_EQ(model.flow.face_mass_flow.size(), model.discrete.faces.size());
    mhs::core::Vector<double> balance(model.discrete.cells.size(), 0);
    bool nonzero = false;
    for (std::size_t id = 0; id < model.discrete.faces.size(); ++id) {
        const auto& f = model.discrete.faces[id];
        if (f.exterior())
            continue;
        const double mass = model.flow.face_mass_flow[id];
        nonzero = nonzero || std::abs(mass) > 1e-20;
        balance[f.minus] += mass;
        balance[f.plus] -= mass;
    }
    double boundary_sum = 0;
    for (const auto& term : model.flow.boundary_terms) {
        const double mass = std::visit(
            [](const auto& flow) {
                using T = std::decay_t<decltype(flow)>;
                EXPECT_GE(flow.mass_flow_rate, 0);
                if constexpr (std::is_same_v<T, mhs::solver::FluidInflow>)
                    return -flow.mass_flow_rate;
                else
                    return flow.mass_flow_rate;
            },
            term.condition);
        balance[term.cell] += mass;
        boundary_sum += mass;
    }
    for (auto cell : model.fluid.fluid_to_global)
        EXPECT_NEAR(balance[cell], 0, 1e-12);
    EXPECT_NEAR(boundary_sum, 0, 1e-12);
    EXPECT_TRUE(nonzero);
    const mhs::core::Vector<double> state(model.discrete.cells.size(), 300);
    // Suppress diffusion/sources to inspect advection through the main assembler.
    auto transport = model;
    transport.boundary_faces.clear();
    transport.source_terms.clear();
    auto library = std::make_shared<mhs::model::CompiledLibrary>(*transport.library);
    for (auto& material : library->material_table)
        material.kx = material.ky = material.kz = mhs::expression::CompiledExpression::make_constant(0);
    transport.library = std::move(library);
    mhs::solver::ThermalAssembler normal_assembler(transport, transport.flow);
    const auto& normal = normal_assembler.assemble(state, 0);
    auto scaled = transport;
    constexpr double factor = 1e-20;
    for (auto& flux : scaled.flow.face_mass_flow)
        flux *= factor;
    for (auto& term : scaled.flow.boundary_terms)
        std::visit([](auto& flow) { flow.mass_flow_rate *= factor; }, term.condition);
    mhs::solver::ThermalAssembler tiny_assembler(scaled, scaled.flow);
    const auto tiny = tiny_assembler.assemble(state, 0);
    auto compare_scaled = [&](std::span<const double> small, std::span<const double> full) {
        ASSERT_EQ(small.size(), full.size());
        double error2 = 0, norm2 = 0;
        for (std::size_t i = 0; i < full.size(); ++i) {
            const double diff = small[i] / factor - full[i];
            error2 += diff * diff;
            norm2 += full[i] * full[i];
        }
        EXPECT_LE(std::sqrt(error2), std::sqrt(norm2) * 1e-12);
    };
    EXPECT_TRUE(std::equal(tiny.K.row_ptr().begin(), tiny.K.row_ptr().end(), normal.K.row_ptr().begin()));
    EXPECT_TRUE(std::equal(tiny.K.col_idx().begin(), tiny.K.col_idx().end(), normal.K.col_idx().begin()));
    compare_scaled(tiny.K.values(), normal.K.values());
    compare_scaled(tiny.f, normal.f);
}

// Two disjoint inlet pieces and an outlet share a single control volume.
// Their enthalpy contributions must survive independent projection.
TEST(FluidModuleTest, PartialInletsAccumulateAndIgnoreThermalSubdivision)
{
    using namespace mhs::model;
    auto d = single_fluid_cell();
    const auto inlet_a = d.model.library->add_fluid_boundary(MassFlowInlet {2, 280});
    const auto inlet_b = d.model.library->add_fluid_boundary(MassFlowInlet {3, 320});
    const auto outlet = d.model.library->add_fluid_boundary(PressureOutlet {0});
    d.model.fom(0).fluid_boundaries = {
        {{{Axis::X, 0, {0, 0.2, 0, 1}}, {Axis::X, 0, {0.1, 0.2, 0, 1}}}, inlet_a},
        {{{Axis::X, 0, {0.8, 1, 0, 1}}}, inlet_b}, {{{Axis::X, 1, {0, 1, 0, 1}}}, outlet}};
    auto verify = [&](const test_pta::Model& model) {
        mhs::solver::ThermalAssembler assembler(model, model.flow);
        const auto& inc = assembler.assemble(mhs::core::Vector<double> {300}, 0);
        const double diagonal = inc.K.coeff(0, 0);
        EXPECT_NEAR(diagonal, 5, 1e-12);
        EXPECT_NEAR(inc.f[0], 2 * 280 + 3 * 320, 1e-10);
    };
    verify(test_pta::build_model(d));
    ThermalBoundarySpec thermal;
    thermal.condition = d.model.library->add_thermal_boundary(NeumannBoundary {"0"});
    thermal.regions.push_back({Axis::X, 0, {0, 0.15, 0, 1}});
    d.model.component(0).thermal_boundaries.push_back(thermal);
    verify(test_pta::build_model(d));
}

TEST(FluidModuleTest, PressureReversalUsesActualInflowTemperature)
{
    using namespace mhs::model;
    auto d = single_fluid_cell();
    // Pressure labels do not determine thermal flow direction: the nominal
    // outlet supplies fluid when its pressure is higher than the inlet's.
    const auto inlet = d.model.library->add_fluid_boundary(PressureInlet {0, 280});
    const auto outlet = d.model.library->add_fluid_boundary(PressureOutlet {1, 315});
    d.model.fom(0).fluid_boundaries
        = {{{{Axis::X, 0, {0, 1, 0, 1}}}, inlet}, {{{Axis::X, 1, {0, 1, 0, 1}}}, outlet}};
    auto reverse = test_pta::build_model(d);
    ASSERT_EQ(reverse.flow.boundary_terms.size(), 2);
    for (const auto& term : reverse.flow.boundary_terms) {
        if (const auto* inflow = std::get_if<mhs::solver::FluidInflow>(&term.condition))
            EXPECT_DOUBLE_EQ(inflow->temperature, 315);
    }
    EXPECT_EQ(std::ranges::count_if(reverse.flow.boundary_terms,
                  [](const auto& term) { return std::holds_alternative<mhs::solver::FluidInflow>(term.condition); }),
        1);
    auto verify_reverse = [&](const test_pta::Model& model, double temperature) {
        mhs::solver::ThermalAssembler assembler(model, model.flow);
        const auto& inc = assembler.assemble(mhs::core::Vector<double> {300}, 0);
        const double diagonal = inc.K.coeff(0, 0);
        ASSERT_GT(diagonal, 0);
        EXPECT_NEAR(inc.f[0], diagonal * temperature, 1e-10);
    };
    verify_reverse(reverse, 315);

    std::get<PressureOutlet>(
        d.model.library->fluid_boundaries[d.model.fom(0).fluid_boundaries.back().condition])
        .pressure = 0;
    const auto stagnant = test_pta::build_model(d);
    mhs::solver::ThermalAssembler stagnant_assembler(stagnant, stagnant.flow);
    const auto& zero = stagnant_assembler.assemble(mhs::core::Vector<double> {300}, 0);
    EXPECT_DOUBLE_EQ(zero.f[0], 0);
    EXPECT_DOUBLE_EQ(mhs::core::norm_inf(zero.K.values()), 0);
}
