#include "core/vector.hpp"
#include "core/vector_ops.hpp"
#include "pta/io.hpp"
#include "pta_test_utils.hpp"
#include "solver/assembler.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace {

    mhs::model::Model load_microfluid_case()
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
        d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
        MaterialSpec mat;
        mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "1";
        mat.density = mat.specific_heat = "1";
        mat.kind = MaterialKind::Fluid;
        mat.dynamic_viscosity = "1";
        d.model.materials.push_back({"fluid", mat});
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
    EXPECT_TRUE(model.fluid.connections.empty());
    EXPECT_TRUE(model.fluid.interface_heat_transfer_factor.empty());
    EXPECT_TRUE(model.fluid.boundary_terms.empty());
}

TEST(FluidModuleTest, FrozenFluxConservesMassAcrossCellsAndBoundaries)
{
    const auto model = load_microfluid_case();
    ASSERT_FALSE(model.fluid.connections.empty());
    mhs::core::Vector<double> balance(model.cells.size(), 0);
    auto density = [&](mhs::model::CellId cell) {
        const auto p = cell_geometry(model.mesh, cell).center();
        return model.material_table[model.cells[cell].material].rho.eval(
            {p[0], p[1], p[2], model.reference_temperature, 0});
    };
    bool nonzero = false;
    for (const auto& flow : model.fluid.connections) {
        const auto [a, b] = flow.cells;
        EXPECT_TRUE(model.cells[a].is_fluid);
        EXPECT_TRUE(model.cells[b].is_fluid);
        const double mass = flow.volume_flux * (density(a) + density(b)) / 2;
        nonzero = nonzero || std::abs(mass) > 1e-20;
        balance[a] += mass;
        balance[b] -= mass;
    }
    double boundary_sum = 0;
    for (const auto& term : model.fluid.boundary_terms) {
        const double mass = std::visit(
            [](const auto& flow) {
                using T = std::decay_t<decltype(flow)>;
                EXPECT_GE(flow.mass_flow_rate, 0);
                if constexpr (std::is_same_v<T, mhs::model::FluidInflow>)
                    return -flow.mass_flow_rate;
                else
                    return flow.mass_flow_rate;
            },
            term.condition);
        balance[model.boundary_facets[term.facet].owner()] += mass;
        boundary_sum += mass;
    }
    for (auto cell : model.fluid.fluid_to_global)
        EXPECT_NEAR(balance[cell], 0, 1e-12);
    EXPECT_NEAR(boundary_sum, 0, 1e-12);
    EXPECT_TRUE(nonzero);
    const mhs::core::Vector<double> state(model.cells.size(), 300);
    // Suppress diffusion/sources to inspect advection through the main assembler.
    auto transport = model;
    std::fill(transport.boundary_thermal_condition.begin(), transport.boundary_thermal_condition.end(),
        mhs::core::invalid_id);
    transport.source_terms.clear();
    for (auto& material : transport.material_table)
        material.kx = material.ky = material.kz = mhs::expression::CompiledExpression::make_constant(0);
    const auto normal = mhs::solver::assemble_thermal(transport, state, 0);
    auto scaled = transport;
    constexpr double factor = 1e-20;
    for (auto& flow : scaled.fluid.connections)
        flow.volume_flux *= factor;
    for (auto& term : scaled.fluid.boundary_terms)
        std::visit([](auto& flow) { flow.mass_flow_rate *= factor; }, term.condition);
    const auto tiny = mhs::solver::assemble_thermal(scaled, state, 0);
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
    d.model.fluid_boundaries = {
        {{{Axis::X, 0, {0, 0.2, 0, 1}}, {Axis::X, 0, {0.1, 0.2, 0, 1}}}, MassFlowInlet {2, 280}},
        {{{Axis::X, 0, {0.8, 1, 0, 1}}}, MassFlowInlet {3, 320}}, {{{Axis::X, 1, {0, 1, 0, 1}}}, PressureOutlet {0}}};
    auto verify = [&](const mhs::model::Model& model) {
        const auto inc = mhs::solver::assemble_thermal(model, mhs::core::Vector<double> {300}, 0);
        const double diagonal = inc.K.coeff(0, 0);
        EXPECT_NEAR(diagonal, 5, 1e-12);
        EXPECT_NEAR(inc.f[0], 2 * 280 + 3 * 320, 1e-10);
    };
    verify(test_pta::build_model(d));
    ThermalBoundarySpec thermal;
    thermal.condition = NeumannBoundary {"0"};
    thermal.regions.push_back({Axis::X, 0, {0, 0.15, 0, 1}});
    d.model.thermal_boundaries.push_back(thermal);
    const auto refined = test_pta::build_model(d);
    verify(refined);

    // One physical X- side is split by both hydraulic and thermal rectangles.
    // Boundary subdivision retains each selected external piece exactly once.
    std::size_t side_facets = 0;
    double side_area = 0.0;
    bool has_cross_cut = false;
    for (const auto& facet : refined.boundary_facets) {
        if (facet.cell != 0 || facet.axis != Axis::X || facet.outward_sign() != -1)
            continue;
        ++side_facets;
        side_area += facet.area();
        has_cross_cut = has_cross_cut || (facet.bounds[0] == 0.1 && facet.bounds[1] == 0.15);
    }
    EXPECT_EQ(side_facets, 5u);
    EXPECT_NEAR(side_area, 1.0, 1e-14);
    EXPECT_TRUE(has_cross_cut);
    EXPECT_EQ(refined.fluid.boundary_terms.size(), 5u);
}

TEST(FluidModuleTest, PressureInletPumpResistanceIsLumpedAcrossFacetSplits)
{
    using namespace mhs::model;
    auto d = single_fluid_cell();
    d.model.fluid_boundaries.push_back({{{Axis::X, 0, {0, 1, 0, 1}}}, PressureInlet {200, 280, 0}});
    d.model.fluid_boundaries.push_back({{{Axis::X, 1, {0, 1, 0, 1}}}, PressureOutlet {0}});

    auto total_inflow = [](const mhs::model::Model& model) {
        double inflow = 0, outflow = 0;
        for (const auto& term : model.fluid.boundary_terms) {
            if (const auto* flow = std::get_if<FluidInflow>(&term.condition)) {
                EXPECT_DOUBLE_EQ(flow->temperature, 280);
                inflow += flow->mass_flow_rate;
            }
            else
                outflow += std::get<FluidOutflow>(term.condition).mass_flow_rate;
        }
        EXPECT_NEAR(inflow, outflow, 1e-7);
        return inflow;
    };

    const auto unrestricted = test_pta::build_model(d);
    const double q0 = total_inflow(unrestricted);
    ASSERT_GT(q0, 0);
    // Match the baseline hydraulic resistance so the pump halves the flow.
    const double resistance = 200.0 / q0;
    std::get<PressureInlet>(d.model.fluid_boundaries.front().condition).pump_resistance = resistance;
    const double expected = q0 / 2;
    const auto restricted = test_pta::build_model(d);
    EXPECT_NEAR(total_inflow(restricted), expected, expected * 1e-7);

    // A thermal cut subdivides the same hydraulic inlet into multiple facets.
    // The external pump resistance must still act on their combined flow.
    ThermalBoundarySpec thermal;
    thermal.condition = NeumannBoundary {"0"};
    thermal.regions.push_back({Axis::X, 0, {0, 0.25, 0, 1}});
    d.model.thermal_boundaries.push_back(thermal);
    const auto split = test_pta::build_model(d);
    EXPECT_GT(split.fluid.boundary_terms.size(), restricted.fluid.boundary_terms.size());
    EXPECT_NEAR(total_inflow(split), expected, expected * 1e-7);
}

TEST(FluidModuleTest, RejectsInvalidPumpResistance)
{
    using namespace mhs::model;
    auto d = single_fluid_cell();
    d.model.fluid_boundaries.push_back({{{Axis::X, 0, {0, 1, 0, 1}}}, PressureInlet {200, 280, 0}});
    d.model.fluid_boundaries.push_back({{{Axis::X, 1, {0, 1, 0, 1}}}, PressureOutlet {0}});
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const double invalid : {-1.0, infinity, nan}) {
        std::get<PressureInlet>(d.model.fluid_boundaries.front().condition).pump_resistance = invalid;
        EXPECT_THROW(test_pta::build_model(d), std::invalid_argument);
    }
}

TEST(FluidModuleTest, PressureReversalUsesActualInflowTemperature)
{
    using namespace mhs::model;
    auto d = single_fluid_cell();
    // Pressure labels do not determine thermal flow direction: the nominal
    // outlet supplies fluid when its pressure is higher than the inlet's.
    d.model.fluid_boundaries
        = {{{{Axis::X, 0, {0, 1, 0, 1}}}, PressureInlet {0, 280}}, {{{Axis::X, 1, {0, 1, 0, 1}}}, PressureOutlet {1}}};
    auto reverse = test_pta::build_model(d);
    ASSERT_EQ(reverse.fluid.boundary_terms.size(), 2);
    for (const auto& term : reverse.fluid.boundary_terms) {
        if (const auto* inflow = std::get_if<mhs::model::FluidInflow>(&term.condition))
            EXPECT_DOUBLE_EQ(inflow->temperature, 300);
    }
    EXPECT_EQ(std::ranges::count_if(reverse.fluid.boundary_terms,
                  [](const auto& term) { return std::holds_alternative<mhs::model::FluidInflow>(term.condition); }),
        1);
    auto verify_reverse = [&](const mhs::model::Model& model, double temperature) {
        const auto inc = mhs::solver::assemble_thermal(model, mhs::core::Vector<double> {300}, 0);
        const double diagonal = inc.K.coeff(0, 0);
        ASSERT_GT(diagonal, 0);
        EXPECT_NEAR(inc.f[0], diagonal * temperature, 1e-10);
    };
    verify_reverse(reverse, 300);

    std::get<PressureOutlet>(d.model.fluid_boundaries.back().condition).pressure = 0;
    const auto stagnant = test_pta::build_model(d);
    const auto zero = mhs::solver::assemble_thermal(stagnant, mhs::core::Vector<double> {300}, 0);
    EXPECT_DOUBLE_EQ(zero.f[0], 0);
    EXPECT_DOUBLE_EQ(mhs::core::norm_inf(zero.K.values()), 0);
}
