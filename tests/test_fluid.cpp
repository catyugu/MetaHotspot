#include "compiler/model_compiler.hpp"
#include "config.h"
#include "io/model_io.hpp"
#include "solver/fluid_assembler.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace {

    mhs::core::Model load_microfluid_case()
    {
        const std::string root = PROJECT_SOURCE_DIR;
        const std::string input = root + "/cases/microfluid_cases/steady_case1.xml";
        EXPECT_TRUE(std::filesystem::exists(input));
        return mhs::sim::build_model(mhs::io::read_xml(input));
    }

    mhs::model::ModelDefinition single_fluid_cell()
    {
        using namespace mhs::model;
        ModelDefinition d;
        d.settings.length_unit = LengthUnit::Meter;
        d.mesh.x_vertices = d.mesh.y_vertices = d.mesh.z_vertices = {0, 1};
        MaterialSpec mat;
        mat.conductivity_x = mat.conductivity_y = mat.conductivity_z = "1";
        mat.density = mat.specific_heat = "1";
        mat.is_fluid = true;
        mat.dynamic_viscosity = "1";
        d.materials.push_back({"fluid", mat});
        LayerSpec layer;
        layer.thickness = "1";
        BlockSpec block;
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
    const std::string input = std::string(PROJECT_SOURCE_DIR) + "/cases/simple_steady_cases/simple_steady_case1.xml";
    auto model = mhs::sim::build_model(mhs::io::read_xml(input));

    EXPECT_TRUE(model.fluid.fluid_to_global.empty());
    EXPECT_TRUE(model.fluid.global_to_fluid.empty());
    EXPECT_TRUE(model.fluid.face_volume_flux.empty());
    EXPECT_TRUE(model.fluid.interface_heat_transfer_factor.empty());
    EXPECT_TRUE(model.fluid.boundary_terms.empty());
}

TEST(FluidModuleTest, FrozenFluxConservesMassAcrossCellsAndBoundaries)
{
    const auto model = load_microfluid_case();
    ASSERT_EQ(model.fluid.face_volume_flux.size(), model.discrete.faces.size());
    std::vector<double> balance(model.discrete.cells.size(), 0);
    auto density = [&](mhs::core::CellId cell) {
        const auto p = model.discrete.cells[cell.value].center();
        return model.material_table[model.cell_material[cell.value]].rho.eval(
            {p[0], p[1], p[2], model.initial_temperature, 0});
    };
    bool nonzero = false;
    for (mhs::core::Index id = 0; id < model.discrete.faces.size(); ++id) {
        const auto& f = model.discrete.faces[id];
        if (!f.neighbor.valid())
            continue;
        const double mass = model.fluid.face_volume_flux[id] * (density(f.owner) + density(f.neighbor)) / 2;
        nonzero = nonzero || std::abs(mass) > 1e-20;
        balance[f.owner.value] += mass;
        balance[f.neighbor.value] -= mass;
    }
    double boundary_sum = 0;
    for (const auto& term : model.fluid.boundary_terms) {
        const double mass = std::visit(
            [](const auto& flow) {
                using T = std::decay_t<decltype(flow)>;
                EXPECT_GE(flow.mass_flow_rate, 0);
                if constexpr (std::is_same_v<T, mhs::core::FluidInflow>)
                    return -flow.mass_flow_rate;
                else
                    return flow.mass_flow_rate;
            },
            term.condition);
        balance[model.discrete.faces[term.face.value].owner.value] += mass;
        boundary_sum += mass;
    }
    for (auto cell : model.fluid.fluid_to_global)
        EXPECT_NEAR(balance[cell], 0, 1e-12);
    EXPECT_NEAR(boundary_sum, 0, 1e-12);
    EXPECT_TRUE(nonzero);
}

// Two disjoint inlet pieces and an outlet share a single control volume.
// Their enthalpy contributions must survive independent projection.
TEST(FluidModuleTest, PartialInletsAccumulateAndIgnoreThermalSubdivision)
{
    using namespace mhs::model;
    auto d = single_fluid_cell();
    d.fluid_boundaries
        = {{{{Axis::X, 0, {{0, 0.2, 0, 1}, {0.1, 0.2, 0, 1}}}}, FluidBoundaryKind::MassFlowInlet, 2, 280},
            {{{Axis::X, 0, {{0.8, 1, 0, 1}}}}, FluidBoundaryKind::MassFlowInlet, 3, 320},
            {{{Axis::X, 1, {{0, 1, 0, 1}}}}, FluidBoundaryKind::Outlet, 0}};
    auto verify = [&](const mhs::core::Model& model) {
        const auto inc = mhs::sim::fluid::assemble_increment(model, std::vector<double> {300}, 0);
        double diagonal = 0;
        for (const auto& e : inc.matrix_entries)
            diagonal += e.value();
        EXPECT_NEAR(diagonal, 5, 1e-12);
        EXPECT_NEAR(inc.rhs[0], 2 * 280 + 3 * 320, 1e-10);
    };
    verify(mhs::sim::build_model(d));
    BoundaryPatch thermal;
    thermal.condition = NeumannBoundary {"0"};
    thermal.regions.push_back({Axis::X, 0, {{0, 0.15, 0, 1}}});
    d.boundaries.push_back(thermal);
    verify(mhs::sim::build_model(d));
}

TEST(FluidModuleTest, PressureReversalUsesActualInflowTemperature)
{
    using namespace mhs::model;
    auto d = single_fluid_cell();
    // Pressure labels do not determine thermal flow direction: the nominal
    // outlet supplies fluid when its pressure is higher than the inlet's.
    d.fluid_boundaries = {{{{Axis::X, 0, {{0, 1, 0, 1}}}}, FluidBoundaryKind::PressureInlet, 0, 280},
        {{{Axis::X, 1, {{0, 1, 0, 1}}}}, FluidBoundaryKind::Outlet, 1, 310}};
    auto reverse = mhs::sim::build_model(d);
    ASSERT_EQ(reverse.fluid.boundary_terms.size(), 2);
    for (const auto& term : reverse.fluid.boundary_terms) {
        const auto& face = reverse.discrete.faces[term.face.value];
        if (face.sign > 0) {
            ASSERT_TRUE(std::holds_alternative<mhs::core::FluidInflow>(term.condition));
            EXPECT_DOUBLE_EQ(std::get<mhs::core::FluidInflow>(term.condition).temperature, 310);
        }
        else {
            EXPECT_TRUE(std::holds_alternative<mhs::core::FluidOutflow>(term.condition));
        }
    }
    auto verify_reverse = [&](const mhs::core::Model& model, double temperature) {
        const auto inc = mhs::sim::fluid::assemble_increment(model, std::vector<double> {300}, 0);
        double diagonal = 0;
        for (const auto& e : inc.matrix_entries)
            diagonal += e.value();
        ASSERT_GT(diagonal, 0);
        EXPECT_NEAR(inc.rhs[0], diagonal * temperature, 1e-10);
    };
    verify_reverse(reverse, 310);
    d.fluid_boundaries.back().inlet_temperature = std::numeric_limits<double>::quiet_NaN();
    verify_reverse(mhs::sim::build_model(d), 300);

    d.fluid_boundaries.back().value = 0;
    const auto stagnant = mhs::sim::build_model(d);
    const auto zero = mhs::sim::fluid::assemble_increment(stagnant, std::vector<double> {300}, 0);
    EXPECT_DOUBLE_EQ(zero.rhs[0], 0);
    for (const auto& e : zero.matrix_entries)
        EXPECT_DOUBLE_EQ(e.value(), 0);
}
