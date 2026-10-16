#include "compiler/model_compiler.hpp"
#include "config.h"
#include "io/model_io.hpp"
#include "solver/fluid_assembler.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

namespace {

    mhs::core::Model load_microfluid_case()
    {
        const std::string root = PROJECT_SOURCE_DIR;
        const std::string input = root + "/cases/microfluid_cases/steady_case1.xml";
        EXPECT_TRUE(std::filesystem::exists(input));
        return mhs::sim::build_model(mhs::io::read_xml(input));
    }

    /// Build the default (uniform initial_temperature) state vector.
    std::vector<double> default_state(const mhs::core::Model& model)
    { return std::vector<double>(static_cast<std::size_t>(model.dofs.size()), model.initial_temperature); }

} // namespace

TEST(FluidModuleTest, PreprocessorStoresOnlyAssemblyReadyFields)
{
    auto model = load_microfluid_case();
    const auto& fluid = model.fluid;

    ASSERT_EQ(fluid.fluid_to_global.size(), 3200u);
    EXPECT_EQ(fluid.global_to_fluid.size(), model.cells.material_id.size());
    EXPECT_EQ(fluid.face_volume_flux.size(), model.discrete.faces.size());
    EXPECT_EQ(fluid.interface_heat_transfer_factor.size(), fluid.fluid_to_global.size());
    EXPECT_FALSE(fluid.boundary_terms.empty());

    EXPECT_TRUE(std::any_of(fluid.face_volume_flux.begin(), fluid.face_volume_flux.end(),
        [](double value) { return std::abs(value) > 0.0; }));
    EXPECT_TRUE(std::all_of(fluid.interface_heat_transfer_factor.begin(), fluid.interface_heat_transfer_factor.end(),
        [](double value) { return value > 0.0; }));
}

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

TEST(FluidModuleTest, FrozenFaceFluxIsAntisymmetric)
{
    auto model = load_microfluid_case();

    ASSERT_EQ(model.fluid.face_volume_flux.size(), model.discrete.faces.size());
    double global_flux = 0;
    bool nonzero = false;
    for (mhs::core::Index fi = 0; fi < model.fluid.fluid_to_global.size(); ++fi) {
        const mhs::core::CellId cell {model.fluid.fluid_to_global[fi]};
        double net = 0;
        bool exterior = false;
        for (auto id : model.discrete.incident_faces(cell)) {
            const auto& f = model.discrete.faces[id.value];
            if (!f.neighbor.valid())
                exterior = true;
            const double q = model.fluid.face_volume_flux[id.value] * (f.owner == cell ? 1 : -1);
            nonzero = nonzero || std::abs(q) > 1e-20;
            net += q;
        }
        global_flux += net;
        if (!exterior)
            EXPECT_NEAR(net, 0, 1e-12);
    }
    EXPECT_TRUE(nonzero);
    EXPECT_NEAR(global_flux, 0, 1e-15);
}

TEST(FluidModuleTest, IncrementDoesNotIntroduceNewSparseCoordinates)
{
    auto model = load_microfluid_case();
    const auto increment = mhs::sim::fluid::assemble_increment(model, default_state(model), 0.0);

    for (const auto& entry : increment.matrix_entries) {
        const mhs::core::Index row = static_cast<mhs::core::Index>(entry.row());
        const mhs::core::Index col = static_cast<mhs::core::Index>(entry.col());
        if (row == col)
            continue;

        const auto cell = model.dofs.cell(mhs::core::DofId {row});
        const auto neighbor = model.dofs.cell(mhs::core::DofId {col});
        bool direct_neighbor = false;
        for (auto id : model.discrete.incident_faces(cell)) {
            if (model.discrete.faces[id.value].other(cell) == neighbor) {
                direct_neighbor = true;
                break;
            }
        }
        EXPECT_TRUE(direct_neighbor) << "fluid increment added non-neighbor coordinate (" << row << ", " << col << ")";
    }
}

// Two disjoint inlet pieces and an outlet share a single control volume.
// Their enthalpy contributions must survive independent projection.
TEST(FluidModuleTest, PartialInletsAccumulateAndIgnoreThermalSubdivision)
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
