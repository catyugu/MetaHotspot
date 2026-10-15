#include "compiler/model_compiler.hpp"
#include "config.h"
#include "core/mesh.hpp"
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
    {
        return std::vector<double>(
            static_cast<std::size_t>(model.cells.cell_to_grid.size()), model.initial_temperature);
    }

} // namespace

TEST(FluidModuleTest, PreprocessorStoresOnlyAssemblyReadyFields)
{
    auto model = load_microfluid_case();
    const auto& fluid = model.fluid;

    ASSERT_EQ(fluid.fluid_to_global.size(), 3200u);
    EXPECT_EQ(fluid.global_to_fluid.size(), model.cells.material_id.size());
    EXPECT_EQ(fluid.face_volume_flux.size(), model.discrete.faces.size());
    EXPECT_EQ(fluid.interface_heat_transfer_factor.size(), fluid.fluid_to_global.size());
    EXPECT_EQ(fluid.boundary_outflux.size(), fluid.fluid_to_global.size());
    EXPECT_EQ(fluid.boundary_temperature.size(), fluid.fluid_to_global.size());

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
    EXPECT_TRUE(model.fluid.boundary_outflux.empty());
    EXPECT_TRUE(model.fluid.boundary_temperature.empty());
}

TEST(FluidModuleTest, FrozenFaceFluxIsAntisymmetric)
{
    auto model = load_microfluid_case();

    ASSERT_EQ(model.fluid.face_volume_flux.size(), model.discrete.faces.size());
    double global_flux=0;
    bool nonzero=false;
    for (mhs::core::Index fi=0;fi<model.fluid.fluid_to_global.size();++fi) {
        const mhs::core::CellId cell{model.fluid.fluid_to_global[fi]};
        double net=0;
        bool exterior=false;
        for(auto id:model.discrete.incident_faces(cell)) {
            const auto& f=model.discrete.faces[id.value];
            if(!f.neighbor.valid()) exterior=true;
            const double q=model.fluid.face_volume_flux[id.value]*(f.owner==cell?1:-1);
            nonzero=nonzero || std::abs(q)>1e-20;
            net+=q;
        }
        global_flux+=net;
        if(!exterior) EXPECT_NEAR(net,0,1e-12);
    }
    EXPECT_TRUE(nonzero);
    EXPECT_NEAR(global_flux,0,1e-15);
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

        const mhs::core::Index old = model.cells.cell_to_grid[row];
        mhs::core::Index ix, iy, iz;
        mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);

        bool direct_neighbor = false;
        for (auto dir : mhs::core::FACE_DIRS) {
            const mhs::core::Index neighbor_old = mhs::utils::neighbor_grid_index(
                ix, iy, iz, dir, model.mesh.nx, model.mesh.ny, model.mesh.nz, model.cells.grid_to_cell);
            if (neighbor_old != mhs::core::invalidIndex && model.cells.grid_to_cell[neighbor_old] == col) {
                direct_neighbor = true;
                break;
            }
        }
        EXPECT_TRUE(direct_neighbor) << "fluid increment added non-neighbor coordinate (" << row << ", " << col << ")";
    }
}
