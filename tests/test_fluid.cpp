#include "compiler/model_compiler.hpp"
#include "config.h"
#include "io/model_io.hpp"
#include "runtime/mesh.hpp"
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
        const std::string overlay = root + "/cases/microfluid_cases/steady_case1_additional.xml";
        EXPECT_TRUE(std::filesystem::exists(input));
        EXPECT_TRUE(std::filesystem::exists(overlay));

        auto definition = mhs::io::read_xml(input);
        EXPECT_TRUE(mhs::io::merge_fluid_xml(overlay, definition));
        return mhs::sim::build_model(definition);
    }

} // namespace

TEST(FluidModuleTest, PreprocessorStoresOnlyAssemblyReadyFields)
{
    auto model = load_microfluid_case();
    const auto& fluid = model.fluid;

    ASSERT_EQ(fluid.fluid_to_global.size(), 3200u);
    EXPECT_EQ(fluid.global_to_fluid.size(), model.cells.material_id.size());
    EXPECT_EQ(fluid.face_volume_flux.size(), fluid.fluid_to_global.size() * mhs::core::FACE_COUNT);
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

    for (mhs::core::Index fi = 0; fi < model.fluid.fluid_to_global.size(); ++fi) {
        const mhs::core::Index cell = model.fluid.fluid_to_global[fi];
        const mhs::core::Index old = model.cells.cell_to_grid[cell];
        mhs::core::Index ix, iy, iz;
        mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);

        for (std::size_t face = 0; face < mhs::core::FACE_COUNT; ++face) {
            const auto dir = mhs::core::FACE_DIRS[face];
            const mhs::core::Index neighbor_old = mhs::utils::neighbor_grid_index(
                ix, iy, iz, dir, model.mesh.nx, model.mesh.ny, model.mesh.nz, model.cells.grid_to_cell);
            if (neighbor_old == mhs::core::invalidIndex)
                continue;
            const mhs::core::Index neighbor = model.cells.grid_to_cell[neighbor_old];
            const mhs::core::Index fn = model.fluid.global_to_fluid[neighbor];
            if (fn == mhs::core::invalidIndex)
                continue;

            const std::size_t opposite = face ^ 1U;
            EXPECT_NEAR(model.fluid.face_volume_flux[fi * mhs::core::FACE_COUNT + face],
                -model.fluid.face_volume_flux[fn * mhs::core::FACE_COUNT + opposite], 1e-18);
        }
    }
}

TEST(FluidModuleTest, ContributionDoesNotIntroduceNewSparseCoordinates)
{
    auto model = load_microfluid_case();
    mhs::sim::AssembleContext context {model.initial_state, 0.0};
    const auto contribution = mhs::sim::fluid::assemble_operator(model, context);
    const auto state_offset = model.dofs.cell_states.begin;

    for (const auto& entry : contribution.stiffness) {
        const mhs::core::Index row = static_cast<mhs::core::Index>(entry.row()) - state_offset;
        const mhs::core::Index col = static_cast<mhs::core::Index>(entry.col()) - state_offset;
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
        EXPECT_TRUE(direct_neighbor)
            << "fluid contribution added non-neighbor coordinate (" << row << ", " << col << ")";
    }
}
