#include "fluid_preprocessor.hpp"

#include "data/tolerance_config.hpp"
#include "linear_solver/linear_solver.hpp"
#include "logger/logger.hpp"
#include "utils/mesh_utils.hpp"
#include "utils/physics_utils.hpp"

#include <Eigen/Sparse>
#include <cassert>
#include <cmath>
#include <limits>

namespace mhs::sim {
    namespace {

        mhs::Index fluid_grid_index(
            const mhs::core::Model& model, const FluidPreprocessWorkspace& workspace, mhs::Index fluid_index)
        {
            const mhs::Index active_index = model.fluid.fluid_to_global[fluid_index];
            assert(active_index < workspace.active_to_grid.size());
            return workspace.active_to_grid[active_index];
        }

        void initialize_hydraulic_conductance(mhs::core::Model& model, const FluidPreprocessWorkspace& workspace)
        {
            for (mhs::Index fluid_index = 0; fluid_index < model.fluid.n_fluid; ++fluid_index) {
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(
                    fluid_grid_index(model, workspace, fluid_index), model.mesh.ny, model.mesh.nz, ix, iy, iz);

                const double dx = model.mesh.dx[ix];
                const double dy = model.mesh.dy[iy];
                const double dz = model.mesh.dz[iz];
                const double permeability = std::pow(model.fluid.hydraulic_diameter[fluid_index], 2.0)
                    / (2.0
                        * mhs::utils::f_re_rectangular(
                            model.fluid.channel_width[fluid_index], model.fluid.channel_height[fluid_index]));
                const double coefficient = permeability / model.fluid.dynamic_viscosity[fluid_index];
                model.fluid.hydroC[0][fluid_index] = coefficient * (dy * dz / dx);
                model.fluid.hydroC[1][fluid_index] = coefficient * (dx * dz / dy);
                model.fluid.hydroC[2][fluid_index] = coefficient * (dx * dy / dz);
            }
        }

        double initial_density(
            const mhs::core::Model& model, const FluidPreprocessWorkspace& workspace, mhs::Index fluid_index)
        {
            const mhs::Index active_index = model.fluid.fluid_to_global[fluid_index];
            mhs::Index ix, iy, iz;
            mhs::utils::decode_index(
                fluid_grid_index(model, workspace, fluid_index), model.mesh.ny, model.mesh.nz, ix, iy, iz);
            const auto& material = model.material_table[model.cells.material_id[active_index]];
            return material.rho.eval(
                {model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz], model.initial_temperature, 0.0});
        }

        bool solve_pressure(mhs::core::Model& model, const FluidPreprocessWorkspace& workspace)
        {
            assert(model.fluid.n_fluid <= static_cast<mhs::Index>(std::numeric_limits<Eigen::Index>::max()));
            const auto count = static_cast<Eigen::Index>(model.fluid.n_fluid);
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(count);
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(static_cast<size_t>(model.fluid.n_fluid) * 7);

            for (mhs::Index fluid_index = 0; fluid_index < model.fluid.n_fluid; ++fluid_index) {
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(
                    fluid_grid_index(model, workspace, fluid_index), model.mesh.ny, model.mesh.nz, ix, iy, iz);

                double diagonal = 0.0;
                for (auto direction : mhs::core::FACE_DIRS) {
                    const mhs::Index neighbor_grid = mhs::utils::neighbor_grid_index(
                        ix, iy, iz, direction, model.mesh.nx, model.mesh.ny, model.mesh.nz, model.cells.index_map);
                    if (neighbor_grid == mhs::invalidIndex)
                        continue;
                    const mhs::Index neighbor_active = model.cells.index_map[neighbor_grid];
                    const mhs::Index neighbor_fluid = model.fluid.global_to_fluid[neighbor_active];
                    if (neighbor_fluid == mhs::invalidIndex)
                        continue;

                    const int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(direction)];
                    const auto& conductance = model.fluid.hydroC[axis];
                    const double effective
                        = mhs::utils::harmonicAverage(conductance[fluid_index], conductance[neighbor_fluid]);
                    diagonal += effective;
                    if (model.fluid.fluid_bcs[fluid_index].kind != mhs::core::FluidBCType::PressureType) {
                        triplets.emplace_back(static_cast<Eigen::Index>(fluid_index),
                            static_cast<Eigen::Index>(neighbor_fluid), -effective);
                    }
                }

                triplets.emplace_back(
                    static_cast<Eigen::Index>(fluid_index), static_cast<Eigen::Index>(fluid_index), diagonal);
                const auto& boundary = model.fluid.fluid_bcs[fluid_index];
                if (boundary.kind == mhs::core::FluidBCType::PressureType) {
                    rhs(static_cast<Eigen::Index>(fluid_index))
                        = model.fluid.fluid_bc_params.pressure[boundary.param_idx] * diagonal;
                }
                else if (boundary.kind == mhs::core::FluidBCType::MassFlowRateType) {
                    const double density = initial_density(model, workspace, fluid_index);
                    const double mass_flow = model.fluid.fluid_bc_params.mass_flow_rate[boundary.param_idx];
                    rhs(static_cast<Eigen::Index>(fluid_index))
                        = (density > mhs::core::zero_guard) ? mass_flow / density : 0.0;
                }
                else if (boundary.kind == mhs::core::FluidBCType::VelocityType) {
                    rhs(static_cast<Eigen::Index>(fluid_index))
                        = model.fluid.fluid_bc_params.velocity[boundary.param_idx]
                        * model.fluid.fluid_face_area[fluid_index];
                }
            }

            Eigen::SparseMatrix<double> system(count, count);
            system.setFromTriplets(triplets.begin(), triplets.end());
            auto solver = mhs::sim::LinearSolver::create();
            solver->compute(system);
            Eigen::VectorXd pressure = solver->solve(rhs);
            if (!solver->success()) {
                MHS_LOG_WARN("Fluid pressure solve failed (nf={}, nz={})", model.fluid.n_fluid,
                    static_cast<int>(system.nonZeros()));
                return false;
            }
            model.fluid.pressure.assign(pressure.data(), pressure.data() + pressure.size());
            return true;
        }

        void precompute_flow_axes(mhs::core::Model& model, const FluidPreprocessWorkspace& workspace)
        {
            for (mhs::Index fluid_index = 0; fluid_index < model.fluid.n_fluid; ++fluid_index) {
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(
                    fluid_grid_index(model, workspace, fluid_index), model.mesh.ny, model.mesh.nz, ix, iy, iz);

                double maximum_flux = -1.0;
                int best_axis = 0;
                for (auto direction : mhs::core::FACE_DIRS) {
                    const mhs::Index neighbor_grid = mhs::utils::neighbor_grid_index(
                        ix, iy, iz, direction, model.mesh.nx, model.mesh.ny, model.mesh.nz, model.cells.index_map);
                    if (neighbor_grid == mhs::invalidIndex)
                        continue;
                    const mhs::Index neighbor_active = model.cells.index_map[neighbor_grid];
                    const mhs::Index neighbor_fluid = model.fluid.global_to_fluid[neighbor_active];
                    if (neighbor_fluid == mhs::invalidIndex)
                        continue;

                    const int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(direction)];
                    const auto& conductance = model.fluid.hydroC[axis];
                    const double pressure_delta
                        = std::fabs(model.fluid.pressure[fluid_index] - model.fluid.pressure[neighbor_fluid]);
                    const double flux = pressure_delta
                        * mhs::utils::harmonicAverage(conductance[fluid_index], conductance[neighbor_fluid]);
                    if (flux > maximum_flux) {
                        maximum_flux = flux;
                        best_axis = axis;
                    }
                }
                model.fluid.flow_axes[fluid_index] = static_cast<int8_t>(best_axis);
            }
        }

    } // namespace

    void solveFluidFlow(mhs::core::Model& model, const FluidPreprocessWorkspace& workspace)
    {
        if (model.fluid.n_fluid == 0)
            return;
        initialize_hydraulic_conductance(model, workspace);
        if (solve_pressure(model, workspace))
            precompute_flow_axes(model, workspace);
    }

} // namespace mhs::sim
