#include "fluid/fluid_preprocessor.hpp"

#include "data/tolerance_config.hpp"
#include "linear_solver/linear_solver.hpp"
#include "logger/logger.hpp"
#include "utils/face_key.hpp"
#include "utils/mesh_utils.hpp"
#include "utils/physics_utils.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

namespace mhs::sim::fluid {

    namespace {
        struct FluidBCParamTable {
            std::vector<double> pressure;
            std::vector<double> mass_flow_rate;
            std::vector<double> velocity;
        };

        struct FluidCellBC {
            mhs::core::FluidBCType kind = mhs::core::FluidBCType::None;
            uint16_t param_idx = std::numeric_limits<uint16_t>::max();
        };

        struct FluidPreprocessWorkspace {
            std::vector<mhs::Index> compact_to_old;
            std::vector<double> viscosity;
            std::vector<double> pressure;
            std::array<std::vector<double>, 3> hydraulic_conductance;
            std::vector<double> hydraulic_diameter;
            std::vector<double> channel_width;
            std::vector<double> channel_height;
            std::vector<FluidCellBC> cell_bcs;
            FluidBCParamTable bc_params;
            std::vector<double> boundary_face_area;
        };

        mhs::Index fluid_count(const mhs::core::Model& model)
        { return static_cast<mhs::Index>(model.fluid.fluid_to_global.size()); }

        void build_compact_to_old(const mhs::core::Model& model, FluidPreprocessWorkspace& workspace)
        {
            workspace.compact_to_old.assign(model.cells.material_id.size(), mhs::invalidIndex);
            for (mhs::Index old = 0; old < model.cells.index_map.size(); ++old) {
                const mhs::Index compact = model.cells.index_map[old];
                if (compact != mhs::invalidIndex)
                    workspace.compact_to_old[compact] = old;
            }
        }

        bool is_fluid_cell(const mhs::core::Model& model, mhs::Index ix, mhs::Index iy, mhs::Index iz)
        {
            if (ix >= model.mesh.nx || iy >= model.mesh.ny || iz >= model.mesh.nz)
                return false;
            const mhs::Index old = ix * model.mesh.ny * model.mesh.nz + iy * model.mesh.nz + iz;
            const mhs::Index compact = model.cells.index_map[old];
            return compact != mhs::invalidIndex && model.fluid.global_to_fluid[compact] != mhs::invalidIndex;
        }

        double measure_fluid_extent(
            const mhs::core::Model& model, mhs::Index ix, mhs::Index iy, mhs::Index iz, int axis)
        {
            const mhs::Index sizes[3] = {model.mesh.nx, model.mesh.ny, model.mesh.nz};
            mhs::Index index[3] = {ix, iy, iz};
            mhs::Index min_index = index[axis];
            mhs::Index max_index = index[axis];

            while (min_index > 0) {
                index[axis] = min_index - 1;
                if (!is_fluid_cell(model, index[0], index[1], index[2]))
                    break;
                --min_index;
            }

            index[axis] = max_index;
            while (max_index < sizes[axis] - 1) {
                index[axis] = max_index + 1;
                if (!is_fluid_cell(model, index[0], index[1], index[2]))
                    break;
                ++max_index;
            }

            const auto& centers = axis == 0 ? model.mesh.cx : axis == 1 ? model.mesh.cy : model.mesh.cz;
            const auto& widths = axis == 0 ? model.mesh.dx : axis == 1 ? model.mesh.dy : model.mesh.dz;
            return (centers[max_index] + widths[max_index] * 0.5) - (centers[min_index] - widths[min_index] * 0.5);
        }

        void compute_channel_dimensions(mhs::core::Model& model, FluidPreprocessWorkspace& workspace)
        {
            for (mhs::Index fi = 0; fi < fluid_count(model); ++fi) {
                const mhs::Index old = workspace.compact_to_old[model.fluid.fluid_to_global[fi]];
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);

                double lengths[3] = {measure_fluid_extent(model, ix, iy, iz, 0),
                    measure_fluid_extent(model, ix, iy, iz, 1), measure_fluid_extent(model, ix, iy, iz, 2)};
                std::sort(lengths, lengths + 3);
                const double width = lengths[0];
                const double height = lengths[1];
                const double diameter
                    = width + height > mhs::core::geometry_eps ? 2.0 * width * height / (width + height) : 0.0;

                workspace.channel_width[fi] = width;
                workspace.channel_height[fi] = height;
                workspace.hydraulic_diameter[fi] = diameter;
                model.fluid.interface_heat_transfer_factor[fi] = diameter > mhs::core::geometry_eps
                    ? mhs::utils::nusselt_rectangular(width, height) / diameter
                    : 0.0;
            }
        }

        uint16_t register_bc_param(FluidBCParamTable& params, const mhs::core::FluidBoundary& boundary, double value)
        {
            switch (boundary.kind) {
            case mhs::core::FluidBCType::PressureType:
                params.pressure.push_back(value);
                return static_cast<uint16_t>(params.pressure.size() - 1);
            case mhs::core::FluidBCType::MassFlowRateType:
                params.mass_flow_rate.push_back(value);
                return static_cast<uint16_t>(params.mass_flow_rate.size() - 1);
            case mhs::core::FluidBCType::VelocityType:
                params.velocity.push_back(value);
                return static_cast<uint16_t>(params.velocity.size() - 1);
            case mhs::core::FluidBCType::None:
            default:
                return std::numeric_limits<uint16_t>::max();
            }
        }

        double face_area(const mhs::utils::FaceKeyInfo& face_key, const mhs::core::MeshGeometry& mesh, mhs::Index ix,
            mhs::Index iy, mhs::Index iz)
        {
            const int axis = face_key.axis == 'X' ? 0 : face_key.axis == 'Y' ? 1 : 2;
            const double a = axis == 0 ? mesh.dy[iy] : mesh.dx[ix];
            const double b = axis == 2 ? mesh.dy[iy] : mesh.dz[iz];
            return a * b;
        }

        void apply_boundaries(mhs::core::Model& model, FluidPreprocessWorkspace& workspace,
            const std::vector<mhs::core::FluidBoundary>& boundaries, double si_scale)
        {
            for (const auto& boundary : boundaries) {
                for (const auto& key : boundary.face_keys) {
                    const auto face_key = mhs::utils::parse_face_key(key, si_scale);
                    const int axis = face_key.axis == 'X' ? 0 : face_key.axis == 'Y' ? 1 : 2;

                    std::vector<mhs::Index> matched;
                    matched.reserve(64);
                    for (mhs::Index fi = 0; fi < fluid_count(model); ++fi) {
                        const mhs::Index old = workspace.compact_to_old[model.fluid.fluid_to_global[fi]];
                        mhs::Index ix, iy, iz;
                        mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                        const double centers[3] = {model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz]};
                        const double widths[3] = {model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]};
                        const double face_minus = centers[axis] - widths[axis] * 0.5;
                        const double face_plus = centers[axis] + widths[axis] * 0.5;
                        if (std::abs(face_minus - face_key.coord_value) >= mhs::core::geometry_eps
                            && std::abs(face_plus - face_key.coord_value) >= mhs::core::geometry_eps) {
                            continue;
                        }

                        const double a = centers[(axis + 1) % 3];
                        const double b = centers[(axis + 2) % 3];
                        if (mhs::utils::point_in_face_rects(face_key, a, b)
                            || mhs::utils::point_in_face_rects(face_key, b, a)) {
                            matched.push_back(fi);
                        }
                    }

                    if (matched.empty())
                        continue;

                    double value = boundary.value;
                    if (boundary.kind == mhs::core::FluidBCType::MassFlowRateType)
                        value /= static_cast<double>(matched.size());
                    const uint16_t param_index = register_bc_param(workspace.bc_params, boundary, value);

                    for (mhs::Index fi : matched) {
                        workspace.cell_bcs[fi] = {boundary.kind, param_index};
                        if (!std::isnan(boundary.inlet_temperature))
                            model.fluid.boundary_temperature[fi] = boundary.inlet_temperature;

                        if (boundary.kind == mhs::core::FluidBCType::MassFlowRateType) {
                            model.fluid.boundary_outflux[fi] = value;
                        }
                        else if (boundary.kind == mhs::core::FluidBCType::VelocityType) {
                            const mhs::Index old = workspace.compact_to_old[model.fluid.fluid_to_global[fi]];
                            mhs::Index ix, iy, iz;
                            mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                            workspace.boundary_face_area[fi] = face_area(face_key, model.mesh, ix, iy, iz);
                            model.fluid.boundary_outflux[fi] = value * workspace.boundary_face_area[fi];
                        }
                    }
                }
            }
        }

        double evaluate_rho_at_initial_temperature(
            const mhs::core::Model& model, const FluidPreprocessWorkspace& workspace, mhs::Index fi)
        {
            const mhs::Index cell = model.fluid.fluid_to_global[fi];
            const mhs::Index old = workspace.compact_to_old[cell];
            mhs::Index ix, iy, iz;
            mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
            const auto& material = model.material_table[model.cells.material_id[cell]];
            return material.rho.eval(
                {model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz], model.initial_temperature, 0.0});
        }

        void initialize_hydraulic_conductance(const mhs::core::Model& model, FluidPreprocessWorkspace& workspace)
        {
            for (mhs::Index fi = 0; fi < fluid_count(model); ++fi) {
                const mhs::Index old = workspace.compact_to_old[model.fluid.fluid_to_global[fi]];
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                const double dx = model.mesh.dx[ix];
                const double dy = model.mesh.dy[iy];
                const double dz = model.mesh.dz[iz];
                const double permeability = std::pow(workspace.hydraulic_diameter[fi], 2.0)
                    / (2.0 * mhs::utils::f_re_rectangular(workspace.channel_width[fi], workspace.channel_height[fi]));
                const double coefficient = permeability / workspace.viscosity[fi];
                workspace.hydraulic_conductance[0][fi] = coefficient * dy * dz / dx;
                workspace.hydraulic_conductance[1][fi] = coefficient * dx * dz / dy;
                workspace.hydraulic_conductance[2][fi] = coefficient * dx * dy / dz;
            }
        }

        bool solve_pressure(const mhs::core::Model& model, FluidPreprocessWorkspace& workspace)
        {
            const mhs::Index count = fluid_count(model);
            assert(count <= static_cast<mhs::Index>(std::numeric_limits<Eigen::Index>::max()));
            const auto eigen_count = static_cast<Eigen::Index>(count);
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(static_cast<std::size_t>(count) * 7);
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(eigen_count);

            for (mhs::Index fi = 0; fi < count; ++fi) {
                const mhs::Index old = workspace.compact_to_old[model.fluid.fluid_to_global[fi]];
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                double diagonal = 0.0;

                for (std::size_t face = 0; face < mhs::core::FACE_COUNT; ++face) {
                    const auto dir = mhs::core::FACE_DIRS[face];
                    const mhs::Index neighbor_old = mhs::utils::neighbor_grid_index(
                        ix, iy, iz, dir, model.mesh.nx, model.mesh.ny, model.mesh.nz, model.cells.index_map);
                    if (neighbor_old == mhs::invalidIndex)
                        continue;
                    const mhs::Index neighbor = model.cells.index_map[neighbor_old];
                    const mhs::Index fn = model.fluid.global_to_fluid[neighbor];
                    if (fn == mhs::invalidIndex)
                        continue;

                    const int axis = mhs::utils::AXIS_OF_DIR[face];
                    const auto& conductance = workspace.hydraulic_conductance[axis];
                    const double effective = mhs::utils::harmonicAverage(conductance[fi], conductance[fn]);
                    diagonal += effective;
                    if (workspace.cell_bcs[fi].kind != mhs::core::FluidBCType::PressureType) {
                        triplets.emplace_back(static_cast<Eigen::Index>(fi), static_cast<Eigen::Index>(fn), -effective);
                    }
                }

                triplets.emplace_back(static_cast<Eigen::Index>(fi), static_cast<Eigen::Index>(fi), diagonal);
                const auto& bc = workspace.cell_bcs[fi];
                if (bc.kind == mhs::core::FluidBCType::PressureType) {
                    rhs(static_cast<Eigen::Index>(fi)) = workspace.bc_params.pressure[bc.param_idx] * diagonal;
                }
                else if (bc.kind == mhs::core::FluidBCType::MassFlowRateType) {
                    const double density = evaluate_rho_at_initial_temperature(model, workspace, fi);
                    const double mass_flow = workspace.bc_params.mass_flow_rate[bc.param_idx];
                    rhs(static_cast<Eigen::Index>(fi)) = density > mhs::core::zero_guard ? mass_flow / density : 0.0;
                }
                else if (bc.kind == mhs::core::FluidBCType::VelocityType) {
                    rhs(static_cast<Eigen::Index>(fi))
                        = workspace.bc_params.velocity[bc.param_idx] * workspace.boundary_face_area[fi];
                }
            }

            Eigen::SparseMatrix<double> matrix(eigen_count, eigen_count);
            matrix.setFromTriplets(triplets.begin(), triplets.end());
            auto solver = mhs::sim::LinearSolver::create();
            solver->compute(matrix);
            Eigen::VectorXd pressure = solver->solve(rhs);
            if (!solver->success()) {
                MHS_LOG_WARN("Fluid pressure solve failed (nf={}, nz={})", count, static_cast<int>(matrix.nonZeros()));
                return false;
            }
            workspace.pressure.assign(pressure.data(), pressure.data() + pressure.size());
            return true;
        }

        void compute_face_volume_flux(mhs::core::Model& model, FluidPreprocessWorkspace& workspace)
        {
            for (mhs::Index fi = 0; fi < fluid_count(model); ++fi) {
                const mhs::Index old = workspace.compact_to_old[model.fluid.fluid_to_global[fi]];
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                for (std::size_t face = 0; face < mhs::core::FACE_COUNT; ++face) {
                    const auto dir = mhs::core::FACE_DIRS[face];
                    const mhs::Index neighbor_old = mhs::utils::neighbor_grid_index(
                        ix, iy, iz, dir, model.mesh.nx, model.mesh.ny, model.mesh.nz, model.cells.index_map);
                    if (neighbor_old == mhs::invalidIndex)
                        continue;
                    const mhs::Index neighbor = model.cells.index_map[neighbor_old];
                    const mhs::Index fn = model.fluid.global_to_fluid[neighbor];
                    if (fn == mhs::invalidIndex)
                        continue;

                    const int axis = mhs::utils::AXIS_OF_DIR[face];
                    const auto& conductance = workspace.hydraulic_conductance[axis];
                    if (conductance[fi] <= mhs::core::zero_guard || conductance[fn] <= mhs::core::zero_guard)
                        continue;
                    const double effective = mhs::utils::harmonicAverage(conductance[fi], conductance[fn]);
                    model.fluid.face_volume_flux[fi * mhs::core::FACE_COUNT + face]
                        = (workspace.pressure[fi] - workspace.pressure[fn]) * effective;
                }
            }
        }

    } // namespace

    void build_domain(mhs::core::Model& model, const std::vector<mhs::core::FluidBoundary>& boundaries, double si_scale,
        const FluidMaterialData& materials)
    {
        model.fluid = {};
        const mhs::Index active_count = static_cast<mhs::Index>(model.cells.material_id.size());
        model.fluid.global_to_fluid.assign(active_count, mhs::invalidIndex);

        FluidPreprocessWorkspace workspace;
        for (mhs::Index cell = 0; cell < active_count; ++cell) {
            const auto material = static_cast<std::size_t>(model.cells.material_id[cell]);
            if (material >= materials.is_fluid.size() || !materials.is_fluid[material])
                continue;
            model.fluid.global_to_fluid[cell] = static_cast<mhs::Index>(model.fluid.fluid_to_global.size());
            model.fluid.fluid_to_global.push_back(cell);
            workspace.viscosity.push_back(materials.initial_viscosity[material]);
        }

        const mhs::Index count = fluid_count(model);
        if (count == 0) {
            model.fluid.global_to_fluid.clear();
            return;
        }

        build_compact_to_old(model, workspace);
        workspace.pressure.assign(count, 0.0);
        for (auto& conductance : workspace.hydraulic_conductance)
            conductance.assign(count, 0.0);
        workspace.hydraulic_diameter.assign(count, 0.0);
        workspace.channel_width.assign(count, 0.0);
        workspace.channel_height.assign(count, 0.0);
        workspace.cell_bcs.assign(count, {});
        workspace.boundary_face_area.assign(count, 0.0);

        model.fluid.face_volume_flux.assign(count * mhs::core::FACE_COUNT, 0.0);
        model.fluid.interface_heat_transfer_factor.assign(count, 0.0);
        model.fluid.boundary_outflux.assign(count, std::numeric_limits<double>::quiet_NaN());
        model.fluid.boundary_temperature.assign(count, std::numeric_limits<double>::quiet_NaN());

        apply_boundaries(model, workspace, boundaries, si_scale);
        compute_channel_dimensions(model, workspace);
        initialize_hydraulic_conductance(model, workspace);
        if (solve_pressure(model, workspace))
            compute_face_volume_flux(model, workspace);
    }

} // namespace mhs::sim::fluid
