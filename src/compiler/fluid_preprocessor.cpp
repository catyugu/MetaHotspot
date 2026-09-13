#include "compiler/fluid_preprocessor.hpp"

#include "compiler/fluid_physics.hpp"
#include "core/constants.hpp"
#include "core/mesh.hpp"
#include "numerics/linear/linear_solver.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

namespace mhs::sim::fluid {

    namespace {
        struct FluidCellBC {
            mhs::model::FluidBoundaryKind kind = mhs::model::FluidBoundaryKind::None;
            double value = 0.0;
        };

        struct FluidPreprocessWorkspace {
            std::vector<double> viscosity;
            std::vector<double> pressure;
            std::array<std::vector<double>, 3> hydraulic_conductance;
            std::vector<double> hydraulic_diameter;
            std::vector<double> channel_width;
            std::vector<double> channel_height;
            std::vector<FluidCellBC> cell_bcs;
            std::vector<double> boundary_face_area;
        };

        mhs::core::Index fluid_count(const mhs::core::Model& model)
        { return static_cast<mhs::core::Index>(model.fluid.fluid_to_global.size()); }

        bool is_fluid_cell(const mhs::core::Model& model, mhs::core::Index ix, mhs::core::Index iy, mhs::core::Index iz)
        {
            if (ix >= model.mesh.nx || iy >= model.mesh.ny || iz >= model.mesh.nz)
                return false;
            const mhs::core::Index old = ix * model.mesh.ny * model.mesh.nz + iy * model.mesh.nz + iz;
            const mhs::core::Index compact = model.cells.grid_to_cell[old];
            return compact != mhs::core::invalidIndex
                && model.fluid.global_to_fluid[compact] != mhs::core::invalidIndex;
        }

        double measure_fluid_extent(
            const mhs::core::Model& model, mhs::core::Index ix, mhs::core::Index iy, mhs::core::Index iz, int axis)
        {
            const mhs::core::Index sizes[3] = {model.mesh.nx, model.mesh.ny, model.mesh.nz};
            mhs::core::Index index[3] = {ix, iy, iz};
            mhs::core::Index min_index = index[axis];
            mhs::core::Index max_index = index[axis];

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
            for (mhs::core::Index fi = 0; fi < fluid_count(model); ++fi) {
                const mhs::core::Index old = model.cells.cell_to_grid[model.fluid.fluid_to_global[fi]];
                mhs::core::Index ix, iy, iz;
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

        int axis_index(mhs::model::Axis axis)
        { return axis == mhs::model::Axis::X ? 0 : axis == mhs::model::Axis::Y ? 1 : 2; }

        bool point_in_region(const mhs::model::FaceRegion& region, double a, double b, double si_scale)
        {
            return std::any_of(region.rectangles.begin(), region.rectangles.end(), [&](const auto& rectangle) {
                return a >= rectangle.a_min * si_scale - mhs::core::geometry_eps
                    && a <= rectangle.a_max * si_scale + mhs::core::geometry_eps
                    && b >= rectangle.b_min * si_scale - mhs::core::geometry_eps
                    && b <= rectangle.b_max * si_scale + mhs::core::geometry_eps;
            });
        }

        double face_area(const mhs::model::FaceRegion& region, const mhs::core::MeshGeometry& mesh, mhs::core::Index ix,
            mhs::core::Index iy, mhs::core::Index iz)
        {
            const int axis = axis_index(region.axis);
            const double a = axis == 0 ? mesh.dy[iy] : mesh.dx[ix];
            const double b = axis == 2 ? mesh.dy[iy] : mesh.dz[iz];
            return a * b;
        }

        void apply_boundaries(mhs::core::Model& model, FluidPreprocessWorkspace& workspace,
            const std::vector<mhs::model::FluidBoundarySpec>& boundaries, double si_scale)
        {
            for (const auto& boundary : boundaries) {
                for (const auto& region : boundary.regions) {
                    const int axis = axis_index(region.axis);
                    const double coordinate = region.coordinate * si_scale;

                    std::vector<mhs::core::Index> matched;
                    matched.reserve(64);
                    for (mhs::core::Index fi = 0; fi < fluid_count(model); ++fi) {
                        const mhs::core::Index old = model.cells.cell_to_grid[model.fluid.fluid_to_global[fi]];
                        mhs::core::Index ix, iy, iz;
                        mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                        const double centers[3] = {model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz]};
                        const double widths[3] = {model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]};
                        const double face_minus = centers[axis] - widths[axis] * 0.5;
                        const double face_plus = centers[axis] + widths[axis] * 0.5;
                        if (std::abs(face_minus - coordinate) >= mhs::core::geometry_eps
                            && std::abs(face_plus - coordinate) >= mhs::core::geometry_eps) {
                            continue;
                        }

                        const double a = axis == 0 ? centers[1] : centers[0];
                        const double b = axis == 2 ? centers[1] : centers[2];
                        if (point_in_region(region, a, b, si_scale)) {
                            matched.push_back(fi);
                        }
                    }

                    if (matched.empty())
                        continue;

                    double value = boundary.value;
                    if (boundary.kind == mhs::model::FluidBoundaryKind::MassFlowRate)
                        value /= static_cast<double>(matched.size());
                    for (mhs::core::Index fi : matched) {
                        workspace.cell_bcs[fi] = {boundary.kind, value};
                        if (!std::isnan(boundary.inlet_temperature))
                            model.fluid.boundary_temperature[fi] = boundary.inlet_temperature;

                        if (boundary.kind == mhs::model::FluidBoundaryKind::MassFlowRate) {
                            model.fluid.boundary_outflux[fi] = value;
                        }
                        else if (boundary.kind == mhs::model::FluidBoundaryKind::Velocity) {
                            const mhs::core::Index old = model.cells.cell_to_grid[model.fluid.fluid_to_global[fi]];
                            mhs::core::Index ix, iy, iz;
                            mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                            workspace.boundary_face_area[fi] = face_area(region, model.mesh, ix, iy, iz);
                            model.fluid.boundary_outflux[fi] = value * workspace.boundary_face_area[fi];
                        }
                    }
                }
            }
        }

        double evaluate_rho_at_initial_temperature(const mhs::core::Model& model, mhs::core::Index fi)
        {
            const mhs::core::Index cell = model.fluid.fluid_to_global[fi];
            const mhs::core::Index old = model.cells.cell_to_grid[cell];
            mhs::core::Index ix, iy, iz;
            mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
            const auto& material = model.material_table[model.cells.material_id[cell]];
            return material.rho.eval(
                {model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz], model.initial_temperature, 0.0});
        }

        void initialize_hydraulic_conductance(const mhs::core::Model& model, FluidPreprocessWorkspace& workspace)
        {
            for (mhs::core::Index fi = 0; fi < fluid_count(model); ++fi) {
                const mhs::core::Index old = model.cells.cell_to_grid[model.fluid.fluid_to_global[fi]];
                mhs::core::Index ix, iy, iz;
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
            const mhs::core::Index count = fluid_count(model);
            assert(count <= static_cast<mhs::core::Index>(std::numeric_limits<Eigen::Index>::max()));
            const auto eigen_count = static_cast<Eigen::Index>(count);
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(static_cast<std::size_t>(count) * 7);
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(eigen_count);

            for (mhs::core::Index fi = 0; fi < count; ++fi) {
                const mhs::core::Index old = model.cells.cell_to_grid[model.fluid.fluid_to_global[fi]];
                mhs::core::Index ix, iy, iz;
                mhs::utils::decode_index(old, model.mesh.ny, model.mesh.nz, ix, iy, iz);
                double diagonal = 0.0;

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

                    const int axis = mhs::utils::AXIS_OF_DIR[face];
                    const auto& conductance = workspace.hydraulic_conductance[axis];
                    const double effective = mhs::utils::harmonicAverage(conductance[fi], conductance[fn]);
                    diagonal += effective;
                    if (workspace.cell_bcs[fi].kind != mhs::model::FluidBoundaryKind::Pressure) {
                        triplets.emplace_back(static_cast<int>(fi), static_cast<int>(fn), -effective);
                    }
                }

                triplets.emplace_back(static_cast<int>(fi), static_cast<int>(fi), diagonal);
                const auto& bc = workspace.cell_bcs[fi];
                if (bc.kind == mhs::model::FluidBoundaryKind::Pressure) {
                    rhs(static_cast<Eigen::Index>(fi)) = bc.value * diagonal;
                }
                else if (bc.kind == mhs::model::FluidBoundaryKind::MassFlowRate) {
                    const double density = evaluate_rho_at_initial_temperature(model, fi);
                    rhs(static_cast<Eigen::Index>(fi)) = density > mhs::core::zero_guard ? bc.value / density : 0.0;
                }
                else if (bc.kind == mhs::model::FluidBoundaryKind::Velocity) {
                    rhs(static_cast<Eigen::Index>(fi)) = bc.value * workspace.boundary_face_area[fi];
                }
            }

            Eigen::SparseMatrix<double> matrix(eigen_count, eigen_count);
            matrix.setFromTriplets(triplets.begin(), triplets.end());
            auto solver = mhs::sim::create_solver();
            solver->compute(matrix);
            // Cold-start from zero via the x0 overload so this works with the
            // default iterative (AMGCL) backend as well as a direct one.
            Eigen::VectorXd pressure = solver->solve(rhs, Eigen::VectorXd::Zero(eigen_count));
            if (!solver->success()) {
                throw std::runtime_error("fluid pressure solve failed");
            }
            workspace.pressure.assign(pressure.data(), pressure.data() + pressure.size());
            return true;
        }

        void compute_face_volume_flux(mhs::core::Model& model, FluidPreprocessWorkspace& workspace)
        {
            for (mhs::core::Index fi = 0; fi < fluid_count(model); ++fi) {
                const mhs::core::Index old = model.cells.cell_to_grid[model.fluid.fluid_to_global[fi]];
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

                    const int axis = mhs::utils::AXIS_OF_DIR[face];
                    const auto& conductance = workspace.hydraulic_conductance[axis];
                    const double effective = mhs::utils::harmonicAverage(conductance[fi], conductance[fn]);
                    model.fluid.face_volume_flux[fi * mhs::core::FACE_COUNT + face]
                        = (workspace.pressure[fi] - workspace.pressure[fn]) * effective;
                }
            }
        }

    } // namespace

    void build_domain(mhs::core::Model& model, const std::vector<mhs::model::FluidBoundarySpec>& boundaries,
        double si_scale, const FluidMaterialData& materials)
    {
        model.fluid = {};
        const mhs::core::Index active_count = static_cast<mhs::core::Index>(model.cells.material_id.size());
        model.fluid.global_to_fluid.assign(active_count, mhs::core::invalidIndex);

        FluidPreprocessWorkspace workspace;
        for (mhs::core::Index cell = 0; cell < active_count; ++cell) {
            const auto material = static_cast<std::size_t>(model.cells.material_id[cell]);
            if (material >= materials.initial_viscosity.size() || !materials.initial_viscosity[material])
                continue;
            model.fluid.global_to_fluid[cell] = static_cast<mhs::core::Index>(model.fluid.fluid_to_global.size());
            model.fluid.fluid_to_global.push_back(cell);
            workspace.viscosity.push_back(*materials.initial_viscosity[material]);
        }

        const mhs::core::Index count = fluid_count(model);
        if (count == 0) {
            model.fluid.global_to_fluid.clear();
            return;
        }

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
        solve_pressure(model, workspace);
        compute_face_volume_flux(model, workspace);
    }

} // namespace mhs::sim::fluid
