#include "fluid_preprocessor.hpp"

#include "data/tolerance_config.hpp"
#include "face_key_processor.hpp"
#include "function_helpers.hpp"
#include "utils/mesh_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace mhs::sim {
    namespace {

        std::vector<mhs::Index> build_active_to_grid(const mhs::core::CellFields& cells)
        {
            std::vector<mhs::Index> result(cells.material_id.size(), mhs::invalidIndex);
            for (mhs::Index grid_index = 0; grid_index < cells.index_map.size(); ++grid_index) {
                const mhs::Index active_index = cells.index_map[grid_index];
                if (active_index != mhs::invalidIndex)
                    result[active_index] = grid_index;
            }
            return result;
        }

        double measure_fluid_extent(
            const mhs::core::Model& model, mhs::Index ix, mhs::Index iy, mhs::Index iz, int axis)
        {
            const auto& mesh = model.mesh;
            const auto& cells = model.cells;
            auto is_fluid_cell = [&](mhs::Index cx, mhs::Index cy, mhs::Index cz) {
                if (cx >= mesh.nx || cy >= mesh.ny || cz >= mesh.nz)
                    return false;
                const mhs::Index grid_index = cx * mesh.ny * mesh.nz + cy * mesh.nz + cz;
                const mhs::Index active_index = cells.index_map[grid_index];
                return active_index != mhs::invalidIndex && active_index < model.fluid.is_fluid.size()
                    && model.fluid.is_fluid[active_index];
            };

            const mhs::Index sizes[3] = {mesh.nx, mesh.ny, mesh.nz};
            mhs::Index index[3] = {ix, iy, iz};
            mhs::Index minimum = index[axis];
            mhs::Index maximum = index[axis];

            while (minimum > 0) {
                index[axis] = minimum - 1;
                if (!is_fluid_cell(index[0], index[1], index[2]))
                    break;
                --minimum;
            }
            index[axis] = maximum;
            while (maximum < sizes[axis] - 1) {
                index[axis] = maximum + 1;
                if (!is_fluid_cell(index[0], index[1], index[2]))
                    break;
                ++maximum;
            }

            const auto& centers = (axis == 0) ? mesh.cx : (axis == 1) ? mesh.cy : mesh.cz;
            const auto& widths = (axis == 0) ? mesh.dx : (axis == 1) ? mesh.dy : mesh.dz;
            return (centers[maximum] + widths[maximum] * 0.5) - (centers[minimum] - widths[minimum] * 0.5);
        }

        void compute_channel_dimensions(mhs::core::Model& model, const std::vector<mhs::Index>& active_to_grid)
        {
            for (mhs::Index fluid_index = 0; fluid_index < model.fluid.n_fluid; ++fluid_index) {
                const mhs::Index active_index = model.fluid.fluid_to_global[fluid_index];
                const mhs::Index grid_index = active_to_grid[active_index];
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(grid_index, model.mesh.ny, model.mesh.nz, ix, iy, iz);

                double lengths[3] = {measure_fluid_extent(model, ix, iy, iz, 0),
                    measure_fluid_extent(model, ix, iy, iz, 1), measure_fluid_extent(model, ix, iy, iz, 2)};
                std::sort(lengths, lengths + 3);
                const double width = lengths[0];
                const double height = lengths[1];
                const double diameter
                    = (width + height > mhs::core::geometry_eps) ? (2.0 * width * height / (width + height)) : 0.0;

                model.fluid.hydraulic_diameter[fluid_index] = diameter;
                model.fluid.channel_width[fluid_index] = width;
                model.fluid.channel_height[fluid_index] = height;
            }
        }

        uint16_t register_fluid_bc_parameter(mhs::core::FluidBCParamTable& parameters,
            const mhs::core::FluidBoundaryOverlay& boundary, double per_cell_value)
        {
            switch (boundary.kind) {
            case mhs::core::FluidBCType::PressureType:
                parameters.pressure.push_back(per_cell_value);
                return static_cast<uint16_t>(parameters.pressure.size() - 1);
            case mhs::core::FluidBCType::MassFlowRateType:
                parameters.mass_flow_rate.push_back(per_cell_value);
                return static_cast<uint16_t>(parameters.mass_flow_rate.size() - 1);
            case mhs::core::FluidBCType::VelocityType:
                parameters.velocity.push_back(per_cell_value);
                return static_cast<uint16_t>(parameters.velocity.size() - 1);
            case mhs::core::FluidBCType::None:
            default:
                return std::numeric_limits<uint16_t>::max();
            }
        }

        double face_area(
            const FaceKeyInfo& face, const mhs::core::MeshGeometry& mesh, mhs::Index ix, mhs::Index iy, mhs::Index iz)
        {
            const int axis = (face.axis == 'X') ? 0 : (face.axis == 'Y') ? 1 : 2;
            const double a = (axis == 0) ? mesh.dy[iy] : mesh.dx[ix];
            const double b = (axis == 2) ? mesh.dy[iy] : mesh.dz[iz];
            return a * b;
        }

        void apply_fluid_boundaries(mhs::core::Model& model, const mhs::core::FluidOverlay& overlay, double si_scale,
            const std::vector<mhs::Index>& active_to_grid)
        {
            const auto& mesh = model.mesh;
            for (const auto& boundary : overlay.boundaries) {
                for (const auto& face_key : boundary.face_keys) {
                    const FaceKeyInfo face = parse_face_key(face_key, si_scale);
                    const int target_axis = (face.axis == 'X') ? 0 : (face.axis == 'Y') ? 1 : 2;

                    std::vector<mhs::Index> matched;
                    for (mhs::Index fluid_index = 0; fluid_index < model.fluid.n_fluid; ++fluid_index) {
                        const mhs::Index active_index = model.fluid.fluid_to_global[fluid_index];
                        const mhs::Index grid_index = active_to_grid[active_index];
                        mhs::Index ix, iy, iz;
                        mhs::utils::decode_index(grid_index, mesh.ny, mesh.nz, ix, iy, iz);

                        const double center[3] = {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz]};
                        const double width[3] = {mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]};
                        const double negative_face = center[target_axis] - width[target_axis] * 0.5;
                        const double positive_face = center[target_axis] + width[target_axis] * 0.5;
                        if (std::abs(negative_face - face.coord_value) >= mhs::core::geometry_eps
                            && std::abs(positive_face - face.coord_value) >= mhs::core::geometry_eps)
                            continue;

                        const double a = center[(target_axis + 1) % 3];
                        const double b = center[(target_axis + 2) % 3];
                        if (point_in_face_rects(face, a, b) || point_in_face_rects(face, b, a))
                            matched.push_back(fluid_index);
                    }

                    if (matched.empty())
                        continue;
                    double per_cell_value = boundary.value;
                    if (boundary.kind == mhs::core::FluidBCType::MassFlowRateType)
                        per_cell_value /= static_cast<double>(matched.size());
                    const uint16_t parameter_index
                        = register_fluid_bc_parameter(model.fluid.fluid_bc_params, boundary, per_cell_value);

                    for (mhs::Index fluid_index : matched) {
                        model.fluid.fluid_bcs[fluid_index] = {boundary.kind, parameter_index};
                        if (!std::isnan(boundary.inlet_temperature))
                            model.fluid.boundary_temperature_fluid[fluid_index] = boundary.inlet_temperature;
                        if (boundary.kind != mhs::core::FluidBCType::VelocityType)
                            continue;

                        const mhs::Index active_index = model.fluid.fluid_to_global[fluid_index];
                        const mhs::Index grid_index = active_to_grid[active_index];
                        mhs::Index ix, iy, iz;
                        mhs::utils::decode_index(grid_index, mesh.ny, mesh.nz, ix, iy, iz);
                        model.fluid.fluid_face_area[fluid_index] = face_area(face, mesh, ix, iy, iz);
                    }
                }
            }
        }

        void allocate_fluid_fields(mhs::core::FluidDomain& fluid)
        {
            const mhs::Index count = fluid.n_fluid;
            fluid.dynamic_viscosity.assign(count, 0.0);
            fluid.pressure.assign(count, 0.0);
            fluid.flow_axes.assign(count, -1);
            for (auto& conductance : fluid.hydroC)
                conductance.assign(count, 0.0);
            fluid.hydraulic_diameter.assign(count, 0.0);
            fluid.channel_width.assign(count, 0.0);
            fluid.channel_height.assign(count, 0.0);
            fluid.fluid_bcs.assign(count, mhs::core::FluidCellBC {});
            fluid.fluid_bc_params = mhs::core::FluidBCParamTable {};
            fluid.fluid_face_area.assign(count, 0.0);
            fluid.boundary_temperature_fluid.assign(count, std::numeric_limits<double>::quiet_NaN());
        }

    } // namespace

    std::optional<FluidPreprocessWorkspace> buildFluidDomain(mhs::core::Model& model,
        const mhs::core::FluidOverlay& overlay, const mhs::core::IOStructure& io_structure,
        const mhs::core::SymbolTable& symbols, const std::vector<std::string>& material_names)
    {
        if (overlay.fluid_materials.empty())
            return std::nullopt;
        assert(material_names.size() == model.material_table.size());

        std::unordered_map<std::string, std::string> viscosity_by_material;
        for (const auto& material : overlay.fluid_materials)
            viscosity_by_material[material.name] = material.dynamic_viscosity;

        for (size_t material_index = 0; material_index < model.material_table.size(); ++material_index) {
            auto& material = model.material_table[material_index];
            material.is_fluid = false;
            const auto viscosity = viscosity_by_material.find(material_names[material_index]);
            if (viscosity == viscosity_by_material.end() || viscosity->second.empty())
                continue;
            material.is_fluid = true;
            material.dynamic_viscosity
                = mhs::core::parse(substitute_function_args(viscosity->second, "T", io_structure.functions), symbols);
        }

        auto& fluid = model.fluid;
        const mhs::Index active_count = model.cells.material_id.size();
        fluid.is_fluid.assign(active_count, 0);
        fluid.fluid_to_global.clear();
        fluid.global_to_fluid.assign(active_count, mhs::invalidIndex);
        for (mhs::Index active_index = 0; active_index < active_count; ++active_index) {
            const uint16_t material_index = model.cells.material_id[active_index];
            if (!model.material_table[material_index].is_fluid)
                continue;
            fluid.is_fluid[active_index] = 1;
            fluid.global_to_fluid[active_index] = fluid.fluid_to_global.size();
            fluid.fluid_to_global.push_back(active_index);
        }
        fluid.n_fluid = fluid.fluid_to_global.size();
        if (fluid.n_fluid == 0)
            return std::nullopt;

        allocate_fluid_fields(fluid);
        for (mhs::Index fluid_index = 0; fluid_index < fluid.n_fluid; ++fluid_index) {
            const mhs::Index active_index = fluid.fluid_to_global[fluid_index];
            const uint16_t material_index = model.cells.material_id[active_index];
            fluid.dynamic_viscosity[fluid_index] = model.material_table[material_index].dynamic_viscosity.eval(
                {0.0, 0.0, 0.0, model.initial_temperature, 0.0});
        }

        FluidPreprocessWorkspace workspace {build_active_to_grid(model.cells)};
        const double si_scale = mhs::utils::length_unit_to_si(io_structure.length_unit);
        apply_fluid_boundaries(model, overlay, si_scale, workspace.active_to_grid);
        compute_channel_dimensions(model, workspace.active_to_grid);
        return workspace;
    }

} // namespace mhs::sim
