#include "runtime/constants.hpp"
#include "compiler/geometry_compiler.hpp"
#include "runtime/mesh.hpp"
#include "numerics/expression/expr.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace mhs::sim {

    constexpr double EPS = mhs::core::geometry_eps;

    namespace {

        bool point_in_region(const CompiledBoundaryRegion& region, double a, double b)
        {
            return std::any_of(region.rectangles.begin(), region.rectangles.end(), [&](const auto& rectangle) {
                return a >= rectangle.a_min - EPS && a <= rectangle.a_max + EPS && b >= rectangle.b_min - EPS
                    && b <= rectangle.b_max + EPS;
            });
        }

        mhs::Index find_block_for_cell(const ResolvedLayerGeometry& resolved_layer, double cx, double cy, double cz)
        {
            for (mhs::Index b = static_cast<mhs::Index>(resolved_layer.blocks.size()) - 1; b != mhs::invalidIndex;
                b--) {
                const auto& block = resolved_layer.blocks[b];
                if (cz < block.z_start - EPS || cz > block.z_end + EPS)
                    continue;
                bool is_inside = false;
                for (const auto& rect : block.rects) {
                    if (cx >= rect.x - EPS && cx <= rect.x + rect.width + EPS && cy >= rect.y - EPS
                        && cy <= rect.y + rect.height + EPS) {
                        is_inside = rect.operation == mhs::model::GeometryOperation::Add;
                    }
                }
                if (is_inside)
                    return b;
            }
            return mhs::invalidIndex;
        }

        std::pair<mhs::core::BcType, uint16_t> match_face_bc(mhs::core::FaceDir dir, mhs::Index ix, mhs::Index iy,
            mhs::Index iz, const mhs::core::MeshGeometry& mesh, const std::vector<CompiledBoundaryRegion>& boundaries,
            const DefaultBoundary& default_boundary)
        {
            assert(static_cast<size_t>(dir) < mhs::core::FACE_COUNT);
            const int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];
            const mhs::model::Axis face_axis = axis == 0 ? mhs::model::Axis::X
                : axis == 1                              ? mhs::model::Axis::Y
                                                         : mhs::model::Axis::Z;

            const double face_coord = mhs::utils::face_coord_value(dir, ix, iy, iz, mesh);
            const int ta = mhs::utils::TANGENT_A_OF_DIR[static_cast<size_t>(dir)];
            const int tb = mhs::utils::TANGENT_B_OF_DIR[static_cast<size_t>(dir)];
            const double centers[3] = {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz]};
            const double a_val = centers[ta];
            const double b_val = centers[tb];

            // Boundary definitions are ordered overlays: the last matching
            // definition wins, just like later blocks in a layer.
            for (auto it = boundaries.rbegin(); it != boundaries.rend(); ++it) {
                const auto& boundary = *it;
                if (boundary.axis == face_axis
                    && std::abs(face_coord - boundary.coordinate) < mhs::core::geometry_eps) {
                    if (point_in_region(boundary, a_val, b_val))
                        return {boundary.type, boundary.parameter_index};
                }
            }

            if (default_boundary.type != mhs::core::BcType::None)
                return {default_boundary.type, default_boundary.parameter_index};
            return {mhs::core::BcType::None, static_cast<uint16_t>(0)};
        }
    } // anonymous namespace

    std::vector<ResolvedLayerGeometry> resolve_geometry(
        const std::vector<mhs::model::LayerSpec>& layers, double si_scale, const mhs::core::SymbolTable& symbols)
    {
        mhs::Index num_layers = static_cast<mhs::Index>(layers.size());
        std::vector<ResolvedLayerGeometry> resolved(num_layers);

        std::vector<double> thickness(num_layers);
        double z_cursor = 0.0;
        for (mhs::Index l = 0; l < num_layers; l++) {
            if (l == 0) {
                double max_t = 0.0;
                for (const auto& b : layers[l].blocks) {
                    if (b.thickness.has_value()) {
                        double t = mhs::core::eval_geometry(*b.thickness, symbols) * si_scale;
                        if (t > max_t)
                            max_t = t;
                    }
                }
                double layer_t = layers[l].thickness.empty()
                    ? 0.0
                    : mhs::core::eval_geometry(layers[l].thickness, symbols) * si_scale;
                thickness[l] = std::max(max_t, layer_t);
            }
            else {
                thickness[l] = mhs::core::eval_geometry(layers[l].thickness, symbols) * si_scale;
            }
            z_cursor += thickness[l];
        }
        for (mhs::Index l = 0; l < num_layers; l++) {
            resolved[l].z_start = z_cursor - thickness[l];
            resolved[l].z_end = z_cursor;
            z_cursor -= thickness[l];
        }

        for (mhs::Index l = 0; l < num_layers; l++) {
            const auto& layer = layers[l];
            double layer_x_off_si = mhs::core::eval_geometry(layer.x_offset, symbols) * si_scale;
            double layer_y_off_si = mhs::core::eval_geometry(layer.y_offset, symbols) * si_scale;

            for (const auto& block : layer.blocks) {
                ResolvedBlock rb;
                double block_x_off_si = mhs::core::eval_geometry(block.x_offset, symbols) * si_scale;
                double block_y_off_si = mhs::core::eval_geometry(block.y_offset, symbols) * si_scale;
                rb.material = block.material;
                rb.volumetric_heat_source = block.volumetric_heat_source;

                if (l == 0 && block.thickness.has_value()) {
                    double b_thick = mhs::core::eval_geometry(*block.thickness, symbols) * si_scale;
                    rb.z_start = resolved[l].z_start;
                    rb.z_end = resolved[l].z_start + b_thick;
                }
                else {
                    rb.z_start = resolved[l].z_start;
                    rb.z_end = resolved[l].z_end;
                }

                for (const auto& operation : block.geometry) {
                    const auto& rect = operation.rect;
                    ResolvedRect rr;
                    rr.operation = operation.operation;
                    double x_val = mhs::core::eval_geometry(rect.x, symbols);
                    double y_val = mhs::core::eval_geometry(rect.y, symbols);
                    double w_val = mhs::core::eval_geometry(rect.width, symbols);
                    double h_val = mhs::core::eval_geometry(rect.height, symbols);
                    if (w_val < 0) {
                        x_val += w_val;
                        w_val = -w_val;
                    }
                    if (h_val < 0) {
                        y_val += h_val;
                        h_val = -h_val;
                    }
                    rr.x = x_val * si_scale + block_x_off_si + layer_x_off_si;
                    rr.y = y_val * si_scale + block_y_off_si + layer_y_off_si;
                    rr.width = w_val * si_scale;
                    rr.height = h_val * si_scale;
                    rb.rects.push_back(rr);
                }
                resolved[l].blocks.push_back(rb);
            }
        }
        return resolved;
    }

    mhs::core::CellFields assign_cell_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const std::unordered_map<std::string, size_t>& name_to_idx,
        const std::vector<std::vector<uint16_t>>& block_hs_map)
    {
        const mhs::Index num_layers = static_cast<mhs::Index>(resolved_layers.size());
        const mhs::Index total = mesh.nx * mesh.ny * mesh.nz;

        mhs::core::CellFields cells;
        cells.grid_to_cell.resize(total, mhs::invalidIndex);
        cells.cell_to_grid.reserve(total);
        cells.material_id.reserve(total);
        cells.heat_source_idx.reserve(total);

        for (mhs::Index ix = 0; ix < mesh.nx; ix++) {
            for (mhs::Index iy = 0; iy < mesh.ny; iy++) {
                for (mhs::Index iz = 0; iz < mesh.nz; iz++) {
                    mhs::Index old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;

                    double cx = mesh.cx[ix];
                    double cy = mesh.cy[iy];
                    double cz = mesh.cz[iz];

                    mhs::Index layer_idx = mhs::invalidIndex;
                    mhs::Index block_idx = mhs::invalidIndex;

                    for (mhs::Index l = 0; l < num_layers; l++) {
                        if (cz >= resolved_layers[l].z_start - EPS && cz <= resolved_layers[l].z_end + EPS) {
                            mhs::Index b = find_block_for_cell(resolved_layers[l], cx, cy, cz);
                            if (b != mhs::invalidIndex) {
                                layer_idx = l;
                                block_idx = b;
                                break;
                            }
                        }
                    }

                    if (layer_idx != mhs::invalidIndex && block_idx != mhs::invalidIndex) {
                        const auto& block = resolved_layers[layer_idx].blocks[block_idx];
                        const mhs::Index c_idx = static_cast<mhs::Index>(cells.material_id.size());
                        cells.grid_to_cell[old_idx] = c_idx;
                        cells.cell_to_grid.push_back(old_idx);
                        cells.material_id.push_back(static_cast<uint16_t>(name_to_idx.at(block.material)));
                        cells.heat_source_idx.push_back(block_hs_map[layer_idx][block_idx]);
                    }
                }
            }
        }
        return cells;
    }

    void resolve_boundary_patches(const mhs::core::MeshGeometry& mesh, const mhs::core::CellFields& cells,
        const std::vector<CompiledBoundaryRegion>& boundaries, const DefaultBoundary& default_boundary,
        std::vector<mhs::core::FaceBC>& face_bcs)
    {
        const mhs::Index compact_count = static_cast<mhs::Index>(cells.material_id.size());
        face_bcs.assign(compact_count * mhs::core::FACE_COUNT, mhs::core::FaceBC {});

        for (mhs::Index ix = 0; ix < mesh.nx; ix++) {
            for (mhs::Index iy = 0; iy < mesh.ny; iy++) {
                for (mhs::Index iz = 0; iz < mesh.nz; iz++) {
                    mhs::Index old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    mhs::Index c_idx = cells.grid_to_cell[old_idx];
                    if (c_idx == mhs::invalidIndex)
                        continue;

                    for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
                        mhs::core::FaceDir dir = mhs::core::FACE_DIRS[f];
                        if (mhs::utils::neighbor_grid_index(
                                ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.grid_to_cell)
                            != mhs::invalidIndex)
                            continue;

                        auto [type, param_idx] = match_face_bc(dir, ix, iy, iz, mesh, boundaries, default_boundary);
                        face_bcs[c_idx * mhs::core::FACE_COUNT + f] = {type, param_idx};
                    }
                }
            }
        }
    }

} // namespace mhs::sim
