#include "solver/postprocessor.hpp"
#include "solver/interpolation.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

namespace mhs::post {

    std::vector<double> interpolate_cell_to_node(
        const mhs::core::Model& model, std::span<const double> cell_temperature, double time)
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;
        const auto& face_bcs = model.face_bcs;
        const auto& bc_params = model.bc_params;

        mhs::core::Index node_nx = mesh.nx + 1;
        mhs::core::Index node_ny = mesh.ny + 1;
        mhs::core::Index node_nz = mesh.nz + 1;
        mhs::core::Index total_nodes = node_nx * node_ny * node_nz;

        std::vector<double> node_T(total_nodes, std::numeric_limits<double>::quiet_NaN());

        for (mhs::core::Index vx = 0; vx < node_nx; vx++) {
            for (mhs::core::Index vy = 0; vy < node_ny; vy++) {
                for (mhs::core::Index vz = 0; vz < node_nz; vz++) {
                    mhs::core::Index node_idx = vx * node_ny * node_nz + vy * node_nz + vz;

                    double node_x = (vx == 0) ? mesh.cx[0] - mesh.dx[0] * 0.5 : mesh.cx[vx - 1] + mesh.dx[vx - 1] * 0.5;
                    double node_y = (vy == 0) ? mesh.cy[0] - mesh.dy[0] * 0.5 : mesh.cy[vy - 1] + mesh.dy[vy - 1] * 0.5;
                    double node_z = (vz == 0) ? mesh.cz[0] - mesh.dz[0] * 0.5 : mesh.cz[vz - 1] + mesh.dz[vz - 1] * 0.5;

                    double dirichlet_sum = 0.0;
                    int dirichlet_count = 0;

                    std::vector<mhs::utils::PointTemperatureSample> samples;
                    samples.reserve(8);

                    // Walk the (at most 8) cells that share this node
                    for (int dx = -1; dx <= 0; dx++) {
                        mhs::core::Index ix = static_cast<mhs::core::Index>(static_cast<int64_t>(vx) + dx);
                        if (ix >= mesh.nx)
                            continue;

                        for (int dy = -1; dy <= 0; dy++) {
                            mhs::core::Index iy = static_cast<mhs::core::Index>(static_cast<int64_t>(vy) + dy);
                            if (iy >= mesh.ny)
                                continue;

                            for (int dz = -1; dz <= 0; dz++) {
                                mhs::core::Index iz = static_cast<mhs::core::Index>(static_cast<int64_t>(vz) + dz);
                                if (iz >= mesh.nz)
                                    continue;

                                mhs::core::Index cell_grid_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                                if (cells.grid_to_cell[cell_grid_idx] == mhs::core::invalidIndex)
                                    continue;

                                mhs::core::Index compact_idx = cells.grid_to_cell[cell_grid_idx];
                                assert(compact_idx != mhs::core::invalidIndex);
                                double T_c = cell_temperature[compact_idx];

                                double cx = mesh.cx[ix];
                                double cy = mesh.cy[iy];
                                double cz = mesh.cz[iz];
                                const auto& material = model.material_table[cells.material_id[compact_idx]];
                                const mhs::core::FieldContext context {cx, cy, cz, T_c, time};
                                const double kx = material.kx.eval(context);
                                const double ky = material.ky.eval(context);
                                const double kz = material.kz.eval(context);

                                // Check for Dirichlet on any face of this cell that touches the node
                                mhs::core::FaceDir f_x = (dx == -1) ? mhs::core::FaceDir::XP : mhs::core::FaceDir::XM;
                                mhs::core::FaceDir f_y = (dy == -1) ? mhs::core::FaceDir::YP : mhs::core::FaceDir::YM;
                                mhs::core::FaceDir f_z = (dz == -1) ? mhs::core::FaceDir::ZP : mhs::core::FaceDir::ZM;
                                mhs::core::FaceDir dirs[3] = {f_x, f_y, f_z};

                                auto* fc = &face_bcs[compact_idx * mhs::core::FACE_COUNT];
                                for (mhs::core::FaceDir dir : dirs) {
                                    const auto& fb = fc[static_cast<size_t>(dir)];
                                    if (fb.type == mhs::core::BcType::FirstType) {
                                        dirichlet_sum += bc_params.dirichlet_T[fb.param_idx].eval(
                                            {node_x, node_y, node_z, T_c, time});
                                        dirichlet_count++;
                                    }
                                }

                                samples.push_back({cx, cy, cz, T_c, kx, ky, kz});
                            }
                        }
                    }

                    if (dirichlet_count > 0) {
                        node_T[node_idx] = dirichlet_sum / dirichlet_count;
                    }
                    else if (!samples.empty()) {
                        node_T[node_idx] = mhs::utils::recover_point_temperature(samples, node_x, node_y, node_z);
                    }
                }
            }
        }

        return node_T;
    }

    double max_temperature(std::span<const double> T)
    {
        auto filtered = T | std::views::filter([](double v) { return !std::isnan(v); });
        if (filtered.empty())
            return std::numeric_limits<double>::quiet_NaN();
        return std::ranges::max(filtered);
    }

    double min_temperature(std::span<const double> T)
    {
        auto filtered = T | std::views::filter([](double v) { return !std::isnan(v); });
        if (filtered.empty())
            return std::numeric_limits<double>::quiet_NaN();
        return std::ranges::min(filtered);
    }

} // namespace mhs::post
