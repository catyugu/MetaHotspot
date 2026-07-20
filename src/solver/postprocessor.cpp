#include "runtime/mesh.hpp"
#include "solver/interpolation.hpp"
#include "solver/postprocessor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace mhs::post {

    std::vector<double> interpolate_cell_to_node(
        const mhs::core::Model& model, const std::vector<double>& cell_temperature, double time)
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
                    samples.reserve(16);
                    double stencil_min = std::numeric_limits<double>::infinity();
                    double stencil_max = -std::numeric_limits<double>::infinity();

                    // 遍历该节点周边相接的最多 8 个体元
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
                                stencil_min = std::min(stencil_min, T_c);
                                stencil_max = std::max(stencil_max, T_c);

                                double cx = mesh.cx[ix];
                                double cy = mesh.cy[iy];
                                double cz = mesh.cz[iz];
                                const auto& material = model.material_table[cells.material_id[compact_idx]];
                                const mhs::core::FieldContext context {cx, cy, cz, T_c, time};
                                const double kx = material.kx.eval(context);
                                const double ky = material.ky.eval(context);
                                const double kz = material.kz.eval(context);
                                double c_dx = cx - node_x;
                                double c_dy = cy - node_y;
                                double c_dz = cz - node_z;
                                const double thermal_distance_squared
                                    = c_dx * c_dx / kx + c_dy * c_dy / ky + c_dz * c_dz / kz;
                                samples.push_back({cx, cy, cz, T_c, 1.0 / thermal_distance_squared, kx, ky, kz,
                                    cells.material_id[compact_idx], true});

                                mhs::core::FaceDir f_x = (dx == -1) ? mhs::core::FaceDir::XP : mhs::core::FaceDir::XM;
                                mhs::core::FaceDir f_y = (dy == -1) ? mhs::core::FaceDir::YP : mhs::core::FaceDir::YM;
                                mhs::core::FaceDir f_z = (dz == -1) ? mhs::core::FaceDir::ZP : mhs::core::FaceDir::ZM;

                                mhs::core::FaceDir dirs[3] = {f_x, f_y, f_z};

                                auto* fc = &face_bcs[compact_idx * mhs::core::FACE_COUNT];
                                for (mhs::core::FaceDir dir : dirs) {
                                    const auto& fb = fc[static_cast<size_t>(dir)];
                                    if (fb.type == mhs::core::BcType::None)
                                        continue;

                                    if (fb.type == mhs::core::BcType::FirstType) {
                                        dirichlet_sum += bc_params.dirichlet_T[fb.param_idx].eval(
                                            {node_x, node_y, node_z, T_c, time});
                                        dirichlet_count++;
                                    }
                                    else {
                                        const double face_k = mhs::utils::k_along(dir, kx, ky, kz);
                                        const double face_temperature = mhs::utils::sample_extrapolate_face_temperature(
                                            dir, fb.type, fb.param_idx, T_c, face_k, mesh, ix, iy, iz, bc_params, time);
                                        stencil_min = std::min(stencil_min, face_temperature);
                                        stencil_max = std::max(stencil_max, face_temperature);
                                        double fx, fy, fz;
                                        mhs::utils::face_center_3d(dir, ix, iy, iz, mesh, fx, fy, fz);
                                        const double fdx = fx - node_x;
                                        const double fdy = fy - node_y;
                                        const double fdz = fz - node_z;
                                        const double face_distance_squared
                                            = fdx * fdx / kx + fdy * fdy / ky + fdz * fdz / kz;
                                        samples.push_back({fx, fy, fz, face_temperature, 1.0 / face_distance_squared,
                                            kx, ky, kz, cells.material_id[compact_idx], false});
                                    }
                                }
                            }
                        }
                    }

                    if (dirichlet_count > 0) {
                        node_T[node_idx] = dirichlet_sum / dirichlet_count;
                    }
                    else if (!samples.empty()) {
                        node_T[node_idx]
                            = std::clamp(mhs::utils::recover_point_temperature(samples, node_x, node_y, node_z),
                                stencil_min, stencil_max);
                    }
                }
            }
        }

        return node_T;
    }

    double max_temperature(const std::vector<double>& T)
    {
        if (T.empty())
            return 0.0;
        double max_val = std::numeric_limits<double>::quiet_NaN();
        for (double v : T) {
            if (std::isnan(v))
                continue;
            if (std::isnan(max_val) || v > max_val)
                max_val = v;
        }
        return std::isnan(max_val) ? 0.0 : max_val;
    }

    double min_temperature(const std::vector<double>& T)
    {
        if (T.empty())
            return 0.0;
        double min_val = std::numeric_limits<double>::quiet_NaN();
        for (double v : T) {
            if (std::isnan(v))
                continue;
            if (std::isnan(min_val) || v < min_val)
                min_val = v;
        }
        return std::isnan(min_val) ? 0.0 : min_val;
    }

} // namespace mhs::post
