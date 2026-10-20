// Assembly of the finite-volume operators of Sec. II-B, eq. (2)-(3), split by
// region, plus the interface faces that eq. (7)-(10) need.
#pragma once

#include "mesh.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace mhs::hdim {

    /// One interface face: the nonlinear cell behind it, the linear cell behind
    /// it on the other side, and the two half resistances of the plane.
    struct InterfaceFace {
        int nonlinear_cell = -1;
        int linear_cell = -1;
        int linear_region = kLinearTop;
        double area = 0.0; // m^2
        double cx = 0.0; // mm, centre
        double cy = 0.0;
        double g_nonlinear = 0.0; // W/K, nonlinear half
        double r_linear = 0.0; // K/W, linear half
    };

    /// Series conductance of an interface face: the nonlinear half cell, the
    /// linear half cell, and nothing else.
    inline double interface_conductance(const InterfaceFace& face)
    {
        return 1.0 / (1.0 / face.g_nonlinear + face.r_linear);
    }

    struct Model {
        Mesh mesh;
        std::vector<int> region_cells[kRegionCount];
        std::vector<int> cell_region;
        std::vector<int> local; // global cell -> index inside its region
        std::vector<double> power; // W per cell
        /// Per die: the cells it dissipates in and the wattage of each, so that
        /// the alternating workload of Sec. III-B can be switched on and off.
        std::array<std::vector<int>, 2> chip_cell; // 0: chip1, 1: chip2
        std::array<std::vector<double>, 2> chip_watt;
    };

    /// The linear system of one region: K = C / dt + G, with the convection
    /// already on the diagonal and its ambient load in `source`.
    struct RegionSystem {
        Eigen::SparseMatrix<double> K;
        Eigen::VectorXd capacity;
        Eigen::VectorXd source;
    };

    struct Assembled {
        RegionSystem region[kRegionCount];
        std::vector<InterfaceFace> interfaces; // both interfaces, in face order
    };

    /// Counters shared by the reference FVM and by the HDI method, so that their
    /// cost can be compared solve by solve.
    struct RunInfo {
        double seconds = 0.0;
        int sweeps = 0; // nonlinear (fixed-point) sweeps
        long long solves = 0; // linear solves
    };

    inline double cell_conductivity(const Layer& layer, double temperature, bool through_plane)
    {
        if (layer.temperature_dependent)
            return die_conductivity(temperature > 1.0 ? temperature : kAmbient);
        return through_plane ? layer.k_perpendicular : layer.k_parallel;
    }

    inline void build_model(Model& model, const Level& level)
    {
        model.mesh = build_mesh(level);
        const Mesh& mesh = model.mesh;
        model.cell_region.assign(static_cast<std::size_t>(mesh.n), -1);
        model.local.assign(static_cast<std::size_t>(mesh.n), -1);
        model.power.assign(static_cast<std::size_t>(mesh.n), 0.0);

        for (int c = 0; c < mesh.n; ++c)
            model.cell_region[static_cast<std::size_t>(c)] = static_cast<int>(
                kStack[static_cast<std::size_t>(mesh.cell_layer[static_cast<std::size_t>(c)])].region);
        for (int r = 0; r < kRegionCount; ++r)
            for (int c = 0; c < mesh.n; ++c)
                if (model.cell_region[static_cast<std::size_t>(c)] == r) {
                    model.local[static_cast<std::size_t>(c)]
                        = static_cast<int>(model.region_cells[r].size());
                    model.region_cells[r].push_back(c);
                }

        // Power: each chiplet's wattage spread over the die cells inside it.
        // Chip1 is the bottom die, kStack[5], i.e. the Multi-GPU chip of Fig. 4;
        // chip2 is kStack[3], the die next to the TIM, the AI chip of Fig. 5.
        const std::array<ChipSpec, 2> chips { multi_gpu_chip(), ai_chip() };
        const std::array<int, 2> die_layer { 5, 3 }; // chip1, chip2
        for (int chip = 0; chip < 2; ++chip) {
            const int layer_index = die_layer[static_cast<std::size_t>(chip)];
            const auto& cells_of_layer = mesh.layers[static_cast<std::size_t>(layer_index)];
            for (const Chiplet& ch : chips[static_cast<std::size_t>(chip)].chiplets) {
                std::vector<int> cells;
                double volume = 0.0;
                for (int k = 0; k < cells_of_layer.nz; ++k)
                    for (int j = 0; j < mesh.ny; ++j) {
                        const double cy = 0.5 * (mesh.y[static_cast<std::size_t>(j)]
                            + mesh.y[static_cast<std::size_t>(j + 1)]);
                        for (int i = 0; i < mesh.nx; ++i) {
                            const double cx = 0.5 * (mesh.x[static_cast<std::size_t>(i)]
                                + mesh.x[static_cast<std::size_t>(i + 1)]);
                            const int id = mesh.id(layer_index, i, j, k);
                            if (id < 0 || !contains(ch.rect, cx + 0.5 * kChipX, cy + 0.5 * kChipY))
                                continue;
                            cells.push_back(id);
                            volume += mesh.volume[static_cast<std::size_t>(id)];
                        }
                    }
                if (cells.empty())
                    throw std::runtime_error(std::string("chiplet outside the mesh: ") + ch.name);
                for (const int id : cells) {
                    const double watt = ch.power * mesh.volume[static_cast<std::size_t>(id)] / volume;
                    model.chip_cell[static_cast<std::size_t>(chip)].push_back(id);
                    model.chip_watt[static_cast<std::size_t>(chip)].push_back(watt);
                    model.power[static_cast<std::size_t>(id)] += watt;
                }
            }
        }
    }

    /// Operating mode of Sec. III-B: which of the two chips dissipates. The idle
    /// one keeps its material properties but no power.
    inline void set_workload(Model& model, std::array<bool, 2> on)
    {
        std::fill(model.power.begin(), model.power.end(), 0.0);
        for (std::size_t chip = 0; chip < 2; ++chip)
            if (on[chip])
                for (std::size_t i = 0; i < model.chip_cell[chip].size(); ++i)
                    model.power[static_cast<std::size_t>(model.chip_cell[chip][i])] += model.chip_watt[chip][i];
    }

    /// Assemble every region at the given temperature (K, per cell; the values are
    /// only read inside the dies) and the step `dt` (s; dt <= 0 means steady state,
    /// which is eq. (3) with the capacity term dropped). `only_region` restricts
    /// the couplings to one region; the interface faces are always produced, and
    /// the other regions come back empty.
    inline Assembled assemble(const Model& model, const std::vector<double>& temperature, double dt,
        int only_region = -1)
    {
        const Mesh& mesh = model.mesh;
        Assembled out;
        std::array<std::vector<Eigen::Triplet<double>>, kRegionCount> triplets;
        for (int r = 0; r < kRegionCount; ++r) {
            const auto n = static_cast<Eigen::Index>(model.region_cells[r].size());
            out.region[r].capacity = Eigen::VectorXd::Zero(n);
            out.region[r].source = Eigen::VectorXd::Zero(n);
        }

        for (int r = 0; r < kRegionCount; ++r)
            for (std::size_t idx = 0; idx < model.region_cells[r].size(); ++idx) {
                const int c = model.region_cells[r][idx];
                const Layer& layer = kStack[static_cast<std::size_t>(mesh.cell_layer[static_cast<std::size_t>(c)])];
                const double capacity = layer.rho * layer.cp * mesh.volume[static_cast<std::size_t>(c)];
                out.region[r].capacity(static_cast<Eigen::Index>(idx)) = capacity;
                out.region[r].source(static_cast<Eigen::Index>(idx)) = model.power[static_cast<std::size_t>(c)];
                if (dt > 0.0) {
                    triplets[r].emplace_back(
                        static_cast<Eigen::Index>(idx), static_cast<Eigen::Index>(idx), capacity / dt);
                }
            }

        auto add_coupling = [&](int a, int b, double conductance) {
            const int r = model.cell_region[static_cast<std::size_t>(a)];
            const auto la = static_cast<Eigen::Index>(model.local[static_cast<std::size_t>(a)]);
            const auto lb = static_cast<Eigen::Index>(model.local[static_cast<std::size_t>(b)]);
            triplets[r].emplace_back(la, la, conductance);
            triplets[r].emplace_back(lb, lb, conductance);
            triplets[r].emplace_back(la, lb, -conductance);
            triplets[r].emplace_back(lb, la, -conductance);
        };

        // In-plane faces never cross a region (regions are whole layers), so they
        // are handled inside each layer, and the through-plane faces are handled by
        // the layer above the pair.
        for (int layer_index = 0; layer_index < static_cast<int>(kStack.size()); ++layer_index) {
            const Layer& layer = kStack[static_cast<std::size_t>(layer_index)];
            const auto& cells = mesh.layers[static_cast<std::size_t>(layer_index)];
            const bool active = only_region < 0 || layer.region == only_region;
            for (int k = 0; k < cells.nz; ++k)
                for (int j = 0; j < mesh.ny; ++j) {
                    const double dy = mesh.y[static_cast<std::size_t>(j + 1)] - mesh.y[static_cast<std::size_t>(j)];
                    const double cy = 0.5 * (mesh.y[static_cast<std::size_t>(j)]
                        + mesh.y[static_cast<std::size_t>(j + 1)]);
                    for (int i = 0; i < mesh.nx; ++i) {
                        const int a = mesh.id(layer_index, i, j, k);
                        if (a < 0)
                            continue;
                        const double dx = mesh.x[static_cast<std::size_t>(i + 1)] - mesh.x[static_cast<std::size_t>(i)];
                        const double cx = 0.5 * (mesh.x[static_cast<std::size_t>(i)]
                            + mesh.x[static_cast<std::size_t>(i + 1)]);
                        const double ta = temperature.empty() ? kAmbient : temperature[static_cast<std::size_t>(a)];
                        const double ka_plane = cell_conductivity(layer, ta, false);
                        const double ka_depth = cell_conductivity(layer, ta, true);

                        // +x
                        const int bx = mesh.id(layer_index, i + 1, j, k);
                        if (active && bx >= 0) {
                            const double dxb = mesh.x[static_cast<std::size_t>(i + 2)]
                                - mesh.x[static_cast<std::size_t>(i + 1)];
                            const double tb = temperature.empty() ? kAmbient : temperature[static_cast<std::size_t>(bx)];
                            const double kb = cell_conductivity(layer, tb, false);
                            const double area = 1e-6 * dy * cells.dz;
                            add_coupling(a, bx, area / (0.5e-3 * dx / ka_plane + 0.5e-3 * dxb / kb));
                        }
                        // +y
                        const int by = mesh.id(layer_index, i, j + 1, k);
                        if (active && by >= 0) {
                            const double dyb = mesh.y[static_cast<std::size_t>(j + 2)]
                                - mesh.y[static_cast<std::size_t>(j + 1)];
                            const double tb = temperature.empty() ? kAmbient : temperature[static_cast<std::size_t>(by)];
                            const double kb = cell_conductivity(layer, tb, false);
                            const double area = 1e-6 * dx * cells.dz;
                            add_coupling(a, by, area / (0.5e-3 * dy / ka_plane + 0.5e-3 * dyb / kb));
                        }
                        // +z within the layer
                        if (active && k + 1 < cells.nz) {
                            const int bz = mesh.id(layer_index, i, j, k + 1);
                            if (bz >= 0) {
                                const double tb = temperature.empty() ? kAmbient : temperature[static_cast<std::size_t>(bz)];
                                const double kb = cell_conductivity(layer, tb, true);
                                const double area = 1e-6 * dx * dy;
                                add_coupling(a, bz, area / (0.5e-3 * cells.dz / ka_depth + 0.5e-3 * cells.dz / kb));
                            }
                        }
                        // +z into the next layer: interior of a region, or an interface
                        if (k + 1 == cells.nz && layer_index + 1 < static_cast<int>(kStack.size())) {
                            const int bz = mesh.id(layer_index + 1, i, j, 0);
                            if (bz >= 0) {
                                const Layer& below = kStack[static_cast<std::size_t>(layer_index + 1)];
                                const double dz_below = mesh.layers[static_cast<std::size_t>(layer_index + 1)].dz;
                                const double tb = temperature.empty() ? kAmbient : temperature[static_cast<std::size_t>(bz)];
                                const double kb_depth = cell_conductivity(below, tb, true);
                                const double area = 1e-6 * dx * dy;
                                if (layer.region == below.region) {
                                    if (active)
                                        add_coupling(a, bz,
                                            area / (0.5e-3 * cells.dz / ka_depth + 0.5e-3 * dz_below / kb_depth));
                                }
                                else {
                                    InterfaceFace face;
                                    face.nonlinear_cell = layer.region == kNonlinear ? a : bz;
                                    face.linear_cell = layer.region == kNonlinear ? bz : a;
                                    face.linear_region = layer.region == kNonlinear ? below.region : layer.region;
                                    face.area = area;
                                    face.cx = cx;
                                    face.cy = cy;
                                    const double dz_n = layer.region == kNonlinear ? cells.dz : dz_below;
                                    const double k_n = layer.region == kNonlinear ? ka_depth : kb_depth;
                                    const double dz_l = layer.region == kNonlinear ? dz_below : cells.dz;
                                    const double k_l = layer.region == kNonlinear ? kb_depth : ka_depth;
                                    face.g_nonlinear = 2.0 * area * k_n / (1e-3 * dz_n);
                                    face.r_linear = 0.5e-3 * dz_l / (area * k_l);
                                    out.interfaces.push_back(face);
                                }
                            }
                        }
                    }
                }
        }

        // Convection: the top of the stack and the bottom of the PCB, eq. (2)'s B.
        for (int side = 0; side < 2; ++side) {
            const int layer_index = side == 0 ? 0 : static_cast<int>(kStack.size()) - 1;
            const Layer& layer = kStack[static_cast<std::size_t>(layer_index)];
            const auto& cells = mesh.layers[static_cast<std::size_t>(layer_index)];
            if (only_region >= 0 && layer.region != only_region)
                continue;
            const int k = side == 0 ? 0 : cells.nz - 1;
            const double h = side == 0 ? kSinkH : kPcbH;
            for (int j = 0; j < mesh.ny; ++j)
                for (int i = 0; i < mesh.nx; ++i) {
                    const int a = mesh.id(layer_index, i, j, k);
                    if (a < 0)
                        continue;
                    const int r = model.cell_region[static_cast<std::size_t>(a)];
                    const auto la = static_cast<Eigen::Index>(model.local[static_cast<std::size_t>(a)]);
                    const double dx = mesh.x[static_cast<std::size_t>(i + 1)] - mesh.x[static_cast<std::size_t>(i)];
                    const double dy = mesh.y[static_cast<std::size_t>(j + 1)] - mesh.y[static_cast<std::size_t>(j)];
                    const double conductance = h * 1e-6 * dx * dy;
                    triplets[r].emplace_back(la, la, conductance);
                    out.region[r].source(la) += conductance * kAmbient;
                }
        }

        for (int r = 0; r < kRegionCount; ++r) {
            const auto n = static_cast<Eigen::Index>(model.region_cells[r].size());
            out.region[r].K.resize(n, n);
            out.region[r].K.setFromTriplets(triplets[r].begin(), triplets[r].end());
            out.region[r].K.makeCompressed();
        }
        return out;
    }

    /// The same discretisation as one global operator: the three region matrices
    /// plus the series conductance of every interface face. This is the standard
    /// FVM of Sec. III, i.e. the baseline the HDI method is measured against.
    inline RegionSystem assemble_whole(const Model& model, const std::vector<double>& temperature, double dt)
    {
        const Assembled assembled = assemble(model, temperature, dt);
        const int n = model.mesh.n;
        RegionSystem whole;
        whole.capacity = Eigen::VectorXd::Zero(n);
        whole.source = Eigen::VectorXd::Zero(n);

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<std::size_t>(n) * 8);
        for (int r = 0; r < kRegionCount; ++r) {
            const RegionSystem& region = assembled.region[r];
            for (int k = 0; k < region.K.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(region.K, k); it; ++it) {
                    triplets.emplace_back(model.region_cells[r][static_cast<std::size_t>(it.row())],
                        model.region_cells[r][static_cast<std::size_t>(it.col())], it.value());
                }
            for (std::size_t idx = 0; idx < model.region_cells[r].size(); ++idx) {
                const int c = model.region_cells[r][idx];
                whole.capacity(c) = region.capacity(static_cast<Eigen::Index>(idx));
                whole.source(c) = region.source(static_cast<Eigen::Index>(idx));
            }
        }
        for (const InterfaceFace& face : assembled.interfaces) {
            const double conductance = interface_conductance(face);
            triplets.emplace_back(face.nonlinear_cell, face.nonlinear_cell, conductance);
            triplets.emplace_back(face.linear_cell, face.linear_cell, conductance);
            triplets.emplace_back(face.nonlinear_cell, face.linear_cell, -conductance);
            triplets.emplace_back(face.linear_cell, face.nonlinear_cell, -conductance);
        }
        whole.K.resize(n, n);
        whole.K.setFromTriplets(triplets.begin(), triplets.end());
        whole.K.makeCompressed();
        return whole;
    }

} // namespace mhs::hdim
