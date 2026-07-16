#include "data/tolerance_config.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "function_helpers.hpp"
#include "linear_solver/linear_solver.hpp"
#include "logger/logger.hpp"
#include "preprocessor/fluid_preprocessor.hpp"
#include "utils/mesh_utils.hpp"
#include "utils/physics_utils.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace mhs::sim {

    namespace {
        static std::vector<mhs::Index> compact_to_old;

        void buildCompactToOld(const mhs::core::CellFields& cells, mhs::Index total_grid)
        {
            compact_to_old.assign(cells.material_id.size(), mhs::invalidIndex);
            for (mhs::Index old_idx = 0; old_idx < total_grid; ++old_idx) {
                mhs::Index c = cells.index_map[old_idx];
                if (c != mhs::invalidIndex)
                    compact_to_old[c] = old_idx;
            }
        }

        double measure_fluid_extent(
            const mhs::core::Model& model, mhs::Index ix, mhs::Index iy, mhs::Index iz, int axis)
        {
            const auto& mesh = model.mesh;
            const auto& cells = model.cells;

            auto is_fluid_cell = [&](mhs::Index cx, mhs::Index cy, mhs::Index cz) {
                if (cx >= mesh.nx || cy >= mesh.ny || cz >= mesh.nz)
                    return false;
                mhs::Index old_idx = cx * mesh.ny * mesh.nz + cy * mesh.nz + cz;
                mhs::Index c_idx = cells.index_map[old_idx];
                return c_idx != mhs::invalidIndex && c_idx < model.fluid.is_fluid.size() && model.fluid.is_fluid[c_idx];
            };

            const mhs::Index sizes[3] = {mesh.nx, mesh.ny, mesh.nz};
            mhs::Index idx[3] = {ix, iy, iz};
            mhs::Index min_idx = idx[axis], max_idx = idx[axis];

            while (min_idx > 0) {
                idx[axis] = min_idx - 1;
                if (!is_fluid_cell(idx[0], idx[1], idx[2]))
                    break;
                min_idx--;
            }

            idx[axis] = max_idx;
            while (max_idx < sizes[axis] - 1) {
                idx[axis] = max_idx + 1;
                if (!is_fluid_cell(idx[0], idx[1], idx[2]))
                    break;
                max_idx++;
            }

            const auto& c_array = (axis == 0) ? mesh.cx : (axis == 1) ? mesh.cy : mesh.cz;
            const auto& d_array = (axis == 0) ? mesh.dx : (axis == 1) ? mesh.dy : mesh.dz;
            return (c_array[max_idx] + d_array[max_idx] * 0.5) - (c_array[min_idx] - d_array[min_idx] * 0.5);
        }

        void computeChannelDimensions(mhs::core::Model& model)
        {
            const auto& mesh = model.mesh;
            for (mhs::Index fi = 0; fi < model.fluid.n_fluid; ++fi) {
                mhs::Index old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
                mhs::Index ix, iy, iz;
                mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                double lengths[3] = {measure_fluid_extent(model, ix, iy, iz, 0),
                    measure_fluid_extent(model, ix, iy, iz, 1), measure_fluid_extent(model, ix, iy, iz, 2)};

                std::sort(lengths, lengths + 3);
                double cross_w = lengths[0];
                double cross_h = lengths[1];

                double dh = (cross_w + cross_h > mhs::core::geometry_eps)
                    ? (2.0 * cross_w * cross_h / (cross_w + cross_h))
                    : 0.0;

                model.fluid.hydraulic_diameter[fi] = dh;
                model.fluid.channel_width[fi] = cross_w;
                model.fluid.channel_height[fi] = cross_h;
            }
        }

        static uint16_t registerFluidBCParam(
            mhs::core::FluidBCParamTable& params, const mhs::core::FluidBoundaryOverlay& fb, double per_cell_value)
        {
            switch (fb.kind) {
            case mhs::core::FluidBCType::PressureType:
                params.pressure.push_back(per_cell_value);
                return static_cast<uint16_t>(params.pressure.size() - 1);
            case mhs::core::FluidBCType::MassFlowRateType:
                params.mass_flow_rate.push_back(per_cell_value);
                return static_cast<uint16_t>(params.mass_flow_rate.size() - 1);
            case mhs::core::FluidBCType::VelocityType:
                params.velocity.push_back(per_cell_value);
                return static_cast<uint16_t>(params.velocity.size() - 1);
            case mhs::core::FluidBCType::None:
            default:
                return static_cast<uint16_t>(std::numeric_limits<uint16_t>::max());
            }
        }

        static void cacheFaceAreaForVelocity(std::vector<double>& face_area, mhs::Index fi, const FaceKeyInfo& fk,
            const mhs::core::MeshGeometry& mesh, mhs::Index ix, mhs::Index iy, mhs::Index iz)
        {
            int axis = (fk.axis == 'X') ? 0 : (fk.axis == 'Y') ? 1 : 2;
            double a = (axis == 0) ? mesh.dy[iy] : mesh.dx[ix];
            double b = (axis == 2) ? mesh.dy[iy] : mesh.dz[iz];
            if (face_area.size() < fi + 1)
                face_area.resize(fi + 1, 0.0);
            face_area[fi] = a * b;
        }

        void applyFluidBoundaries(mhs::core::Model& model, const mhs::core::FluidOverlay& overlay, double si_scale)
        {
            const auto& mesh = model.mesh;
            for (const auto& fb : overlay.boundaries) {
                for (const auto& keyStr : fb.face_keys) {
                    FaceKeyInfo fk = parse_face_key(keyStr, si_scale);
                    int target_axis = (fk.axis == 'X') ? 0 : (fk.axis == 'Y') ? 1 : 2;

                    std::vector<mhs::Index> matched;
                    matched.reserve(64);
                    for (mhs::Index fi = 0; fi < model.fluid.n_fluid; ++fi) {
                        mhs::Index old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
                        mhs::Index ix, iy, iz;
                        mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                        double c[3] = {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz]};
                        double d[3] = {mesh.dx[ix], mesh.dy[iy], mesh.dz[iz]};

                        double face_m = c[target_axis] - d[target_axis] * 0.5;
                        double face_p = c[target_axis] + d[target_axis] * 0.5;

                        if (std::abs(face_m - fk.coord_value) < mhs::core::geometry_eps
                            || std::abs(face_p - fk.coord_value) < mhs::core::geometry_eps) {
                            double a = c[(target_axis + 1) % 3];
                            double b = c[(target_axis + 2) % 3];

                            if (point_in_face_rects(fk, a, b) || point_in_face_rects(fk, b, a)) {
                                matched.push_back(fi);
                            }
                        }
                    }

                    if (matched.empty())
                        continue;

                    double per_cell_value = fb.value;
                    if (fb.kind == mhs::core::FluidBCType::MassFlowRateType) {
                        per_cell_value = fb.value / static_cast<double>(matched.size());
                    }

                    uint16_t param_idx = registerFluidBCParam(model.fluid.fluid_bc_params, fb, per_cell_value);

                    for (mhs::Index fi : matched) {
                        mhs::Index old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
                        mhs::Index ix, iy, iz;
                        mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

                        mhs::core::FluidCellBC cell_bc;
                        cell_bc.kind = fb.kind;
                        cell_bc.param_idx = param_idx;
                        model.fluid.fluid_bcs[fi] = cell_bc;
                        if (!std::isnan(fb.inlet_temperature)) {
                            model.fluid.boundary_temperature_fluid[fi] = fb.inlet_temperature;
                        }
                        if (fb.kind == mhs::core::FluidBCType::VelocityType) {
                            cacheFaceAreaForVelocity(model.fluid.fluid_face_area, fi, fk, mesh, ix, iy, iz);
                        }
                    }
                }
            }
        }

        static double evaluateFluidRhoAtInitT(const mhs::core::Model& model, mhs::Index fi)
        {
            const auto& cells = model.cells;
            const auto& mesh = model.mesh;
            mhs::Index c_idx = model.fluid.fluid_to_global[fi];
            mhs::Index old_idx = compact_to_old[c_idx];
            mhs::Index ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);
            int mat_id = cells.material_id[c_idx];
            const auto& mp = model.material_table[mat_id];
            mhs::core::FieldContext ctx {mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], model.initial_temperature, 0.0};
            return mp.rho.eval(ctx);
        }

    } // anonymous namespace

    static void initCellHydroProperties(mhs::core::Model& model)
    {
        for (mhs::Index fi = 0; fi < model.fluid.n_fluid; ++fi) {
            mhs::Index old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
            mhs::Index ix, iy, iz;
            mhs::utils::decode_index(old_idx, model.mesh.ny, model.mesh.nz, ix, iy, iz);

            double dx = model.mesh.dx[ix], dy = model.mesh.dy[iy], dz = model.mesh.dz[iz];
            double K_perm = std::pow(model.fluid.hydraulic_diameter[fi], 2.0)
                / (2.0 * utils::f_re_rectangular(model.fluid.channel_width[fi], model.fluid.channel_height[fi]));

            double coef = K_perm / model.fluid.dynamic_viscosity[fi];
            model.fluid.hydroC[0][fi] = coef * (dy * dz / dx);
            model.fluid.hydroC[1][fi] = coef * (dx * dz / dy);
            model.fluid.hydroC[2][fi] = coef * (dx * dy / dz);
        }
    }

    static bool solvePressure(mhs::core::Model& model)
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;

        std::vector<Eigen::Triplet<double>> triplets;
        assert(model.fluid.n_fluid <= static_cast<mhs::Index>(std::numeric_limits<Eigen::Index>::max()));
        const auto eigen_n_fluid = static_cast<Eigen::Index>(model.fluid.n_fluid);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(eigen_n_fluid);
        triplets.reserve(static_cast<std::size_t>(model.fluid.n_fluid) * 7);

        for (mhs::Index fi = 0; fi < model.fluid.n_fluid; ++fi) {
            mhs::Index old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
            mhs::Index ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            double diagSum = 0.0;
            for (auto dir : mhs::core::FACE_DIRS) {
                mhs::Index n_old
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (n_old == mhs::invalidIndex)
                    continue;

                mhs::Index n_c = cells.index_map[n_old];
                assert(n_c != mhs::invalidIndex);
                mhs::Index fn = (n_c < static_cast<mhs::Index>(model.fluid.global_to_fluid.size()))
                    ? model.fluid.global_to_fluid[n_c]
                    : mhs::invalidIndex;
                if (fn == mhs::invalidIndex)
                    continue;

                int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];
                const auto& hc = model.fluid.hydroC[axis];
                double C_eff = mhs::utils::harmonicAverage(hc[fi], hc[fn]);
                diagSum += C_eff;

                if (model.fluid.fluid_bcs[fi].kind != mhs::core::FluidBCType::PressureType) {
                    triplets.emplace_back(static_cast<Eigen::Index>(fi), static_cast<Eigen::Index>(fn), -C_eff);
                }
            }

            triplets.emplace_back(static_cast<Eigen::Index>(fi), static_cast<Eigen::Index>(fi), diagSum);
            if (model.fluid.fluid_bcs[fi].kind == mhs::core::FluidBCType::PressureType) {
                const auto& params = model.fluid.fluid_bc_params.pressure;
                rhs(static_cast<Eigen::Index>(fi)) = params[model.fluid.fluid_bcs[fi].param_idx] * diagSum;
            }
            else if (model.fluid.fluid_bcs[fi].kind == mhs::core::FluidBCType::MassFlowRateType) {
                const double mdot_cell
                    = model.fluid.fluid_bc_params.mass_flow_rate[model.fluid.fluid_bcs[fi].param_idx];
                const double rho_cell = evaluateFluidRhoAtInitT(model, fi);
                rhs(static_cast<Eigen::Index>(fi)) = (rho_cell > mhs::core::zero_guard) ? (mdot_cell / rho_cell) : 0.0;
            }
            else if (model.fluid.fluid_bcs[fi].kind == mhs::core::FluidBCType::VelocityType) {
                const double vel = model.fluid.fluid_bc_params.velocity[model.fluid.fluid_bcs[fi].param_idx];
                rhs(static_cast<Eigen::Index>(fi)) = vel * model.fluid.fluid_face_area[fi];
            }
        }

        Eigen::SparseMatrix<double> A(eigen_n_fluid, eigen_n_fluid);
        A.setFromTriplets(triplets.begin(), triplets.end());

        auto solver = mhs::sim::LinearSolver::create();
        solver->compute(A);
        Eigen::VectorXd x = solver->solve(rhs);
        if (!solver->success()) {
            MHS_LOG_WARN(
                "Fluid pressure solve failed (nf={}, nz={})", model.fluid.n_fluid, static_cast<int>(A.nonZeros()));
            return false;
        }
        model.fluid.pressure = std::vector<double>(x.data(), x.data() + x.size());
        return true;
    }

    static void precomputeFlowAxes(mhs::core::Model& model)
    {
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;

        for (mhs::Index fi = 0; fi < model.fluid.n_fluid; ++fi) {
            mhs::Index old_idx = compact_to_old[model.fluid.fluid_to_global[fi]];
            mhs::Index ix, iy, iz;
            mhs::utils::decode_index(old_idx, mesh.ny, mesh.nz, ix, iy, iz);

            double maxFlux = -1.0;
            int bestAxis = 0;

            for (auto dir : mhs::core::FACE_DIRS) {
                mhs::Index n_old
                    = mhs::utils::neighbor_grid_index(ix, iy, iz, dir, mesh.nx, mesh.ny, mesh.nz, cells.index_map);
                if (n_old == mhs::invalidIndex)
                    continue;

                mhs::Index n_c = cells.index_map[n_old];
                assert(n_c != mhs::invalidIndex);
                mhs::Index fn = (n_c < static_cast<mhs::Index>(model.fluid.global_to_fluid.size()))
                    ? model.fluid.global_to_fluid[n_c]
                    : mhs::invalidIndex;
                if (fn == mhs::invalidIndex)
                    continue;

                int axis = mhs::utils::AXIS_OF_DIR[static_cast<size_t>(dir)];
                const auto& hc = model.fluid.hydroC[axis];
                double dp = std::fabs(model.fluid.pressure[fi] - model.fluid.pressure[fn]);
                double flux = dp * mhs::utils::harmonicAverage(hc[fi], hc[fn]);

                if (flux > maxFlux) {
                    maxFlux = flux;
                    bestAxis = axis;
                }
            }
            model.fluid.flow_axes[fi] = static_cast<int8_t>(bestAxis);
        }
    }

    void applyFluidOverlay(mhs::core::Model& model, const std::optional<mhs::core::FluidOverlay>& overlay,
        const mhs::core::IOStructure& ioStructure, const mhs::core::SymbolTable& symbols)
    {
        if (!overlay.has_value() || overlay->fluid_materials.empty())
            return;

        const double si_scale = mhs::utils::length_unit_to_si(ioStructure.length_unit);
        const mhs::Index N = static_cast<mhs::Index>(model.cells.material_id.size());

        buildCompactToOld(model.cells, model.mesh.nx * model.mesh.ny * model.mesh.nz);

        std::unordered_map<std::string, uint16_t> matNameToTableIdx;
        for (const auto& layer : ioStructure.layers) {
            for (const auto& block : layer.blocks) {
                matNameToTableIdx.emplace(block.material_name, static_cast<uint16_t>(matNameToTableIdx.size()));
            }
        }
        std::vector<std::string> matNamesByTableIdx(matNameToTableIdx.size());
        for (const auto& [name, idx] : matNameToTableIdx) {
            matNamesByTableIdx[idx] = name;
        }

        std::unordered_map<std::string, std::string> fluidViscosityMap;
        for (const auto& fm : overlay->fluid_materials)
            fluidViscosityMap[fm.name] = fm.dynamic_viscosity;

        model.fluid.is_fluid.assign(N, 0);
        std::vector<double> visc_temp(N, 0.0);

        for (uint16_t matIdx = 0; matIdx < static_cast<uint16_t>(model.material_table.size()); ++matIdx) {
            auto visIt = fluidViscosityMap.find(matNamesByTableIdx[matIdx]);
            if (visIt != fluidViscosityMap.end() && !visIt->second.empty()) {
                model.material_table[matIdx].is_fluid = true;
                model.material_table[matIdx].dynamic_viscosity
                    = mhs::core::parse(substitute_function_args(visIt->second, "T", ioStructure.functions), symbols);
            }
        }

        for (mhs::Index c = 0; c < N; ++c) {
            uint16_t matIdx = model.cells.material_id[c];
            if (matIdx < model.material_table.size() && model.material_table[matIdx].is_fluid) {
                model.fluid.is_fluid[c] = 1;
                visc_temp[c]
                    = model.material_table[matIdx].dynamic_viscosity.eval({0, 0, 0, model.initial_temperature, 0});
            }
        }

        model.fluid.fluid_to_global.clear();
        model.fluid.global_to_fluid.assign(N, mhs::invalidIndex);
        for (mhs::Index c = 0; c < N; ++c) {
            if (model.fluid.is_fluid[c]) {
                model.fluid.global_to_fluid[c] = static_cast<mhs::Index>(model.fluid.fluid_to_global.size());
                model.fluid.fluid_to_global.push_back(c);
            }
        }
        model.fluid.n_fluid = static_cast<mhs::Index>(model.fluid.fluid_to_global.size());

        if (model.fluid.n_fluid == 0)
            return;

        model.fluid.dynamic_viscosity.assign(model.fluid.n_fluid, 0.0);
        model.fluid.pressure.assign(model.fluid.n_fluid, 0.0);
        model.fluid.flow_axes.assign(model.fluid.n_fluid, -1);
        for (auto& hc : model.fluid.hydroC)
            hc.assign(model.fluid.n_fluid, 0.0);
        model.fluid.hydraulic_diameter.assign(model.fluid.n_fluid, 0.0);
        model.fluid.channel_width.assign(model.fluid.n_fluid, 0.0);
        model.fluid.channel_height.assign(model.fluid.n_fluid, 0.0);

        model.fluid.fluid_bcs.assign(model.fluid.n_fluid, mhs::core::FluidCellBC {});
        model.fluid.fluid_bc_params = mhs::core::FluidBCParamTable {};
        model.fluid.fluid_face_area.assign(model.fluid.n_fluid, 0.0);
        model.fluid.boundary_temperature_fluid.assign(model.fluid.n_fluid, std::numeric_limits<double>::quiet_NaN());

        for (mhs::Index fi = 0; fi < model.fluid.n_fluid; ++fi) {
            model.fluid.dynamic_viscosity[fi] = visc_temp[model.fluid.fluid_to_global[fi]];
        }
        applyFluidBoundaries(model, overlay.value(), si_scale);
        computeChannelDimensions(model);
    }

    void solveFluidFlow(mhs::core::Model& model)
    {
        if (model.fluid.n_fluid == 0)
            return;
        initCellHydroProperties(model);
        if (!solvePressure(model))
            return;
        precomputeFlowAxes(model);
    }

} // namespace mhs::sim
