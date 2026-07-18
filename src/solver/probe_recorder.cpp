#include "runtime/mesh.hpp"
#include "solver/interpolation.hpp"
#include "solver/probe_recorder.hpp"

#include <limits>

namespace mhs::sim {

    namespace {
        template <typename T>
        mhs::core::Index locate_cell_index(const std::vector<T>& centers, const std::vector<T>& sizes, T value)
        {
            mhs::core::Index n = static_cast<mhs::core::Index>(centers.size());
            if (n == 0)
                return mhs::core::invalidIndex;
            T lo_bound = centers[0] - sizes[0] * T(0.5);
            T hi_bound = centers[n - 1] + sizes[n - 1] * T(0.5);
            if (value < lo_bound || value > hi_bound)
                return mhs::core::invalidIndex;

            mhs::core::Index lo = 0, hi = n - 1;
            while (lo < hi) {
                mhs::core::Index mid = (lo + hi + 1) / 2;
                if (centers[mid] <= value)
                    lo = mid;
                else
                    hi = mid - 1;
            }
            T half = sizes[lo] * T(0.5);
            if (value >= centers[lo] + half && lo + 1 < n)
                return lo + 1;
            if (value >= centers[lo] - half)
                return lo;
            return lo;
        }
    } // namespace

    void ProbeRecorder::initialize(const mhs::core::Model& model)
    {
        model_ = &model;
        traces_.clear();
        slots_.clear();
        traces_.reserve(model.observation_points.size());
        slots_.reserve(model.observation_points.size());

        for (const auto& op : model.observation_points) {
            mhs::core::ProbeTrace t;
            t.name = op.name;
            traces_.push_back(std::move(t));

            ProbeSlot slot;
            slot.px = op.x;
            slot.py = op.y;
            slot.pz = op.z;
            slot.ix = locate_cell_index(model.mesh.cx, model.mesh.dx, op.x);
            slot.iy = locate_cell_index(model.mesh.cy, model.mesh.dy, op.y);
            slot.iz = locate_cell_index(model.mesh.cz, model.mesh.dz, op.z);
            if (slot.ix == mhs::core::invalidIndex || slot.iy == mhs::core::invalidIndex
                || slot.iz == mhs::core::invalidIndex) {
                slot.valid = false;
            }
            else {
                slot.grid_idx = slot.ix * model.mesh.ny * model.mesh.nz + slot.iy * model.mesh.nz + slot.iz;
                slot.valid = (model.cells.grid_to_cell[slot.grid_idx] != mhs::core::invalidIndex);
            }
            slots_.push_back(std::move(slot));
        }
    }

    void ProbeRecorder::record(double time, const std::vector<double>& cell_T)
    {
        if (slots_.empty())
            return;

        for (size_t i = 0; i < slots_.size(); ++i) {
            const ProbeSlot& slot = slots_[i];
            double v = slot.valid ? sample_one(slot, cell_T, time) : std::numeric_limits<double>::quiet_NaN();
            traces_[i].times.push_back(time);
            traces_[i].values.push_back(v);
        }
    }

    double ProbeRecorder::sample_one(const ProbeSlot& slot, const std::vector<double>& cell_T, double time) const
    {
        const auto& mesh = model_->mesh;
        const auto& cells = model_->cells;
        const auto& face_bcs = model_->face_bcs;
        const auto& bc_params = model_->bc_params;
        const mhs::core::Index ix = slot.ix;
        const mhs::core::Index iy = slot.iy;
        const mhs::core::Index iz = slot.iz;
        const mhs::core::Index grid_idx = slot.grid_idx;
        const double px = slot.px;
        const double py = slot.py;
        const double pz = slot.pz;

        mhs::core::Index compact_idx = cells.grid_to_cell[grid_idx];
        assert(compact_idx != mhs::core::invalidIndex);

        std::vector<mhs::utils::SampleDataPoint> pts;
        pts.reserve(8 + mhs::core::FACE_COUNT);

        double sum_T = 0.0;
        mhs::core::Index cnt = 0;
        for (mhs::core::Index dx = 0; dx <= 1; ++dx) {
            for (mhs::core::Index dy = 0; dy <= 1; ++dy) {
                for (mhs::core::Index dz = 0; dz <= 1; ++dz) {
                    mhs::core::Index ngx = ix + dx;
                    mhs::core::Index ngy = iy + dy;
                    mhs::core::Index ngz = iz + dz;
                    if (ngx >= mesh.nx || ngy >= mesh.ny || ngz >= mesh.nz)
                        continue;
                    mhs::core::Index ng = ngx * mesh.ny * mesh.nz + ngy * mesh.nz + ngz;
                    if (cells.grid_to_cell[ng] == mhs::core::invalidIndex)
                        continue;
                    sum_T += cell_T[cells.grid_to_cell[ng]];
                    ++cnt;
                }
            }
        }
        if (cnt == 0)
            return std::numeric_limits<double>::quiet_NaN();
        const double T_c = sum_T / static_cast<double>(cnt);

        auto* fc = &face_bcs[compact_idx * mhs::core::FACE_COUNT];
        for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
            const auto& fb = fc[f];
            if (fb.type == mhs::core::BcType::FirstType
                && mhs::utils::is_grid_boundary_face(mhs::core::FACE_DIRS[f], ix, iy, iz, mesh)) {
                return bc_params.dirichlet_T[fb.param_idx].eval({px, py, pz, T_c, time});
            }
        }

        const auto& mp = model_->material_table[cells.material_id[compact_idx]];
        mhs::core::FieldContext ctx {px, py, pz, T_c, time};
        double kx_c = mp.kx.eval(ctx);
        double ky_c = mp.ky.eval(ctx);
        double kz_c = mp.kz.eval(ctx);

        for (mhs::core::Index dx = 0; dx <= 1; ++dx) {
            for (mhs::core::Index dy = 0; dy <= 1; ++dy) {
                for (mhs::core::Index dz = 0; dz <= 1; ++dz) {
                    mhs::core::Index ngx = ix + dx;
                    mhs::core::Index ngy = iy + dy;
                    mhs::core::Index ngz = iz + dz;
                    if (ngx >= mesh.nx || ngy >= mesh.ny || ngz >= mesh.nz)
                        continue;
                    mhs::core::Index ng = ngx * mesh.ny * mesh.nz + ngy * mesh.nz + ngz;
                    if (cells.grid_to_cell[ng] == mhs::core::invalidIndex)
                        continue;
                    double T_i = cell_T[cells.grid_to_cell[ng]];
                    double cdx = mesh.cx[ngx] - px;
                    double cdy = mesh.cy[ngy] - py;
                    double cdz = mesh.cz[ngz] - pz;
                    double dist_k = (cdx * cdx) / kx_c + (cdy * cdy) / ky_c + (cdz * cdz) / kz_c;
                    pts.push_back({mesh.cx[ngx], mesh.cy[ngy], mesh.cz[ngz], T_i, 1.0 / dist_k});
                }
            }
        }

        for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
            const auto& fb = fc[f];
            if (fb.type == mhs::core::BcType::None || fb.type == mhs::core::BcType::FirstType)
                continue;

            auto dir = mhs::core::FACE_DIRS[f];
            double k_face = mhs::utils::k_along(dir, kx_c, ky_c, kz_c);
            double T_f = mhs::utils::sample_extrapolate_face_temperature(
                dir, fb.type, fb.param_idx, T_c, k_face, mesh, ix, iy, iz, bc_params, time);

            double fx, fy, fz;
            mhs::utils::sample_face_center(dir, ix, iy, iz, mesh, fx, fy, fz);
            double fdx = fx - px;
            double fdy = fy - py;
            double fdz = fz - pz;
            double fdist_k = (fdx * fdx) / kx_c + (fdy * fdy) / ky_c + (fdz * fdz) / kz_c;
            pts.push_back({fx, fy, fz, T_f, 1.0 / fdist_k});
        }

        return mhs::utils::sample_solve_least_squares(pts, px, py, pz);
    }

} // namespace mhs::sim
