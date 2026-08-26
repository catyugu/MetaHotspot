#include "solver/probe_recorder.hpp"
#include "core/mesh.hpp"
#include "solver/interpolation.hpp"

#include <span>

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

    void ProbeRecorder::record(double time, std::span<const double> cell_T)
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

    double ProbeRecorder::sample_one(const ProbeSlot& slot, std::span<const double> cell_T, double time) const
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

        const double T_c = cell_T[compact_idx];

        auto* fc = &face_bcs[compact_idx * mhs::core::FACE_COUNT];
        for (size_t f = 0; f < mhs::core::FACE_COUNT; f++) {
            const auto& fb = fc[f];
            if (fb.type == mhs::core::BcType::FirstType
                && mhs::utils::is_grid_boundary_face(mhs::core::FACE_DIRS[f], ix, iy, iz, mesh)) {
                return bc_params.dirichlet_T[fb.param_idx].eval({px, py, pz, T_c, time});
            }
        }

        const auto gradient = mhs::utils::reconstruct_cell_gradient(*model_, cell_T, time, ix, iy, iz);
        return mhs::utils::extrapolate_cell_temperature(
            T_c, gradient, mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], px, py, pz);
    }

} // namespace mhs::sim
