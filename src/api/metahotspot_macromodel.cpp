#include "api/metahotspot_macromodel.h"
#include "api/internal.h"

#include "macromodel/modal_port.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

struct mhs_macro_port_map_t {
    const mhs_compiled_t* owner = nullptr;
    mhs::macro::PortMap map;
};

namespace {

    mhs::core::FaceDir to_face(mhs_face_t face)
    {
        switch (face) {
        case MHS_FACE_XM:
            return mhs::core::FaceDir::XM;
        case MHS_FACE_XP:
            return mhs::core::FaceDir::XP;
        case MHS_FACE_YM:
            return mhs::core::FaceDir::YM;
        case MHS_FACE_YP:
            return mhs::core::FaceDir::YP;
        case MHS_FACE_ZM:
            return mhs::core::FaceDir::ZM;
        case MHS_FACE_ZP:
            return mhs::core::FaceDir::ZP;
        default:
            throw std::invalid_argument("invalid face value: " + std::to_string(face));
        }
    }

    Eigen::SparseMatrix<double> csc_to_eigen(const mhs_csc_view_t& view)
    {
        if (view.rows < 0 || view.columns < 0 || view.nnz < 0 || !view.outer_indices
            || (view.nnz > 0 && (!view.inner_indices || !view.values))) {
            throw std::invalid_argument("invalid CSC matrix view");
        }
        if (view.outer_indices[0] != 0 || view.outer_indices[view.columns] != view.nnz)
            throw std::invalid_argument("invalid CSC outer-index range");
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(static_cast<std::size_t>(view.nnz));
        for (int32_t column = 0; column < view.columns; ++column) {
            const int32_t begin = view.outer_indices[column];
            const int32_t end = view.outer_indices[column + 1];
            if (begin < 0 || end < begin || end > view.nnz)
                throw std::invalid_argument("invalid CSC column offsets");
            for (int32_t entry = begin; entry < end; ++entry) {
                const int32_t row = view.inner_indices[entry];
                if (row < 0 || row >= view.rows)
                    throw std::invalid_argument("invalid CSC row index");
                entries.emplace_back(row, column, view.values[entry]);
            }
        }
        Eigen::SparseMatrix<double> result(view.rows, view.columns);
        result.setFromTriplets(entries.begin(), entries.end());
        return result;
    }

    void eigen_to_csc_view(const Eigen::SparseMatrix<double>& matrix, mhs_csc_view_t* out)
    {
        out->rows = static_cast<int32_t>(matrix.rows());
        out->columns = static_cast<int32_t>(matrix.cols());
        out->nnz = static_cast<int32_t>(matrix.nonZeros());
        out->outer_indices = matrix.outerIndexPtr();
        out->inner_indices = matrix.innerIndexPtr();
        out->values = matrix.valuePtr();
    }

    void operators_to_view(const mhs::sim::Operators& operators, mhs_operators_t* out)
    {
        eigen_to_csc_view(operators.K, &out->K);
        eigen_to_csc_view(operators.C, &out->C);
        out->rhs = operators.f.data();
        out->n = static_cast<size_t>(operators.f.size());
    }

    void validate_owner(const mhs_compiled_t* compiled, const mhs_macro_port_map_t* ports)
    {
        if (ports->owner != compiled)
            throw std::invalid_argument("port map belongs to a different compiled model");
    }

} // namespace

MHS_API mhs_status_t mhs_macromodel_port_map_create(const mhs_compiled_t* compiled,
    const mhs_macro_port_patch_t* patches, size_t patch_count, mhs_macro_port_map_t** out)
{
    CHECK_NULL(compiled);
    CHECK_NULL(patches);
    CHECK_NULL(out);
    *out = nullptr;
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        if (patch_count == 0)
            throw std::invalid_argument("patch_count must be > 0");
        std::vector<mhs::macro::PortPatch> cpp_patches;
        cpp_patches.reserve(patch_count);
        for (size_t i = 0; i < patch_count; ++i) {
            const auto& patch = patches[i];
            cpp_patches.push_back({to_face(patch.face), patch.coordinate, patch.rectangle.a_min, patch.rectangle.a_max,
                patch.rectangle.b_min, patch.rectangle.b_max});
        }
        const auto compiled_map
            = mhs::macro::compile_port_map(compiled->model, std::span<const mhs::macro::PortPatch>(cpp_patches));
        auto* result = new (std::nothrow) mhs_macro_port_map_t;
        if (!result)
            throw std::bad_alloc();
        result->owner = compiled;
        result->map = compiled_map;
        *out = result;
    });
}

MHS_API void mhs_macromodel_port_map_destroy(mhs_macro_port_map_t* map) { delete map; }

MHS_API size_t mhs_macromodel_port_count(const mhs_macro_port_map_t* map) { return map ? map->map.port_count : 0; }

MHS_API mhs_status_t mhs_macromodel_assemble_dtn(const mhs_compiled_t* compiled, const mhs_macro_port_map_t* ports,
    const double* state, size_t state_count, double time, mhs_operators_t* out)
{
    CHECK_NULL(compiled);
    CHECK_NULL(ports);
    CHECK_NULL(state);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_ASSEMBLE, {
        validate_owner(compiled, ports);
        const auto cell_count = compiled->model.cells.cell_to_grid.size();
        if (state_count != cell_count)
            throw std::invalid_argument("state_count must equal compiled cell count");
        auto& scratch = const_cast<mhs_compiled_t*>(compiled)->assemble_scratch;
        scratch
            = mhs::macro::assemble_dtn(compiled->model, ports->map, std::span<const double>(state, state_count), time);
        operators_to_view(scratch, out);
    });
}

MHS_API mhs_status_t mhs_macromodel_solve(const mhs_compiled_t* compiled, const mhs_macro_port_map_t* ports,
    const mhs_macro_dtn_model_t* dtn, const double* state, size_t state_count, const mhs_solve_options_t* opts,
    mhs_solution_t** out)
{
    CHECK_NULL(compiled);
    CHECK_NULL(ports);
    CHECK_NULL(dtn);
    CHECK_NULL(dtn->operators.rhs);
    CHECK_NULL(state);
    CHECK_NULL(out);
    *out = nullptr;
    MHS_TRY(MHS_ERR_SOLVE, {
        validate_owner(compiled, ports);
        const auto fvm_count = compiled->model.cells.cell_to_grid.size();
        const auto dtn_state_count = dtn->operators.n;
        if (dtn_state_count < ports->map.port_count)
            throw std::invalid_argument("DtN states must begin with one state per physical port");
        if (state_count != fvm_count + dtn_state_count)
            throw std::invalid_argument("state_count must equal cell_count + DtN state count");

        mhs::macro::DtNModel model;
        model.operators.K = csc_to_eigen(dtn->operators.K);
        model.operators.C = csc_to_eigen(dtn->operators.C);
        model.operators.f
            = Eigen::Map<const Eigen::VectorXd>(dtn->operators.rhs, static_cast<Eigen::Index>(dtn_state_count));

        auto result = mhs::macro::solve(compiled->model, model, ports->map, std::span<const double>(state, state_count),
            to_solve_options(opts, compiled->model.transient_duration));
        auto* solution = new (std::nothrow) mhs_solution_t;
        if (!solution)
            throw std::bad_alloc();
        solution->sol = std::move(result);
        solution->sol.fvm_count = fvm_count;
        *out = solution;
    });
}
