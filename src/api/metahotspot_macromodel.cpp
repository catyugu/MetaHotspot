#include "api/internal.hpp"
#include "api/metahotspot.h"

#include "macromodel/modal_port.hpp"
#include "solver/scheduler.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <span>

/* ------------------------------------------------------------------ */
/*  Enum conversions                                                  */
/* ------------------------------------------------------------------ */

static mhs::core::FaceDir _to_face(mhs_face_t face)
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

/* ------------------------------------------------------------------ */
/*  CSC helpers                                                       */
/* ------------------------------------------------------------------ */

static Eigen::SparseMatrix<double> _csc_view_to_eigen(const mhs_csc_view_t& view)
{
    if (view.rows < 0 || view.columns < 0 || view.nnz < 0 || !view.outer_indices
        || (view.nnz > 0 && (!view.inner_indices || !view.values))) {
        throw std::invalid_argument("invalid CSC matrix view");
    }
    if (view.outer_indices[0] != 0 || view.outer_indices[view.columns] != view.nnz) {
        throw std::invalid_argument("invalid CSC outer-index range");
    }

    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(static_cast<std::size_t>(view.nnz));
    for (int32_t column = 0; column < view.columns; ++column) {
        const int32_t begin = view.outer_indices[column];
        const int32_t end = view.outer_indices[column + 1];
        if (begin < 0 || end < begin || end > view.nnz) {
            throw std::invalid_argument("invalid CSC column offsets");
        }
        for (int32_t entry = begin; entry < end; ++entry) {
            const int32_t row = view.inner_indices[entry];
            if (row < 0 || row >= view.rows) {
                throw std::invalid_argument("invalid CSC row index");
            }
            entries.emplace_back(row, column, view.values[entry]);
        }
    }

    Eigen::SparseMatrix<double> result(view.rows, view.columns);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_solve_modal_port(const mhs_compiled_t* c,
    const mhs_modal_port_view_t* macro, const double* state, size_t state_count,
    const mhs_solver_opts_t* opts, mhs_solution_t** out)
{
    CHECK_NULL(c);
    CHECK_NULL(macro);
    CHECK_NULL(macro->basis);
    CHECK_NULL(macro->model_cells);
    CHECK_NULL(macro->exterior_half_conductance);
    CHECK_NULL(macro->operators.rhs);
    CHECK_NULL(state);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_SOLVE, {
        if (macro->physical_port_count == 0 || macro->mode_count == 0
            || macro->operators.n != macro->mode_count) {
            SET_ERR("invalid modal port dimensions");
            return MHS_ERR_INVALID_ARG;
        }
        const auto fvm_count = c->model.cells.cell_to_grid.size();
        if (state_count != fvm_count + macro->mode_count) {
            SET_ERR("state_count must equal cell_count + mode_count");
            return MHS_ERR_INVALID_ARG;
        }
        if (state_count == 0) {
            SET_ERR("state vector is empty");
            return MHS_ERR_INVALID_ARG;
        }

        mhs::sim::ModalPort modal_port;
        modal_port.operators.K = _csc_view_to_eigen(macro->operators.K);
        modal_port.operators.C = _csc_view_to_eigen(macro->operators.C);
        modal_port.operators.f = Eigen::Map<const Eigen::VectorXd>(
            macro->operators.rhs, static_cast<Eigen::Index>(macro->mode_count));
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> basis_map(
            macro->basis, static_cast<Eigen::Index>(macro->physical_port_count),
            static_cast<Eigen::Index>(macro->mode_count));
        modal_port.basis = basis_map;

        mhs::sim::ThermalPortInterface interface;
        interface.model_cells.assign(
            macro->model_cells, macro->model_cells + macro->physical_port_count);
        interface.model_face = _to_face(macro->model_face);
        interface.exterior_half_conductance = Eigen::Map<const Eigen::VectorXd>(
            macro->exterior_half_conductance,
            static_cast<Eigen::Index>(macro->physical_port_count));

        mhs::sim::Study study {
            c->model.study_type,
            c->model.transient_duration,
            c->model.transient_time_step,
        };

        // Reconstruct SolverOpts from the C-level opts struct using the shared helper.
        auto so = to_solver_opts(opts, c->model.transient_duration);

        mhs::sim::SystemAssembler assemble = [&](std::span<const double> current_state, double time) {
            return mhs::sim::assemble_modal_port_system(
                c->model, modal_port, interface, current_state, time);
        };
        auto result = mhs::sim::solve_system(
            study, assemble, std::span<const double>(state, state_count), so);

        auto* s = new (std::nothrow) mhs_solution_t;
        if (!s) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERR_OOM;
        }
        s->result = std::move(result);
        s->fvm_count = fvm_count;
        *out = s;
    });
}
