#include "api/internal.hpp"
#include "api/metahotspot.h"
#include "api/metahotspot_macromodel.h"

#include "macromodel/modal_port.hpp"
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

MHS_API mhs_status_t mhs_macromodel_solve(const mhs_compiled_t* c,
    const mhs_macro_port_model_t* macro, const double* state, size_t state_count,
    const mhs_solve_options_t* opts, mhs_solution_t** out)
{
    CHECK_NULL(c);
    CHECK_NULL(macro);
    CHECK_NULL(macro->model_cells);
    CHECK_NULL(macro->exterior_half_conductance);
    CHECK_NULL(macro->operators.rhs);
    CHECK_NULL(state);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_SOLVE, {
        const auto fvm_count = c->model.cells.cell_to_grid.size();
        const bool has_basis = (macro->basis != nullptr);

        // Validate dimensions
        if (macro->physical_port_count == 0) {
            SET_ERR("physical_port_count must be > 0");
            return MHS_ERR_INVALID_ARG;
        }
        if (has_basis) {
            // Basis present: mode count = operators.n
            if (macro->operators.n != macro->physical_port_count) {
                // With basis, operators.n is the macro state (mode) count.
                // No constraint relating operators.n to physical_port_count
                // beyond macro_state_count = operators.n.
            }
        }
        else {
            // Unit basis: macro_state_count == physical_port_count
            if (macro->operators.n != macro->physical_port_count) {
                SET_ERR("unit-basis macro: operators.n must equal physical_port_count");
                return MHS_ERR_INVALID_ARG;
            }
        }

        const auto macro_state_count = macro->operators.n;
        if (state_count != fvm_count + macro_state_count) {
            SET_ERR("state_count must equal cell_count + macro_state_count");
            return MHS_ERR_INVALID_ARG;
        }
        if (state_count == 0) {
            SET_ERR("state vector is empty");
            return MHS_ERR_INVALID_ARG;
        }

        // Build PortModel
        mhs::macro::PortModel port_model;
        port_model.operators.K = _csc_view_to_eigen(macro->operators.K);
        port_model.operators.C = _csc_view_to_eigen(macro->operators.C);
        port_model.operators.f = Eigen::Map<const Eigen::VectorXd>(
            macro->operators.rhs, static_cast<Eigen::Index>(macro_state_count));
        port_model.physical_port_count = macro->physical_port_count;
        if (has_basis) {
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> basis_map(
                macro->basis, static_cast<Eigen::Index>(macro->physical_port_count),
                static_cast<Eigen::Index>(macro_state_count));
            port_model.basis = basis_map;
        }
        // else: basis stays empty (rows=0, cols=0) → unit basis

        // Build PortCoupling
        mhs::macro::PortCoupling coupling;
        coupling.model_cells.assign(
            macro->model_cells, macro->model_cells + macro->physical_port_count);
        coupling.model_face = _to_face(macro->model_face);
        coupling.exterior_half_conductance = Eigen::Map<const Eigen::VectorXd>(
            macro->exterior_half_conductance,
            static_cast<Eigen::Index>(macro->physical_port_count));

        // Reconstruct options and solve
        auto so = to_solve_options(opts, c->model.transient_duration);

        auto result = mhs::macro::solve(
            c->model, port_model, coupling,
            std::span<const double>(state, state_count), so);

        auto* s = new (std::nothrow) mhs_solution_t;
        if (!s) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERR_OOM;
        }
        s->sol = std::move(result);
        s->sol.fvm_count = fvm_count;
        *out = s;
    });
}
