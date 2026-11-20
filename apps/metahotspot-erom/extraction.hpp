#pragma once

#include "erom/extraction.hpp"
#include "model/fom.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace mhs::app::erom {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    struct Prepared {
        Sparse K, L;
        Vector capacity;
        model::Erom asset;
        core::Vector<Eigen::Index> boundary_offsets {0}; // interface rows grouped by HTC group
        core::Vector<model::CellId> boundary_cells; // one per interface row

        // Each source uses the same cell weights for distributing power and averaging temperature.
        auto source_weights() const { return L.bottomRows(static_cast<Eigen::Index>(asset.sources.size())); }
        void add_convection(Vector& diagonal, Vector& conductances, const Vector& h) const;
    };

    Prepared prepare(const model::Fom& fom, const model::FomDefinition& definition, const HtcRanges& ranges);

    // Application-local numerical kernels; no extraction implementation enters the SDK.
    Sparse add_diagonal(const Sparse& matrix, const Vector& diagonal);
    Vector solve_positive(const Matrix& matrix, const Vector& rhs);
    Matrix build_basis(const Prepared& project, const Vector& lower, const Vector& upper,
        const core::Vector<int>& regions, double tolerance, Statistics& statistics);
} // namespace mhs::app::erom
