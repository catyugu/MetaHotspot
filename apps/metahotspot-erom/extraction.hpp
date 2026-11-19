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
        Matrix B;
        model::Erom asset;
        core::Vector<model::CellId> boundary_cells; // indexed by connectable FacetId
        // Conductances are indexed by the same connectable FacetIds.
        void set_conductances(Eigen::Ref<Vector> values, const Vector& h) const;
        core::Vector<core::Vector<Eigen::Index>> training_facets;
        Vector parameters(double model::EromSurface::* field) const;
        void add_boundary(Eigen::Ref<Vector> cells, Eigen::Ref<const Vector> facets) const;
    };

    Prepared prepare(const model::Fom& fom, const model::FomDefinition& definition, const HtcRanges& ranges);

    // Application-local numerical kernels; no extraction implementation enters the SDK.
    Sparse add_diagonal(const Sparse& matrix, const Vector& diagonal);
    Vector solve_positive(const Matrix& matrix, const Vector& rhs);
    Matrix solve_positive(const Matrix& matrix, const Matrix& rhs);
    Matrix build_basis(
        const Prepared& project, const core::Vector<int>& regions, double tolerance, Statistics& statistics);
} // namespace mhs::app::erom
