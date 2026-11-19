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
    };

    // Conductance vectors use compact indices into the distinct boundary cells.
    struct HtcOperators {
        HtcOperators(const Prepared& project, std::size_t group_count);

        void add_conductance(Eigen::Ref<Vector> diagonal, std::size_t group, double h) const;
        Vector diagonal(const Vector& h) const;
        Vector parameters(double model::EromSurfaceTraining::* field) const;

        const model::Erom& asset;
        std::size_t group_count;
        core::Vector<Eigen::Index> cells;
        core::Vector<Eigen::Index> facet_rows;
    };

    Prepared prepare(const model::Fom& fom, const model::FomDefinition& definition, const HtcRanges& ranges);

    // Application-local numerical kernels; no extraction implementation enters the SDK.
    Sparse add_diagonal(const Sparse& matrix, const Vector& diagonal);
    Vector solve_positive(const Matrix& matrix, const Vector& rhs);
    Matrix solve_positive(const Matrix& matrix, const Matrix& rhs);
    Matrix build_basis(const Prepared& project, const HtcOperators& htc, const core::Vector<int>& regions,
        double tolerance, Statistics& statistics);
} // namespace mhs::app::erom
