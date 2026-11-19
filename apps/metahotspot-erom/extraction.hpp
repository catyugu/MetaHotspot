#pragma once

#include "erom/extraction.hpp"
#include "model/fom.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace mhs::app::erom {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    struct SurfaceData {
        model::SurfaceId id;
        std::string name;
        std::optional<model::EromSurfaceTraining> training;
        core::Vector<model::Facet> facets;
        core::Vector<model::CellId> cells;
        core::Vector<double> resistance;
    };

    // All conductance vectors use compact indices into the distinct boundary cells.
    struct HtcOperators {
        explicit HtcOperators(const core::Vector<SurfaceData>& surfaces);

        void add_conductance(Eigen::Ref<Vector> diagonal, std::size_t group, double h) const;
        Vector diagonal(const Vector& h) const;
        Vector parameters(double model::EromSurfaceTraining::* field) const;

        core::Vector<const SurfaceData*> groups;
        core::Vector<Eigen::Index> cells;
        core::Vector<core::Vector<Eigen::Index>> facet_rows;
    };

    struct Prepared {
        Sparse K;
        Vector capacity;
        Matrix B;
        Sparse L;
        core::Vector<SurfaceData> surfaces;
        core::Vector<model::EromSource> sources;
        core::Vector<model::EromOutput> outputs;
        double reference;
    };

    Prepared prepare(const model::Fom& fom, const HtcRanges& ranges);

    // Application-local numerical kernels; no extraction implementation enters the SDK.
    Sparse add_diagonal(const Sparse& matrix, const Vector& diagonal);
    Vector solve_positive(const Matrix& matrix, const Vector& rhs);
    Matrix solve_positive(const Matrix& matrix, const Matrix& rhs);
    Matrix build_basis(const Prepared& project, const HtcOperators& htc, const core::Vector<int>& regions,
        double tolerance, Statistics& statistics);
} // namespace mhs::app::erom
