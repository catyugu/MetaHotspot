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
        Vector capacity, fixed_diagonal, fixed_load;
        core::Vector<double> output_offset;
        Matrix B;
        struct HtcGroup {
            model::EromBoundaryGroup boundary_group;
            Eigen::Index begin, end;
        };
        double reference_temperature;
        model::Vec3 origin;
        core::Vector<HtcGroup> training;
        core::Vector<model::EromInterface> interfaces;
        core::Vector<model::CellId> boundary_cells; // one per interface row
        core::Vector<model::EromOutline> outline;
        core::Vector<model::EromSource> sources;
        core::Vector<model::EromOutput> outputs;

        void set_conductances(Eigen::Ref<Vector> values, const Vector& h) const;
        void set_default_conductances(Eigen::Ref<Vector> values) const;
        void add_boundary(Eigen::Ref<Vector> cells, Eigen::Ref<const Vector> facets) const;
    };

    Prepared prepare(const model::Fom& fom, const model::FomDefinition& definition, const HtcRanges& ranges);

    // Application-local numerical kernels; no extraction implementation enters the SDK.
    Sparse add_diagonal(const Sparse& matrix, const Vector& diagonal);
    Vector solve_positive(const Matrix& matrix, const Vector& rhs);
    Matrix build_basis(const Prepared& project, const Vector& lower, const Vector& upper,
        const core::Vector<int>& regions, double tolerance, Statistics& statistics);
} // namespace mhs::app::erom
