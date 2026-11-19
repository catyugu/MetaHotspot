#pragma once

#include "model/erom.hpp"
#include "model/fom.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <map>
#include <set>

namespace mhs::app::erom {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    struct Selection {
        std::map<model::SurfaceId, model::EromSurfaceTraining> htc;
        std::set<std::string> probes;
        bool all_probes = true;
        bool source_temperatures = true;
    };

    struct Options {
        double tolerance = 1e-3;
        int max_order = 512;
        double output_error = 0.02;
    };

    struct SurfaceData {
        model::SurfaceId id;
        std::string name;
        std::optional<model::EromSurfaceTraining> training;
        core::Vector<model::Facet> facets;
        core::Vector<model::CellId> cells;
        core::Vector<double> resistance;
    };

    struct Prepared {
        Sparse K;
        Vector capacity;
        Matrix B;
        Sparse L;
        core::Vector<SurfaceData> surfaces;
        core::Vector<model::EromSource> sources;
        core::Vector<model::EromOutput> outputs;
        double reference, initial;
    };

    struct Statistics {
        int shifts = 0;
        int response_solves = 0;
        int response_amg_setups = 0;
        std::size_t cg_iterations = 0;
        double lambda_min = 0, lambda_max = 0;
        double output_error = 0;
    };

    struct Extraction {
        model::Erom asset;
        Statistics statistics;
    };

    Prepared prepare(const model::Fom& fom, double initial_temperature, Selection selection);
    Extraction extract(const Prepared& project, const Options& options);
    void write_erom(const std::filesystem::path& path, const model::Erom& asset);

    // Application-local numerical kernels; no extraction implementation enters the SDK.
    Sparse add_diagonal(const Sparse& matrix, const Vector& diagonal);
    Vector solve_positive(const Matrix& matrix, const Vector& rhs);
    Matrix solve_positive(const Matrix& matrix, const Matrix& rhs);
    Matrix build_basis(const Sparse& K, const Vector& capacity, const Matrix& slopes,
        const core::Vector<std::pair<double, double>>& ranges, const Matrix& inputs, const Options& options,
        Statistics& statistics);
} // namespace mhs::app::erom
