#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>

namespace mhs::sim {

    /// Runtime state used to evaluate state-dependent operators.
    struct AssembleContext {
        const std::vector<double>& state;
        double current_time = 0.0;
    };

    /// Operators for C * dx/dt + K * x = f.
    struct AssemblyResult {
        Eigen::SparseMatrix<double> K;
        Eigen::SparseMatrix<double> C;
        Eigen::VectorXd f;
    };

    struct SourceEntry {
        Eigen::Index row = 0;
        double value = 0.0;
    };

    /// Sparse entries emitted by one independent physical contribution.
    struct OperatorContribution {
        std::vector<Eigen::Triplet<double>> stiffness;
        std::vector<Eigen::Triplet<double>> capacity;
        std::vector<SourceEntry> source;
    };

    /// Collect independent contributions and construct each global operator once.
    class OperatorAccumulator {
    public:
        explicit OperatorAccumulator(Eigen::Index state_count);

        void add(OperatorContribution contribution);
        AssemblyResult finish() &&;

    private:
        Eigen::Index state_count_ = 0;
        std::vector<Eigen::Triplet<double>> stiffness_;
        std::vector<Eigen::Triplet<double>> capacity_;
        Eigen::VectorXd source_;
    };

} // namespace mhs::sim
