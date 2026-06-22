#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace mhs::sim {

    /// Final linear system: A * x = b
    struct LinearSystem {
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
    };

    struct AssemblyResult {
        Eigen::SparseMatrix<double> K;
        Eigen::VectorXd f;
        Eigen::VectorXd M_diag;
    };

} // namespace mhs::sim
