#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace mhs::sim {
    /// Final linear system: A * x = b
    struct LinearSystem {
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
    };
} // namespace mhs::sim
