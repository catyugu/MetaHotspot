#pragma once

#include <Eigen/Sparse>

#include "common/internal_model.hpp"

namespace mhs::sim {

    struct LinearSystem {
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        Eigen::VectorXd residual;
    };

    class Assembler {
    public:
        explicit Assembler(const mhs::core::InternalModel& model) : model_(model) { }
        ~Assembler() = default;

        LinearSystem assemble(const mhs::core::GlobalState& state);

    private:
        const mhs::core::InternalModel& model_;
    };

} // namespace mhs::sim
