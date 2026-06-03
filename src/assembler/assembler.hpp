#pragma once

#include "common/internal_model.hpp"
#include <Eigen/Sparse>

namespace mhs::assembler {

    struct LinearSystem {
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        Eigen::VectorXd residual;
    };

    class Assembler {
    public:
        explicit Assembler(const InternalModel& model) : model_(model) { }
        ~Assembler() = default;

        LinearSystem assemble(const GlobalState& state);

    private:
        const InternalModel& model_;
    };

} // namespace mhs::assembler