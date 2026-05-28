#pragma once

#include "model/internal_model.hpp"
#include <Eigen/Sparse>
#include <vector>

namespace mhs {

struct AssemblerResult {
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
};

class Assembler {
public:
    Assembler() = default;
    ~Assembler() = default;

    AssemblerResult assemble(const model::InternalModel& model,
                            const std::vector<double>& T,
                            double t);

    void setTime(double t) { t_ = t; }

private:
    double t_ = 0.0;

    double get_distance(int di, int dj, int dk,
                        const model::MeshGeometry& mesh,
                        int i, int j, int k);
};

} // namespace mhs