#include "extraction.hpp"

#include <Eigen/QR>
#include <Eigen/SVD>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mhs::app::erom {
    Matrix compress_snapshots(const core::Vector<Vector>& snapshots, double tolerance)
    {
        const auto n = snapshots.front().size();
        Matrix centered(n, static_cast<Eigen::Index>(snapshots.size()));
        Eigen::Index count = 0;
        for (const auto& response : snapshots) {
            Vector column = response.array() - response.mean();
            const double norm = column.norm();
            if (norm > std::numeric_limits<double>::epsilon() * response.norm())
                centered.col(count++) = column / norm;
        }
        centered.conservativeResize(Eigen::NoChange, count);
        Matrix basis(n, 0);
        if (count) {
            // Thin QR followed by a small SVD avoids forming a squared-condition
            // Gram matrix while keeping the tall snapshot SVD out of the SDK.
            Eigen::HouseholderQR<Matrix> qr(centered);
            const auto r = std::min(n, count);
            const Matrix triangular = qr.matrixQR().topRows(r).template triangularView<Eigen::Upper>();
            Eigen::JacobiSVD<Matrix> svd(triangular, Eigen::ComputeThinU);
            if (svd.info() != Eigen::Success)
                throw std::runtime_error("response SVD did not converge");
            const auto& values = svd.singularValues();
            Eigen::Index keep = 0;
            while (keep < values.size() && values[keep] >= tolerance * values[0])
                ++keep;
            Matrix left = Matrix::Zero(n, keep);
            left.topRows(r) = svd.matrixU().leftCols(keep);
            basis = qr.householderQ() * left;
        }
        Vector constant = Vector::Constant(n, 1 / std::sqrt(static_cast<double>(n)));
        for (int pass = 0; pass < 2; ++pass)
            constant -= basis * (basis.transpose() * constant);
        const double norm = constant.norm();
        if (norm > std::numeric_limits<double>::epsilon() * n) {
            const auto col = basis.cols();
            basis.conservativeResize(Eigen::NoChange, col + 1);
            basis.col(col) = constant / norm;
        }
        if (!basis.allFinite() || basis.cols() == 0
            || (basis.transpose() * basis - Matrix::Identity(basis.cols(), basis.cols())).cwiseAbs().maxCoeff() > 1e-8)
            throw std::runtime_error("invalid reduced basis orthogonality");
        return basis;
    }
} // namespace mhs::app::erom
