#pragma once

#include <Eigen/Sparse>
#include <cstring>

namespace mhs::sim {

    /// Check whether two sparse matrices have the same nonzero pattern
    /// by comparing their outer and inner index arrays.
    inline bool same_pattern(const Eigen::SparseMatrix<double>& a, const Eigen::SparseMatrix<double>& b)
    {
        return a.rows() == b.rows() && a.cols() == b.cols()
            && a.nonZeros() == b.nonZeros()
            && std::memcmp(a.outerIndexPtr(), b.outerIndexPtr(),
                           static_cast<std::size_t>(a.outerSize() + 1) * sizeof(typename Eigen::SparseMatrix<double>::StorageIndex)) == 0
            && std::memcmp(a.innerIndexPtr(), b.innerIndexPtr(),
                           static_cast<std::size_t>(a.nonZeros()) * sizeof(typename Eigen::SparseMatrix<double>::StorageIndex)) == 0;
    }

} // namespace mhs::sim
