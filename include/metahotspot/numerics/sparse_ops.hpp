#pragma once

#include "core/csr_matrix.hpp"
#include <cmath>
#include <tuple>

namespace mhs::numerics {
    struct Triplet {
        core::SparseIndex row, col;
        Triplet(std::size_t r, std::size_t c, double v)
            : row(static_cast<core::SparseIndex>(core::CSRMatrix<double>::checked(r)))
            , col(static_cast<core::SparseIndex>(core::CSRMatrix<double>::checked(c)))
            , value(v)
        {
        }
        double value;
    };

    inline core::CSRMatrix<double> build_csr(std::size_t rows, std::size_t cols, core::Vector<Triplet> entries)
    {
        core::CSRMatrix<double>::checked(rows);
        core::CSRMatrix<double>::checked(cols);
        core::CSRMatrix<double>::checked(entries.size());
        core::Vector<core::SparseIndex> offsets(rows + 1, 0);
        for (const auto& e : entries) {
            if (e.row < 0 || e.col < 0 || static_cast<std::size_t>(e.row) >= rows
                || static_cast<std::size_t>(e.col) >= cols)
                throw std::out_of_range("CSR triplet outside matrix");
            ++offsets[static_cast<std::size_t>(e.row) + 1];
        }
        for (std::size_t i = 0; i < rows; ++i)
            offsets[i + 1] += offsets[i];
        auto next = offsets;
        // In-place counting permutation by row, followed by short row sorts.
        for (std::size_t i = 0; i < rows; ++i)
            while (next[i] < offsets[i + 1]) {
                const auto row = static_cast<std::size_t>(entries[next[i]].row);
                if (row == i)
                    ++next[i];
                else
                    std::swap(entries[next[i]], entries[next[row]++]);
            }
        core::Vector<core::SparseIndex> ptr(rows + 1, 0), col;
        core::Vector<double> val;
        col.reserve(entries.size());
        val.reserve(entries.size());
        for (std::size_t i = 0; i < rows; ++i) {
            auto first = entries.begin() + offsets[i], last = entries.begin() + offsets[i + 1];
            std::sort(first, last, [](const auto& a, const auto& b) { return a.col < b.col; });
            while (first != last) {
                const auto column = first->col;
                double value = 0;
                do {
                    value += (first++)->value;
                } while (first != last && first->col == column);
                col.push_back(column);
                val.push_back(value);
            }
            ptr[i + 1] = static_cast<core::SparseIndex>(val.size());
        }
        return {rows, cols, std::move(ptr), std::move(col), std::move(val)};
    }

    // y = alpha A x + beta y. x and y must not overlap.
    inline void spmv(core::CSRMatrixView<double> a, std::span<const double> x, std::span<double> y, double alpha = 1.0,
        double beta = 0.0)
    {
        if (x.size() != a.cols || y.size() != a.rows)
            throw std::invalid_argument("spmv dimension mismatch");
        for (std::size_t i = 0; i < a.rows; ++i) {
            double sum = 0;
            for (auto k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k)
                sum += a.values[k] * x[a.col_idx[k]];
            y[i] = alpha * sum + (beta == 0.0 ? 0.0 : beta * y[i]);
        }
    }

    inline core::CSRMatrix<double> add_scaled(
        const core::CSRMatrix<double>& a, double alpha, const core::CSRMatrix<double>& b)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::invalid_argument("CSR sum dimension mismatch");
        core::Vector<core::SparseIndex> ptr(a.rows() + 1, 0), col;
        core::Vector<double> val;
        col.reserve(a.nnz() + b.nnz());
        val.reserve(a.nnz() + b.nnz());
        for (std::size_t i = 0; i < a.rows(); ++i) {
            auto ka = a.row_ptr()[i], kb = b.row_ptr()[i];
            const auto ea = a.row_ptr()[i + 1], eb = b.row_ptr()[i + 1];
            while (ka < ea || kb < eb) {
                if (kb == eb || (ka < ea && a.col_idx()[ka] < b.col_idx()[kb])) {
                    col.push_back(a.col_idx()[ka]);
                    val.push_back(a.values()[ka++]);
                }
                else if (ka == ea || b.col_idx()[kb] < a.col_idx()[ka]) {
                    col.push_back(b.col_idx()[kb]);
                    val.push_back(alpha * b.values()[kb++]);
                }
                else {
                    col.push_back(a.col_idx()[ka]);
                    val.push_back(a.values()[ka++] + alpha * b.values()[kb++]);
                }
            }
            ptr[i + 1] = static_cast<core::SparseIndex>(core::CSRMatrix<double>::checked(val.size()));
        }
        return {a.rows(), a.cols(), std::move(ptr), std::move(col), std::move(val)};
    }

    inline bool numerically_symmetric(const core::CSRMatrix<double>& a, double tolerance)
    {
        if (a.rows() != a.cols())
            return false;
        double asym2 = 0, scale2 = 0;
        for (std::size_t i = 0; i < a.rows(); ++i)
            for (auto k = a.row_ptr()[i]; k < a.row_ptr()[i + 1]; ++k) {
                const auto j = static_cast<std::size_t>(a.col_idx()[k]);
                const double v = a.values()[k];
                const auto columns = a.col_idx().subspan(static_cast<std::size_t>(a.row_ptr()[j]),
                    static_cast<std::size_t>(a.row_ptr()[j + 1] - a.row_ptr()[j]));
                const auto pos = std::lower_bound(columns.begin(), columns.end(), i);
                const bool present = pos != columns.end() && static_cast<std::size_t>(*pos) == i;
                const double transpose = present ? a.values()[a.row_ptr()[j] + (pos - columns.begin())] : 0.0;
                scale2 += v * v;
                const double d = v - transpose;
                asym2 += d * d;
                // The missing transpose is a second entry of A - A^T.
                if (!present)
                    asym2 += v * v;
            }
        return std::sqrt(asym2) <= tolerance * std::sqrt(scale2) + 1e-300;
    }
} // namespace mhs::numerics
