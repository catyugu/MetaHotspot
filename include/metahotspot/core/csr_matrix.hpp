#pragma once

#include "core/vector.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace mhs::core {
    using SparseIndex = std::int32_t;

    namespace detail {
        inline std::size_t checked_sparse_size(std::size_t n)
        {
            if (n > static_cast<std::size_t>(std::numeric_limits<SparseIndex>::max()))
                throw std::length_error("CSR exceeds 32-bit index capacity");
            return n;
        }
    } // namespace detail

    template <class T> struct Triplet {
        SparseIndex row, col;
        T value;

        Triplet(std::size_t r, std::size_t c, T v)
            : row(static_cast<SparseIndex>(detail::checked_sparse_size(r)))
            , col(static_cast<SparseIndex>(detail::checked_sparse_size(c)))
            , value(v)
        {
        }
    };

    template <class T> struct CSRMatrixView {
        std::size_t rows, cols;
        std::span<const SparseIndex> row_ptr, col_idx;
        std::span<const T> values;
    };

    // Canonical zero-based CSR: sorted unique columns in every row. Explicit
    // zeros are retained, including when duplicate assembly entries cancel.
    template <class T> class CSRMatrix {
    public:
        CSRMatrix() = default;
        CSRMatrix(std::size_t rows, std::size_t cols)
            : rows_(detail::checked_sparse_size(rows)), cols_(detail::checked_sparse_size(cols)), row_ptr_(rows + 1, 0)
        {
        }
        CSRMatrix(std::size_t rows, std::size_t cols, Vector<SparseIndex> row_ptr, Vector<SparseIndex> col_idx,
            Vector<T> values)
            : rows_(detail::checked_sparse_size(rows))
            , cols_(detail::checked_sparse_size(cols))
            , row_ptr_(std::move(row_ptr))
            , col_idx_(std::move(col_idx))
            , values_(std::move(values))
        {
            detail::checked_sparse_size(values_.size());
            if (row_ptr_.size() != rows + 1 || col_idx_.size() != values_.size() || row_ptr_.front() != 0
                || row_ptr_.back() != static_cast<SparseIndex>(values_.size()))
                throw std::invalid_argument("Invalid CSR arrays");
            for (std::size_t i = 0; i < rows; ++i) {
                const auto first = row_ptr_[i], last = row_ptr_[i + 1];
                if (first < 0 || last < first || static_cast<std::size_t>(last) > values_.size())
                    throw std::invalid_argument("Invalid CSR row pointers");
                SparseIndex previous = -1;
                for (auto k = first; k < last; ++k) {
                    const auto col = col_idx_[k];
                    if (col <= previous || static_cast<std::size_t>(col) >= cols)
                        throw std::invalid_argument("CSR columns must be sorted, unique and in range");
                    previous = col;
                }
            }
        }
        std::size_t rows() const noexcept { return rows_; }
        std::size_t cols() const noexcept { return cols_; }
        std::size_t nnz() const noexcept { return values_.size(); }
        std::span<const SparseIndex> row_ptr() const noexcept { return row_ptr_; }
        std::span<const SparseIndex> col_idx() const noexcept { return col_idx_; }
        std::span<T> values() noexcept { return values_; }
        std::span<const T> values() const noexcept { return values_; }
        CSRMatrixView<T> view() const noexcept { return {rows_, cols_, row_ptr_, col_idx_, values_}; }
        T coeff(std::size_t i, std::size_t j) const
        {
            const auto first = col_idx_.begin() + row_ptr_[i], last = col_idx_.begin() + row_ptr_[i + 1];
            const auto pos = std::lower_bound(first, last, j);
            return pos != last && static_cast<std::size_t>(*pos) == j ? values_[pos - col_idx_.begin()] : T {};
        }

    private:
        // Only construction kernels that already establish the canonical
        // invariant may bypass the public array constructor's validation.
        struct CanonicalTag { };
        CSRMatrix(CanonicalTag, std::size_t rows, std::size_t cols, Vector<SparseIndex> row_ptr,
            Vector<SparseIndex> col_idx, Vector<T> values)
            : rows_(rows)
            , cols_(cols)
            , row_ptr_(std::move(row_ptr))
            , col_idx_(std::move(col_idx))
            , values_(std::move(values))
        {
        }
        friend CSRMatrix<double> build_csr(std::size_t, std::size_t, Vector<Triplet<double>>);
        friend CSRMatrix<double> add_scaled(const CSRMatrix<double>&, double, const CSRMatrix<double>&);

        std::size_t rows_ = 0, cols_ = 0;
        Vector<SparseIndex> row_ptr_ {0};
        Vector<SparseIndex> col_idx_;
        Vector<T> values_;
    };
} // namespace mhs::core
