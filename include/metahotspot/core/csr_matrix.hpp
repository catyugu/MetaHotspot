#pragma once
#include "core/types.hpp"

#include "core/vector.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace mhs::core {

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
    };

    template <class T> struct CSRMatrixView {
        std::size_t rows, cols;
        std::span<const SparseIndex> row_ptr, col_idx;
        std::span<const T> values;
    };

    template <class T> class CSRMatrix;

    // Canonical zero-based CSR structure. Shared through const ownership;
    // sorted unique columns and explicit zero slots survive numerical updates.
    class CSRPattern {
    public:
        CSRPattern(std::size_t rows, std::size_t cols);
        CSRPattern(std::size_t rows, std::size_t cols, Vector<SparseIndex> row_ptr, Vector<SparseIndex> col_idx);
        CSRPattern(const CSRPattern&) = delete;
        CSRPattern& operator=(const CSRPattern&) = delete;
        std::size_t rows() const noexcept { return rows_; }
        std::size_t cols() const noexcept { return cols_; }
        std::size_t nnz() const noexcept { return col_idx_.size(); }
        std::span<const SparseIndex> row_ptr() const noexcept { return row_ptr_; }
        std::span<const SparseIndex> col_idx() const noexcept { return col_idx_; }

    private:
        // Construction kernels already establish the canonical invariant.
        struct CanonicalTag { };
        CSRPattern(
            CanonicalTag, std::size_t rows, std::size_t cols, Vector<SparseIndex> row_ptr, Vector<SparseIndex> col_idx)
            : rows_(rows), cols_(cols), row_ptr_(std::move(row_ptr)), col_idx_(std::move(col_idx))
        {
        }
        friend CSRMatrix<double> build_csr(std::size_t, std::size_t, Vector<Triplet<double>>);
        friend CSRMatrix<double> add_scaled(const CSRMatrix<double>&, double, const CSRMatrix<double>&);

        std::size_t rows_, cols_;
        Vector<SparseIndex> row_ptr_, col_idx_;
    };

    template <class T> class CSRMatrix {
    public:
        using PatternPtr = std::shared_ptr<const CSRPattern>;

        CSRMatrix() : CSRMatrix(0, 0) { }
        CSRMatrix(std::size_t rows, std::size_t cols) : CSRMatrix(std::make_shared<const CSRPattern>(rows, cols)) { }
        explicit CSRMatrix(PatternPtr pattern) : pattern_(require_pattern(std::move(pattern))), values_(pattern_->nnz())
        {
        }
        CSRMatrix(PatternPtr pattern, Vector<T> values)
            : pattern_(require_pattern(std::move(pattern))), values_(std::move(values))
        {
            if (values_.size() != this->pattern().nnz())
                throw std::invalid_argument("CSR values do not match pattern");
        }
        CSRMatrix(std::size_t rows, std::size_t cols, Vector<SparseIndex> row_ptr, Vector<SparseIndex> col_idx,
            Vector<T> values)
            : CSRMatrix(std::make_shared<const CSRPattern>(rows, cols, std::move(row_ptr), std::move(col_idx)),
                  std::move(values))
        {
        }
        const CSRPattern& pattern() const noexcept { return *pattern_; }
        PatternPtr shared_pattern() const noexcept { return pattern_; }
        std::size_t rows() const noexcept { return pattern_->rows(); }
        std::size_t cols() const noexcept { return pattern_->cols(); }
        std::size_t nnz() const noexcept { return values_.size(); }
        std::span<const SparseIndex> row_ptr() const noexcept { return pattern_->row_ptr(); }
        std::span<const SparseIndex> col_idx() const noexcept { return pattern_->col_idx(); }
        std::span<T> values() noexcept { return values_; }
        std::span<const T> values() const noexcept { return values_; }
        CSRMatrixView<T> view() const noexcept { return {rows(), cols(), row_ptr(), col_idx(), values_}; }
        T coeff(std::size_t i, std::size_t j) const
        {
            const auto columns = col_idx();
            const auto first = columns.begin() + row_ptr()[i], last = columns.begin() + row_ptr()[i + 1];
            const auto pos = std::lower_bound(first, last, j);
            return pos != last && static_cast<std::size_t>(*pos) == j ? values_[pos - columns.begin()] : T {};
        }

    private:
        static PatternPtr require_pattern(PatternPtr pattern)
        {
            if (!pattern)
                throw std::invalid_argument("CSR pattern is null");
            return pattern;
        }
        PatternPtr pattern_;
        Vector<T> values_;
    };
} // namespace mhs::core
