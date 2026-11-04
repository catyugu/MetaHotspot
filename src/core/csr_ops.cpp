#include "core/csr_ops.hpp"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace mhs::core {
    CSRPattern::CSRPattern(std::size_t rows, std::size_t cols)
        : rows_(detail::checked_sparse_size(rows)), cols_(detail::checked_sparse_size(cols)), row_ptr_(rows + 1, 0)
    {
    }

    CSRPattern::CSRPattern(std::size_t rows, std::size_t cols, Vector<SparseIndex> row_ptr, Vector<SparseIndex> col_idx)
        : rows_(detail::checked_sparse_size(rows))
        , cols_(detail::checked_sparse_size(cols))
        , row_ptr_(std::move(row_ptr))
        , col_idx_(std::move(col_idx))
    {
        detail::checked_sparse_size(col_idx_.size());
        if (row_ptr_.size() != rows + 1 || row_ptr_.front() != 0
            || row_ptr_.back() != static_cast<SparseIndex>(col_idx_.size()))
            throw std::invalid_argument("Invalid CSR arrays");
        for (std::size_t i = 0; i < rows; ++i) {
            const auto first = row_ptr_[i], last = row_ptr_[i + 1];
            if (first < 0 || last < first || static_cast<std::size_t>(last) > col_idx_.size())
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

    core::CSRMatrix<double> build_csr(std::size_t rows, std::size_t cols, core::Vector<core::Triplet<double>> entries)
    {
        core::detail::checked_sparse_size(rows);
        core::detail::checked_sparse_size(cols);
        core::detail::checked_sparse_size(entries.size());
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
        // Row-independent sorting/reduction starts only after serial bucketing.
        // Small assemblies retain the single-pass append path below.
        if (entries.size() >= 262144 && rows >= 512) {
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, rows, 256), [&](const auto& range) {
                for (auto i = range.begin(); i != range.end(); ++i) {
                    auto first = entries.begin() + offsets[i], last = entries.begin() + offsets[i + 1];
                    std::sort(first, last, [](const auto& a, const auto& b) { return a.col < b.col; });
                    auto out = first;
                    while (first != last) {
                        const auto column = first->col;
                        double value = 0;
                        do {
                            value += (first++)->value;
                        } while (first != last && first->col == column);
                        out->col = column;
                        (out++)->value = value;
                    }
                    next[i] = static_cast<SparseIndex>(out - (entries.begin() + offsets[i]));
                }
            });
            Vector<SparseIndex> ptr(rows + 1, 0);
            for (std::size_t i = 0; i < rows; ++i)
                ptr[i + 1] = ptr[i] + next[i];
            Vector<SparseIndex> col(ptr.back());
            Vector<double> val(ptr.back());
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, rows, 256), [&](const auto& range) {
                for (auto i = range.begin(); i != range.end(); ++i)
                    for (SparseIndex k = 0; k < next[i]; ++k) {
                        const auto& entry = entries[offsets[i] + k];
                        col[ptr[i] + k] = entry.col;
                        val[ptr[i] + k] = entry.value;
                    }
            });
            return {std::shared_ptr<const CSRPattern>(
                        new CSRPattern(CSRPattern::CanonicalTag {}, rows, cols, std::move(ptr), std::move(col))),
                std::move(val)};
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
        return {std::shared_ptr<const CSRPattern>(
                    new CSRPattern(CSRPattern::CanonicalTag {}, rows, cols, std::move(ptr), std::move(col))),
            std::move(val)};
    }

    void spmv(core::CSRMatrixView<double> a, std::span<const double> x, std::span<double> y, double alpha, double beta)
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

    core::CSRMatrix<double> add_scaled(const core::CSRMatrix<double>& a, double alpha, const core::CSRMatrix<double>& b)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::invalid_argument("CSR sum dimension mismatch");
        if (a.shared_pattern() == b.shared_pattern()) {
            CSRMatrix<double> result(a.shared_pattern());
            for (std::size_t k = 0; k < result.nnz(); ++k)
                result.values()[k] = a.values()[k] + alpha * b.values()[k];
            return result;
        }
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
            ptr[i + 1] = static_cast<core::SparseIndex>(core::detail::checked_sparse_size(val.size()));
        }
        if (col.size() == a.nnz())
            return {a.shared_pattern(), std::move(val)};
        if (col.size() == b.nnz())
            return {b.shared_pattern(), std::move(val)};
        return {std::shared_ptr<const CSRPattern>(
                    new CSRPattern(CSRPattern::CanonicalTag {}, a.rows(), a.cols(), std::move(ptr), std::move(col))),
            std::move(val)};
    }
    void add_scaled_into(
        const CSRMatrix<double>& a, double alpha, const CSRMatrix<double>& b, CSRMatrix<double>& result)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols() || result.rows() != a.rows() || result.cols() != a.cols())
            throw std::invalid_argument("CSR sum dimension mismatch");
        const auto av = a.view(), bv = b.view(), out = result.view();
        auto values = result.values();
        for (std::size_t i = 0; i < out.rows; ++i) {
            auto ka = av.row_ptr[i], kb = bv.row_ptr[i];
            const auto ea = av.row_ptr[i + 1], eb = bv.row_ptr[i + 1];
            for (auto k = out.row_ptr[i]; k < out.row_ptr[i + 1]; ++k) {
                const auto col = out.col_idx[k];
                double value = 0.0;
                if (ka < ea && av.col_idx[ka] == col)
                    value += av.values[ka++];
                if (kb < eb && bv.col_idx[kb] == col)
                    value += alpha * bv.values[kb++];
                values[k] = value;
            }
            if (ka != ea || kb != eb)
                throw std::invalid_argument("CSR output pattern does not contain input patterns");
        }
    }
} // namespace mhs::core
