#pragma once

#include "core/vector.hpp"
#include <span>
#include <stdexcept>

namespace mhs::core {
    // Contiguous row-major owning matrix; algorithms belong to numerics.
    template <class T> class Matrix {
    public:
        Matrix() = default;
        Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(count(rows, cols)) { }
        std::size_t rows() const noexcept { return rows_; }
        std::size_t cols() const noexcept { return cols_; }
        T* data() noexcept { return data_.data(); }
        const T* data() const noexcept { return data_.data(); }
        T& operator()(std::size_t i, std::size_t j) noexcept { return data_[i * cols_ + j]; }
        const T& operator()(std::size_t i, std::size_t j) const noexcept { return data_[i * cols_ + j]; }
        std::span<T> row(std::size_t i) noexcept { return std::span<T>(data_).subspan(i * cols_, cols_); }
        std::span<const T> row(std::size_t i) const noexcept
        { return std::span<const T>(data_).subspan(i * cols_, cols_); }

    private:
        static std::size_t count(std::size_t rows, std::size_t cols)
        {
            if (cols && rows > Vector<T>().max_size() / cols)
                throw std::length_error("Matrix dimensions overflow");
            return rows * cols;
        }
        std::size_t rows_ = 0, cols_ = 0;
        Vector<T> data_;
    };
} // namespace mhs::core
