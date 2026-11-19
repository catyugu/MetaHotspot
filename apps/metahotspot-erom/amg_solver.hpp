#pragma once

#include "extraction.hpp"
#include <memory>

namespace mhs::app::erom {
    // Own one hierarchy independently of the matrix supplied to solve().
    // A frequency's lower-HTC hierarchy is shared by every port and HTC sample.
    class ReusableAmg {
    public:
        explicit ReusableAmg(const Sparse& reference, double tolerance = 1e-10);
        ~ReusableAmg();
        ReusableAmg(const ReusableAmg&) = delete;
        ReusableAmg& operator=(const ReusableAmg&) = delete;
        Vector apply(const Vector& rhs) const;
        Vector solve(const Sparse& matrix, const Vector& rhs, Vector initial) const;
        std::size_t iterations() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace mhs::app::erom
