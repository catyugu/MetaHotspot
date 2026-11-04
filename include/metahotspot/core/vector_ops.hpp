#pragma once

#include <span>

namespace mhs::core {
    double norm_inf(std::span<const double> x);

    double distance_inf(std::span<const double> x, std::span<const double> y);

    bool all_finite(std::span<const double> x);
} // namespace mhs::core
