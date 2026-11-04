#pragma once

#include <algorithm>
#include <cmath>
#include <span>

namespace mhs::numerics {
    inline double norm_inf(std::span<const double> x)
    {
        double result = 0;
        for (double v : x)
            result = std::max(result, std::abs(v));
        return result;
    }
    inline double distance_inf(std::span<const double> x, std::span<const double> y)
    {
        double result = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            result = std::max(result, std::abs(x[i] - y[i]));
        return result;
    }
    inline bool all_finite(std::span<const double> x)
    {
        return std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); });
    }
} // namespace mhs::numerics
