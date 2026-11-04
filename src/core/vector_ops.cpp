#include "core/vector_ops.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace mhs::core {
    double norm_inf(std::span<const double> x)
    {
        double result = 0;
        for (double v : x)
            result = std::max(result, std::abs(v));
        return result;
    }

    double distance_inf(std::span<const double> x, std::span<const double> y)
    {
        assert(x.size() == y.size());
        double result = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            result = std::max(result, std::abs(x[i] - y[i]));
        return result;
    }

    bool all_finite(std::span<const double> x)
    {
        return std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); });
    }
} // namespace mhs::core
