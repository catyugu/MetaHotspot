#pragma once

#include <cstddef>
#include <vector>

namespace mhs::sim::time_scheme {

    /// Immutable output time grid.  Strict convention for uniform grids:
    /// times_[i] = i * output_step for i = 0..N where N = floor(duration / output_step).
    /// The endpoint `duration` is included only when `duration / output_step` is integer.
    class OutputTimeGrid {
    public:
        OutputTimeGrid() = default;

        /// Uniform grid: t_i = i * output_step for i in [0, floor(duration / output_step)].
        /// If output_step <= 0, the grid is empty (solver-only output).
        explicit OutputTimeGrid(double duration, double output_step)
        {
            if (duration <= 0.0 || output_step <= 0.0)
                return;
            const std::size_t n = static_cast<std::size_t>(duration / output_step);
            times_.reserve(n + 1);
            for (std::size_t i = 0; i <= n; ++i)
                times_.push_back(static_cast<double>(i) * output_step);
        }

        /// Explicit times (non-uniform).  Must be sorted ascending.
        explicit OutputTimeGrid(std::vector<double> times) : times_(std::move(times))
        {
        }

        const std::vector<double>& times() const noexcept { return times_; }
        std::size_t size() const noexcept { return times_.size(); }
        bool empty() const noexcept { return times_.empty(); }

    private:
        std::vector<double> times_;
    };

} // namespace mhs::sim::time_scheme