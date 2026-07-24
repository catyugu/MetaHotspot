#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace mhs::core {

    /// Ring-buffer of accepted (state, time) snapshots for BDF-k multi-step schemes.
    ///
    /// Indexing convention (valid for i < size()):
    ///   at(0) == current()  — most recently accepted solution
    ///   at(1)               — one step before current
    ///   at(i)               — i steps before current
    ///
    /// A default-constructed object has size() == 0 and is not usable until
    /// initialize() is called.
    class SolutionHistory {
    public:
        explicit SolutionHistory(std::size_t state_count, std::size_t capacity)
            : slots_(capacity, std::vector<double>(state_count)), times_(capacity, 0.0), cap_(capacity)
        {
        }

        /// Initialize with the first (t=0) solution snapshot.
        /// Post-condition: size() == 1, current() == initial_state, time_at(0) == t0.
        inline void initialize(std::span<const double> initial_state, double t0 = 0.0)
        {
            head_ = 1;
            stored_ = 1;
            slots_[0].assign(initial_state.begin(), initial_state.end());
            times_[0] = t0;
        }

        /// Record a newly accepted solution snapshot.
        /// The buffer wraps around once full so the k most recent steps are kept.
        inline void accept(std::span<const double> state, double time)
        {
            slots_[head_].assign(state.begin(), state.end());
            times_[head_] = time;
            if (stored_ < cap_)
                ++stored_;
            ++head_;
            if (head_ == cap_)
                head_ = 0;
        }

        /// Most recently accepted solution vector.
        /// UB if no snapshot has ever been stored (returns a reference to an
        /// empty vector for the default-constructed case).
        inline std::span<const double> current() const noexcept
        {
            if (stored_ == 0)
                return slots_[0];
            return at(0);
        }

        /// Solution vector i steps before current (i=0 → current).
        /// Pre-condition: i < size().
        inline std::span<const double> at(std::size_t i) const noexcept { return slots_[ring_index(i)]; }

        /// Timestamp of the snapshot i steps before current.
        inline double time_at(std::size_t i) const noexcept { return times_[ring_index(i)]; }

        /// Previous accepted step size = time_at(0) - time_at(1).
        /// Pre-condition: size() >= 2.
        inline double previous_dt() const noexcept { return time_at(0) - time_at(1); }

        /// Number of snapshots currently stored (0 right after default construction).
        std::size_t size() const noexcept { return stored_; }

        /// Maximum number of snapshots this buffer can hold.
        std::size_t capacity() const noexcept { return cap_; }

    private:
        /// Ring-buffer slot index for the i-th snapshot before current.
        /// Pre-condition: i < stored_.
        inline std::size_t ring_index(std::size_t i) const noexcept
        {
            // tail always points to the slot holding the current (latest) snapshot.
            std::size_t tail = (head_ + cap_ - 1) % cap_;
            return (tail + cap_ - i) % cap_;
        }

        std::vector<std::vector<double>> slots_;
        std::vector<double> times_;
        std::size_t head_ = 0; // next write position (ring)
        std::size_t stored_ = 0; // how many valid snapshots
        std::size_t cap_;
    };

} // namespace mhs::core
