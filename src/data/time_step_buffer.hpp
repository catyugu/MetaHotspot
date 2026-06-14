#pragma once

#include <cstddef>
#include <vector>

namespace mhs::core {

/// Ring-buffer for time-step history: stores temperature vectors and their
/// associated timestamps.  Intended for BDF-k schemes where the k most recent
/// solution snapshots are needed.
///
/// Indexing convention:
///   at(0)  == latest  (most recently pushed)
///   at(1)  == one step before latest
///   at(i)  == i steps before latest
///
/// Pre-condition: `push` (or `reset`) must have been called at least once
/// before `latest()` / `at()` / `time_at()` / `dt_to()` are used.
/// Violating this returns a reference to an empty vector (documented UB).
class TimeStepBuffer {
public:
    explicit TimeStepBuffer(std::size_t cell_count, std::size_t capacity)
        : slots_(capacity, std::vector<double>(cell_count)), times_(capacity, 0.0), cap_(capacity)
    {
    }

    /// Reset buffer to a single slot filled with T_initial at t=0.
    inline void reset(const std::vector<double>& T_initial)
    {
        head_  = 1;
        stored_ = 1;
        slots_[0] = T_initial;
        times_[0] = 0.0;
    }

    /// Push a new temperature snapshot at the given time.
    inline void push(const std::vector<double>& T_new, double time)
    {
        if (stored_ < cap_) {
            slots_[head_] = T_new;
            times_[head_] = time;
            ++stored_;
            ++head_;
            if (head_ == cap_)
                head_ = 0;
        }
        else {
            // Buffer full: overwrite oldest (head_ points at the oldest slot).
            slots_[head_] = T_new;
            times_[head_] = time;
            ++head_;
            if (head_ == cap_)
                head_ = 0;
        }
    }

    /// Return the most recently pushed temperature vector.
    /// UB if no push has occurred (returns empty vector).
    inline const std::vector<double>& latest() const noexcept
    {
        if (stored_ == 0)
            return slots_[0];
        return at(0);
    }

    /// Return the temperature vector i steps before latest.
    /// at(0) == latest().  UB if i >= size().
    inline const std::vector<double>& at(std::size_t i) const noexcept { return slots_[ring_index(i)]; }

    /// Return the timestamp associated with the slot i steps before latest.
    inline double time_at(std::size_t i) const noexcept { return times_[ring_index(i)]; }

    /// Return dt = time_at(0) - time_at(i).
    inline double dt_to(std::size_t i) const noexcept { return time_at(0) - time_at(i); }

    /// Number of snapshots currently stored (0 right after construction,
    /// >= 1 after reset/push).
    std::size_t size() const noexcept { return stored_; }

    /// Maximum number of snapshots this buffer can hold.
    std::size_t capacity() const noexcept { return cap_; }

private:
    /// Compute the ring-buffer slot index for the i-th snapshot before latest.
    /// i == 0 returns latest, i == 1 returns one step before, etc.
    /// Precondition: i < stored_.
    inline std::size_t ring_index(std::size_t i) const noexcept
    {
        if (stored_ < cap_) {
            return stored_ - 1 - i;
        }
        // stored_ == cap_: head_ points to the oldest slot.
        std::size_t head_to_latest = (head_ == 0) ? cap_ - 1 : head_ - 1;
        return (head_to_latest + cap_ - i) % cap_;
    }

    std::vector<std::vector<double>> slots_;
    std::vector<double> times_;
    std::size_t head_   = 0;   // next write position (ring)
    std::size_t stored_ = 0;   // how many valid snapshots
    std::size_t cap_;
};

} // namespace mhs::core
