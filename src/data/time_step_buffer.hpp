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
    explicit TimeStepBuffer(std::size_t cell_count, std::size_t capacity);

    /// Reset buffer to a single slot filled with T_initial at t=0.
    void reset(const std::vector<double>& T_initial);

    /// Push a new temperature snapshot at the given time.
    void push(const std::vector<double>& T_new, double time);

    /// Return the most recently pushed temperature vector.
    /// UB if no push has occurred (returns empty vector).
    const std::vector<double>& latest() const noexcept;

    /// Return the temperature vector i steps before latest.
    /// at(0) == latest().  UB if i >= size().
    const std::vector<double>& at(std::size_t i) const noexcept;

    /// Return the timestamp associated with the slot i steps before latest.
    double time_at(std::size_t i) const noexcept;

    /// Return dt = time_at(0) - time_at(i).
    double dt_to(std::size_t i) const noexcept;

    /// Number of snapshots currently stored (0 right after construction,
    /// ≥ 1 after reset/push).
    std::size_t size() const noexcept;

    /// Maximum number of snapshots this buffer can hold.
    std::size_t capacity() const noexcept;

private:
    std::vector<std::vector<double>> slots_;
    std::vector<double> times_;
    std::size_t head_   = 0;   // next write position (ring)
    std::size_t stored_ = 0;   // how many valid snapshots
    std::size_t cap_;
};

} // namespace mhs::core
