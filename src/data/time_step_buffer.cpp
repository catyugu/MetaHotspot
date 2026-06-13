#include "data/time_step_buffer.hpp"

namespace mhs::core {

    TimeStepBuffer::TimeStepBuffer(std::size_t cell_count, std::size_t capacity)
        : slots_(capacity, std::vector<double>(cell_count)), times_(capacity, 0.0), cap_(capacity)
    {
    }

    void TimeStepBuffer::reset(const std::vector<double>& T_initial)
    {
        // Reset writes the initial snapshot into slot 0; future push()es will
        // land in slot 1, 2, ... wrapping around if capacity is exceeded.
        head_ = 1;
        stored_ = 1;
        slots_[0] = T_initial;
        times_[0] = 0.0;
    }

    void TimeStepBuffer::push(const std::vector<double>& T_new, double time)
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

    const std::vector<double>& TimeStepBuffer::latest() const noexcept
    {
        if (stored_ == 0)
            return slots_[0]; // empty buffer; UB documented
        return at(0);
    }

    std::size_t TimeStepBuffer::ring_index(std::size_t i) const noexcept
    {
        if (stored_ < cap_) {
            return stored_ - 1 - i;
        }
        // stored_ == cap_: head_ points to the oldest slot.
        std::size_t head_to_latest = (head_ == 0) ? cap_ - 1 : head_ - 1;
        return (head_to_latest + cap_ - i) % cap_;
    }

    const std::vector<double>& TimeStepBuffer::at(std::size_t i) const noexcept { return slots_[ring_index(i)]; }

    double TimeStepBuffer::time_at(std::size_t i) const noexcept { return times_[ring_index(i)]; }

    double TimeStepBuffer::dt_to(std::size_t i) const noexcept { return time_at(0) - time_at(i); }

    std::size_t TimeStepBuffer::size() const noexcept { return stored_; }
    std::size_t TimeStepBuffer::capacity() const noexcept { return cap_; }

} // namespace mhs::core
