#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mhs::core {
    using SparseIndex = std::int32_t;

    constexpr std::uint32_t checked_id_index(std::size_t index)
    {
        if (index >= std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("entity ID exceeds uint32_t range (UINT32_MAX is reserved)");
        return static_cast<std::uint32_t>(index);
    }

    // Entity references have a platform-independent representation. Array sizes
    // and offsets remain std::size_t; CSR storage uses SparseIndex.
    template <class Tag> struct Id {
        using rep_type = std::uint32_t;
        static constexpr rep_type invalid_value = std::numeric_limits<rep_type>::max();
        rep_type value = invalid_value;

        constexpr Id() noexcept = default;
        constexpr explicit Id(std::size_t index) : value(checked_id_index(index)) { }
        constexpr bool valid() const noexcept { return value != invalid_value; }
        auto operator<=>(const Id&) const = default;
    };
} // namespace mhs::core
