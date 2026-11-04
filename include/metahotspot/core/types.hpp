#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mhs::core {
    // Entity references have a platform-independent representation. Array sizes
    // and offsets remain std::size_t; CSR storage uses SparseIndex.
    using EntityIndex = std::uint32_t;
    using SparseIndex = std::int32_t;
    inline constexpr EntityIndex invalid_id = std::numeric_limits<EntityIndex>::max();

    // Validate counts once at data production boundaries, before narrowing indices.
    constexpr void require_entity_count(std::size_t count)
    {
        if (count > invalid_id)
            throw std::length_error("entity count exceeds 32-bit capacity (UINT32_MAX is reserved)");
    }
} // namespace mhs::core
