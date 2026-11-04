#pragma once

#include <cstdint>
#include <limits>

namespace mhs::core {
    // Entity references have a platform-independent representation. Array sizes
    // and offsets remain std::size_t; CSR storage uses SparseIndex.
    using EntityIndex = std::uint32_t;
    using SparseIndex = std::int32_t;
    inline constexpr EntityIndex invalid_id = std::numeric_limits<EntityIndex>::max();
} // namespace mhs::core
