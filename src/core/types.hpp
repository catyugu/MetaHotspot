#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace mhs::core {

    using Index = std::size_t;
    inline constexpr Index invalidIndex = std::numeric_limits<Index>::max();

    using TableIndex = uint32_t;
    inline constexpr TableIndex invalidTableIndex = std::numeric_limits<TableIndex>::max();

    enum class StudyType { Steady, Transient };

} // namespace mhs::core
