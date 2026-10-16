#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace mhs::core {

    using Index = std::size_t;
    inline constexpr Index invalidIndex = std::numeric_limits<Index>::max();

    using TableIndex = uint32_t;
    inline constexpr TableIndex invalidTableIndex = std::numeric_limits<TableIndex>::max();

    enum class StudyType { Steady, Transient };

    enum class BcType : uint8_t { None = 0, FirstType = 1, SecondType = 2, ThirdType = 3 };

} // namespace mhs::core
