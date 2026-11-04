#pragma once

#include "core/types.hpp"

namespace mhs::model {
    using core::Index;
    using core::invalidIndex;
    using TableIndex = uint32_t;
    inline constexpr TableIndex invalidTableIndex = std::numeric_limits<TableIndex>::max();

    enum class StudyType : uint8_t { Steady, Transient };

} // namespace mhs::model
