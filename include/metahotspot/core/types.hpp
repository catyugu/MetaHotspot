#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace mhs::core {

    using Index = std::size_t;
    inline constexpr Index invalidIndex = std::numeric_limits<Index>::max();

} // namespace mhs::core
