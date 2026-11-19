#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <string_view>

namespace mhs::model::erom_codec {
    inline constexpr std::string_view magic = "MHEROM\r\n";
    inline constexpr std::uint32_t version = 2;
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);

    template <typename T> T little_endian(T value)
    {
        if constexpr (std::endian::native != std::endian::little) {
            auto bytes = std::bit_cast<std::array<char, sizeof(T)>>(value);
            std::ranges::reverse(bytes);
            return std::bit_cast<T>(bytes);
        }
        return value;
    }
} // namespace mhs::model::erom_codec
