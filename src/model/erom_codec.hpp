#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>

namespace mhs::model::erom_codec {
    inline constexpr std::string_view magic = "MHEROM\r\n";
    inline constexpr std::uint32_t version = 2;
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);

    // Numeric arrays are contiguous, with matrices in row-major order.
    template <typename Matrix> auto elements(Matrix& value)
    {
        return std::span(value.data(), value.rows() * value.cols());
    }

    template <typename T> void transfer(std::span<T> values, const auto& bytes)
    {
        if constexpr (std::is_const_v<T> && std::endian::native != std::endian::little) {
            for (const auto& value : values) {
                auto buffer = std::bit_cast<std::array<char, sizeof(T)>>(value);
                std::ranges::reverse(buffer);
                bytes(buffer.data(), buffer.size());
            }
        }
        else if (!values.empty()) {
            using Char = std::conditional_t<std::is_const_v<T>, const char, char>;
            auto* data = reinterpret_cast<Char*>(values.data());
            bytes(data, values.size_bytes());
            if constexpr (std::endian::native != std::endian::little)
                for (std::size_t offset = 0; offset < values.size_bytes(); offset += sizeof(T))
                    std::reverse(data + offset, data + offset + sizeof(T));
        }
    }
} // namespace mhs::model::erom_codec
