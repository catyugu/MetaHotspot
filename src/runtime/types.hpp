#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace mhs {
    using Index = size_t;
    inline constexpr Index invalidIndex = std::numeric_limits<Index>::max();
}

namespace mhs::core {

    using TableIndex = uint32_t;

    enum class StudyType { Steady, Transient };

    enum class BcType : uint8_t { None = 0, FirstType = 1, SecondType = 2, ThirdType = 3 };

    enum class FaceDir : uint8_t { XM = 0, XP = 1, YM = 2, YP = 3, ZM = 4, ZP = 5 };

    constexpr size_t FACE_COUNT = 6;

    constexpr std::array<FaceDir, FACE_COUNT> FACE_DIRS
        = {FaceDir::XM, FaceDir::XP, FaceDir::YM, FaceDir::YP, FaceDir::ZM, FaceDir::ZP};

} // namespace mhs::core
