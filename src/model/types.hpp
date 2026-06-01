#pragma once

#include <array>
#include <cstdint>
#include <functional>

namespace mhs {

    enum class StudyType { Steady,
        Transient };

    enum class BcType : uint8_t { None = 0,
        FirstType = 1,
        SecondType = 2,
        ThirdType = 3 };

    struct FieldContext {
        double x = 0.0, y = 0.0, z = 0.0;
        double T = 0.0;
        double t = 0.0;
    };

    using FieldEvaluator = std::function<double(const FieldContext&)>;

    enum class FaceDir : size_t { XM = 0,
        XP = 1,
        YM = 2,
        YP = 3,
        ZM = 4,
        ZP = 5 };

    constexpr size_t FACE_COUNT = 6;

    constexpr std::array<FaceDir, FACE_COUNT> FACE_DIRS = {
        FaceDir::XM, FaceDir::XP, FaceDir::YM, FaceDir::YP, FaceDir::ZM, FaceDir::ZP};

} // namespace mhs