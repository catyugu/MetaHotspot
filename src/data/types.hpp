#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace mhs::core {

    enum class StudyType { Steady, Transient };

    enum class BcType : uint8_t { None = 0, FirstType = 1, SecondType = 2, ThirdType = 3 };

    // Fluid boundary condition type. Independent of BcType — the same face
    // can carry both a thermal BC and a fluid BC simultaneously.
    enum class FluidBCType : uint8_t {
        None             = 0,
        PressureType     = 1, // Dirichlet on p
        MassFlowRateType = 2, // Neumann: m_dot [kg/s] -> overrides energy netOutflux
        VelocityType     = 3, // Neumann: u [m/s] normal -> overrides energy netOutflux
    };

    enum class FaceDir : size_t { XM = 0, XP = 1, YM = 2, YP = 3, ZM = 4, ZP = 5 };

    constexpr size_t FACE_COUNT = 6;

    constexpr std::array<FaceDir, FACE_COUNT> FACE_DIRS
        = {FaceDir::XM, FaceDir::XP, FaceDir::YM, FaceDir::YP, FaceDir::ZM, FaceDir::ZP};

} // namespace mhs::core
