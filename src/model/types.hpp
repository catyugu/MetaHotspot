#pragma once

#include <cstdint>
#include <functional>

namespace mhs {

    enum class StudyType { Steady,
        Transient };

    enum class BcType : uint8_t { None = 0,
        FirstType = 1,
        SecondType = 2,
        ThirdType = 3 };

    enum class ConvergenceStatus { Running,
        Converged,
        Diverged,
        Stalled };

    struct FieldContext {
        double x = 0.0, y = 0.0, z = 0.0;
        double T = 0.0;
        double t = 0.0;
    };

    using FieldEvaluator = std::function<double(const FieldContext&)>;

} // namespace mhs