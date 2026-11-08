#pragma once

#include "model/model_definition.hpp"
#include <cmath>
#include <stdexcept>

namespace mhs::compiler {
    inline void validate_study_settings(const mhs::model::ModelSettings& settings)
    {
        using mhs::model::StudyType;
        if (settings.study_type != StudyType::Steady && settings.study_type != StudyType::Transient)
            throw std::invalid_argument("invalid study type");
        if (!std::isfinite(settings.initial_temperature))
            throw std::invalid_argument("initial temperature must be finite");
        if (!std::isfinite(settings.transient_duration))
            throw std::invalid_argument("duration must be finite");
        if (settings.study_type == StudyType::Transient && settings.transient_duration <= 0)
            throw std::invalid_argument("transient duration must be positive");
        if (!std::isfinite(settings.transient_output_interval) || settings.transient_output_interval < 0)
            throw std::invalid_argument("output interval must be finite and nonnegative");
    }
} // namespace mhs::compiler
