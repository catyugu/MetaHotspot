#pragma once

#include "model/types.hpp"

namespace mhs::solver {

    struct StudyOptions {
        mhs::model::StudyType type = mhs::model::StudyType::Steady;
        double duration = 0.0;
        double output_interval = 1.0;
    };

} // namespace mhs::solver
