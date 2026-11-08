#pragma once

#include "model/types.hpp"
#include <string>

namespace mhs::solver {

    struct StudyOptions {
        mhs::model::StudyType type = mhs::model::StudyType::Steady;
        double duration = 0.0;
        double output_interval = 1.0;
    };

    struct ProbeSpec {
        std::string name;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

} // namespace mhs::solver
