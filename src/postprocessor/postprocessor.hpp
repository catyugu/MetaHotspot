#pragma once

#include "common/internal_model.hpp"
#include <vector>

namespace mhs {

    class Postprocessor {
    public:
        Postprocessor() = default;
        ~Postprocessor() = default;

        std::vector<double> interpolate_cell_to_node(const model::InternalModel& model,
            const std::vector<double>& cell_temperature) const;

        double max_temperature(const std::vector<double>& T) const;
        double min_temperature(const std::vector<double>& T) const;
    };

} // namespace mhs