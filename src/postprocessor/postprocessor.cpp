#include "postprocessor.hpp"
#include <vector>

namespace mhs {

    std::vector<double> Postprocessor::interpolate_cell_to_node(
        const model::InternalModel& model,
        const std::vector<double>& cell_temperature) const
    {
        (void)model;
        (void)cell_temperature;
        return std::vector<double>(0);
    }

    double Postprocessor::max_temperature(const std::vector<double>& T) const
    {
        if (T.empty()) {
            return 0.0;
        }
        double max_val = T[0];
        for (const auto& v : T) {
            if (v > max_val) {
                max_val = v;
            }
        }
        return max_val;
    }

    double Postprocessor::min_temperature(const std::vector<double>& T) const
    {
        if (T.empty()) {
            return 0.0;
        }
        double min_val = T[0];
        for (const auto& v : T) {
            if (v < min_val) {
                min_val = v;
            }
        }
        return min_val;
    }

} // namespace mhs