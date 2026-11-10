#pragma once

#include "definition.hpp"

#include <string>
#include <string_view>
#include <unordered_map>

namespace mhs::app::pta {

    // Private to the PTA adapter. Results use the input's length unit.
    class GeometryEvaluator {
    public:
        explicit GeometryEvaluator(const mhs::core::Vector<VariableSpec>& variables);
        double evaluate(std::string_view expr) const;

    private:
        std::unordered_map<std::string, double> values_;
    };

} // namespace mhs::app::pta
