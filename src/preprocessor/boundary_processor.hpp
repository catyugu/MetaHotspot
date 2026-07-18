#pragma once

#include "data/model.hpp"
#include "expr/expr.hpp"
#include "model/model_definition.hpp"

#include <functional>
#include <vector>

namespace mhs::sim {

    template <typename... Ts> struct overloaded : Ts... {
        using Ts::operator()...;
    };
    template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

    struct CompiledBoundaryRegion {
        mhs::model::Axis axis = mhs::model::Axis::Z;
        double coordinate = 0.0;
        std::vector<mhs::model::RegionRect> rectangles;
        mhs::core::BcType type = mhs::core::BcType::None;
        uint16_t parameter_index = 0;
    };

    struct DefaultBoundary {
        mhs::core::BcType type = mhs::core::BcType::None;
        uint16_t parameter_index = 0;
    };

    std::vector<CompiledBoundaryRegion> compile_boundary_patches(
        const std::vector<mhs::model::BoundaryPatch>& boundaries, mhs::core::BCParamTable& parameters,
        double si_scale, const std::function<std::string(const std::string&)>& rewriter,
        const mhs::core::SymbolTable& symbols);

    bool point_in_region(const CompiledBoundaryRegion& region, double a, double b);

} // namespace mhs::sim
