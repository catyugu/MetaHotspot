#pragma once

#include "model/project.hpp"
#include <numeric>

namespace mhs::solver {
    struct Side {
        std::size_t first;
        std::span<const double> weights;
        double reference;

        double temperature(std::span<const double> state) const
        {
            return reference + std::inner_product(weights.begin(), weights.end(), state.begin() + first, 0.0);
        }
    };

    inline Side contact_side(const model::Project& project, model::ContactSide side)
    {
        const auto offset = project.dof_offset(side.component_index);
        const auto& asset = project.components()[side.component_index].asset;
        if (const auto* fom = std::get_if<std::shared_ptr<const model::Fom>>(&asset)) {
            static constexpr double unit_weight = 1;
            return {offset + (*fom)->mesh.exterior_facets[side.face_index].owner, {&unit_weight, 1}, 0};
        }
        const auto& erom = *std::get<std::shared_ptr<const model::Erom>>(asset);
        return {offset, erom.Vb.row(side.face_index), erom.reference_temperature};
    }
} // namespace mhs::solver
