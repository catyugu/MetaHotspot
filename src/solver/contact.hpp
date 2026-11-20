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
        const auto& component = project.components()[side.component_index].model;
        if (const auto* facet = std::get_if<model::FacetId>(&side.face)) {
            static constexpr double unit_weight = 1;
            const auto& fom = *std::get<model::FomInstance>(component).asset;
            return {offset + fom.discrete.cell_facets[*facet].owner(), {&unit_weight, 1}, 0};
        }
        const auto& erom = *std::get<model::EromInstance>(component).asset;
        const auto id = static_cast<core::EntityIndex>(std::get<model::EromInterfaceId>(side.face));
        return {offset, erom.Vb.row(id), erom.reference_temperature};
    }
} // namespace mhs::solver
