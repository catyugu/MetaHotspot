#pragma once

#include "model/project.hpp"
#include <numeric>

namespace mhs::solver {
    struct Side {
        std::size_t first;
        std::span<const double> modes;
        double reference;

        std::size_t count() const { return modes.empty() ? 1 : modes.size(); }
        double value(std::size_t i) const { return modes.empty() ? 1.0 : modes[i]; }

        double temperature(std::span<const double> state) const
        {
            const auto values = state.subspan(first, count());
            return reference
                + (modes.empty() ? values.front()
                                 : std::inner_product(modes.begin(), modes.end(), values.begin(), 0.0));
        }
    };

    inline Side contact_side(const model::Project& project, std::size_t i, model::FacetId facet)
    {
        const auto offset = project.dof_offset(i);
        const auto& asset = project.components()[i].asset;
        if (const auto* fom = std::get_if<std::shared_ptr<const model::Fom>>(&asset))
            return {offset + (*fom)->discrete.cell_facets[facet].owner(), {}, 0};
        const auto& erom = *std::get<std::shared_ptr<const model::Erom>>(asset);
        return {offset, erom.Vb.row(facet), erom.reference_temperature};
    }
} // namespace mhs::solver
