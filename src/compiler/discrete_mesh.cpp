#include "compiler/geometry_compiler.hpp"
#include <numeric>

namespace mhs::compiler {
    void build_cell_facet_incidence(mhs::model::DiscreteMesh& mesh)
    {
        using namespace mhs::model;
        using namespace mhs::core;
        auto& [cells, facets, cell_facet_offsets, cell_facet_ids] = mesh;
        cell_facet_offsets.assign(cells.size() + 1, 0);
        for (const auto& f : facets) {
            if (f.minus != invalid_id)
                ++cell_facet_offsets[f.minus + 1];
            if (f.plus != invalid_id)
                ++cell_facet_offsets[f.plus + 1];
        }
        std::partial_sum(cell_facet_offsets.begin(), cell_facet_offsets.end(), cell_facet_offsets.begin());
        cell_facet_ids.resize(cell_facet_offsets.back());
        auto next = cell_facet_offsets;
        for (std::size_t i = 0; i < facets.size(); ++i) {
            const auto& f = facets[i];
            if (f.minus != invalid_id)
                cell_facet_ids[next[f.minus]++] = static_cast<FacetId>(i);
            if (f.plus != invalid_id)
                cell_facet_ids[next[f.plus]++] = static_cast<FacetId>(i);
        }
    }
} // namespace mhs::compiler
