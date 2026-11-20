#include "compiler/geometry_compiler.hpp"
#include <numeric>

namespace mhs::compiler {
    void build_cell_facet_incidence(mhs::model::ConformalMesh& mesh)
    {
        using namespace mhs::model;
        using namespace mhs::core;
        auto& cell_facets = mesh.cell_facets;
        auto& cell_facet_offsets = mesh.cell_facet_offsets;
        auto& cell_facet_ids = mesh.cell_facet_ids;
        cell_facet_offsets.assign(mesh.cell_count() + 1, 0);
        for (const auto& f : cell_facets) {
            if (f.minus != invalid_id)
                ++cell_facet_offsets[f.minus + 1];
            if (f.plus != invalid_id)
                ++cell_facet_offsets[f.plus + 1];
        }
        std::partial_sum(cell_facet_offsets.begin(), cell_facet_offsets.end(), cell_facet_offsets.begin());
        cell_facet_ids.resize(cell_facet_offsets.back());
        auto next = cell_facet_offsets;
        for (std::size_t i = 0; i < cell_facets.size(); ++i) {
            const auto& f = cell_facets[i];
            if (f.minus != invalid_id)
                cell_facet_ids[next[f.minus]++] = static_cast<FacetId>(i);
            if (f.plus != invalid_id)
                cell_facet_ids[next[f.plus]++] = static_cast<FacetId>(i);
        }
    }
} // namespace mhs::compiler
