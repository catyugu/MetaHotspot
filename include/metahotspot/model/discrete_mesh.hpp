#pragma once
#include "core/vector.hpp"
#include "model/surface.hpp"
#include <span>

namespace mhs::model {
    struct CellGeometry {
        Vec3 min {}, max {};
        Vec3 center() const { return {(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2}; }
        Vec3 size() const { return {max[0] - min[0], max[1] - min[1], max[2] - min[2]}; }
        double volume() const
        {
            auto d = size();
            return d[0] * d[1] * d[2];
        }
    };

    // Positive axis points from minus to plus. An exterior facet has one missing cell.
    struct CellFacet {
        CellId minus = core::invalid_id, plus = core::invalid_id;

        bool exterior() const { return minus == core::invalid_id || plus == core::invalid_id; }
        CellId owner() const { return minus != core::invalid_id ? minus : plus; }
    };

    // Canonical integration geometry and cell-facet incidence. A physical boundary
    // can be partitioned for multiple physics without duplicating facet geometry.
    struct DiscreteMesh {
        mhs::core::Vector<CellGeometry> cells;
        mhs::core::Vector<Facet> facets;
        mhs::core::Vector<CellFacet> cell_facets; // same FacetId as facets
        mhs::core::Vector<std::size_t> cell_facet_offsets;
        mhs::core::Vector<FacetId> cell_facet_ids;
        std::span<const FacetId> incident_facets(CellId c) const
        {
            return std::span(cell_facet_ids)
                .subspan(cell_facet_offsets[c], cell_facet_offsets[c + 1] - cell_facet_offsets[c]);
        }
    };
} // namespace mhs::model
