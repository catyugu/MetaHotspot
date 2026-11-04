// Builds a mhs::model::Model for the chiplet 2.5D stack at one mesh level.
//
// The mesh is a graded Cartesian grid whose in-plane lines include every chiplet
// edge, so a chiplet is always a whole set of cells and the integrated power
// equals the specified total. The z distribution is solved so the cell count
// equals the paper's Table I unknown count exactly.
#pragma once

#include "model/model.hpp"
#include "stack.hpp"
#include <cstdint>

#include "core/vector.hpp"
#include <array>

namespace mhs::dtdm {

    /// A model plus the per-cell bookkeeping the DTDM needs.
    struct StackMesh {
        mhs::model::Model model;
        std::array<mhs::core::Vector<double>, 3> axes;
        mhs::core::Vector<std::size_t> cells_of_module[kModuleCount];
        mhs::core::Vector<mhs::core::Vector<std::size_t>> cells_of_layer;
        mhs::core::Vector<double> cell_k; // conductivity per cell
        std::array<int, kModuleCount> nz_per_layer {{0, 0, 0}};
        std::size_t dof = 0;
        double total_power = 0.0;
        std::array<int, 4> group_nz {{0, 0, 0, 0}};
    };

    /// Graded axis over the sink footprint with the chiplet edges as grid lines.
    /// ``axis`` is 0 for x and 1 for y: the two axes have different chiplet edges.
    mhs::core::Vector<double> build_axis(const Layout& layout, const mhs::core::Vector<Chiplet>& chiplets, int axis);

    /// Per-layer z cell counts whose total cell count equals ``layout.target``.
    mhs::core::Vector<int> solve_layer_nz(const StackMesh& probe, const Layout& layout,
        const mhs::core::Vector<Chiplet>& chiplets, std::array<int, 4>& group_nz);

    /// Build the full model for one level. Throws if the DoF count misses.
    StackMesh build_stack_mesh(const SystemSpec& spec, const Layout& layout);

    /// In-plane cell count covered by the union of the chiplet rectangles.
    std::int64_t covered_cells(const mhs::core::Vector<double>& x, const mhs::core::Vector<double>& y,
        const mhs::core::Vector<Chiplet>& chiplets);

} // namespace mhs::dtdm
