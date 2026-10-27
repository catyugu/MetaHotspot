// Builds a mhs::core::Model for the chiplet 2.5D stack at one mesh level.
//
// The mesh is a graded Cartesian grid whose in-plane lines include every chiplet
// edge, so a chiplet is always a whole set of cells and the integrated power
// equals the specified total. The z distribution is solved so the cell count
// equals the paper's Table I unknown count exactly.
#pragma once

#include "core/model.hpp"
#include "stack.hpp"

#include <array>
#include <vector>

namespace mhs::dtdm {

    /// A model plus the per-cell bookkeeping the DTDM needs.
    struct StackMesh {
        mhs::core::Model model;
        std::array<std::vector<double>, 3> axes;
        std::vector<mhs::core::Index> cells_of_module[kModuleCount];
        std::vector<std::vector<mhs::core::Index>> cells_of_layer;
        std::vector<double> cell_k; // conductivity per cell
        std::array<int, kModuleCount> nz_per_layer {{0, 0, 0}};
        long dof = 0;
        double total_power = 0.0;
        std::array<int, 4> group_nz {{0, 0, 0, 0}};
    };

    /// Graded axis over the sink footprint with the chiplet edges as grid lines.
    /// ``axis`` is 0 for x and 1 for y: the two axes have different chiplet edges.
    std::vector<double> build_axis(const Layout& layout, const std::vector<Chiplet>& chiplets, int axis);

    /// Per-layer z cell counts whose total cell count equals ``layout.target``.
    std::vector<int> solve_layer_nz(const StackMesh& probe, const Layout& layout, const std::vector<Chiplet>& chiplets,
        std::array<int, 4>& group_nz);

    /// Build the full model for one level. Throws if the DoF count misses.
    StackMesh build_stack_mesh(const SystemSpec& spec, const Layout& layout);

    /// In-plane cell count covered by the union of the chiplet rectangles.
    long covered_cells(
        const std::vector<double>& x, const std::vector<double>& y, const std::vector<Chiplet>& chiplets);

} // namespace mhs::dtdm
