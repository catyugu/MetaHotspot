#pragma once

#include "model/internal_model.hpp"
#include "model/io_model.hpp"

namespace mhs::preprocessor {

    // Convert length unit to SI (meters) scale factor
    double length_unit_to_si(model::LengthUnit unit);

    // Compute layer Z ranges from IO layers (top-down stacking)
    // Returns z_start and z_end vectors (in SI meters)
    void compute_layer_z_ranges(const std::vector<model::Layer>& layers,
        double si_scale,
        std::vector<double>& z_start,
        std::vector<double>& z_end);

    // Determine which block a cell at (cx, cy, cz) belongs to in a layer
    // Returns block index or -1 if cell is virtual
    int find_block_for_cell(const model::Layer& layer,
        double cx, double cy, double cz,
        double si_scale,
        double layer_z_start, double layer_z_end);

    // Resolve cell validity, layer assignment, and material assignment
    // Populates valid_mask, index_map, layer_id, material_id in CellFields
    void resolve_layers(const std::vector<model::Layer>& layers,
        const model::MeshGeometry& mesh,
        double si_scale,
        const std::vector<double>& layer_z_start,
        const std::vector<double>& layer_z_end,
        const std::unordered_map<std::string, size_t>& name_to_idx,
        model::CellFields& cells);

} // namespace mhs::preprocessor