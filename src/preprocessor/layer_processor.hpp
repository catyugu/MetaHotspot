#pragma once

#include "data/io_structure.hpp"
#include "data/model.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp" // for ParsedFaceKey

namespace mhs::sim {

    // Pre-resolved geometry for a single rect (all values in SI meters)
    struct ResolvedRect {
        bool add_sub;
        double x; // absolute SI x coordinate of rect origin
        double y; // absolute SI y coordinate of rect origin
        double width; // SI width (always positive after normalization)
        double height; // SI height (always positive after normalization)
    };

    // Pre-resolved geometry for a single block
    struct ResolvedBlock {
        std::vector<ResolvedRect> rects;
        std::string material_name;
        std::string ti_reyuan_expr; // kept as string for later mhs::core::parse

        // 该 Block 在世界坐标系中的 Z 范围
        double z_start = 0.0;
        double z_end = 0.0;

        // SmartMacro support
        bool is_smart_macro = false;
    };

    // Pre-resolved geometry for a single layer
    struct ResolvedLayerGeometry {
        std::vector<ResolvedBlock> blocks;
        double z_start; // SI z coordinate of layer bottom
        double z_end; // SI z coordinate of layer top
    };

    // Pre-evaluate all geometry expressions for all layers, including Z ranges
    // This eliminates repeated eval_geometry calls in the cell loops.
    // `symbols` provides the geometry variables each expression may reference.
    std::vector<ResolvedLayerGeometry> resolve_geometry(
        const std::vector<mhs::core::Layer>& layers, double si_scale, const mhs::core::SymbolTable& symbols);

    // Determine which block a cell at (cx, cy, cz) belongs to in a resolved layer
    // Uses pre-evaluated geometry values — no expression evaluation at runtime
    // Traverses blocks in reverse order (last block wins in overlap regions)
    // Returns block index or -1 if cell is virtual
    int find_block_for_cell(const ResolvedLayerGeometry& resolved_layer, double cx, double cy, double cz);

    // Assign every grid cell to its layer + block and write volumetric cell fields.
    // Returns CellFields with index_map (full-grid; invalidIndex = virtual),
    // material_id and heat_source_idx (both compact by active count).
    //
    // `block_hs_map[l][b]` = heat_source_table index for layer l / block b.
    // No BC parameters needed — boundary resolution is a separate step.
    mhs::core::CellFields assign_cell_layers(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const std::unordered_map<std::string, size_t>& name_to_idx,
        const std::vector<std::vector<uint16_t>>& block_hs_map);

    // Resolve boundary patches for every exposed face of every active cell.
    // Single grid traversal: writes directly into boundary.face_bcs[]. No
    // prefix-sum or intermediate scan needed — face_bcs is [N_active * 6].
    //
    // `cells` must already have a valid index_map (from assign_cell_layers).
    // `parsed_face_keys` comes from parse_all_face_keys().
    // Other_bc is the fallback BC for faces that don't match any face key.
    void resolve_boundary_patches(const mhs::core::MeshGeometry& mesh, const mhs::core::CellFields& cells,
        const std::vector<ParsedFaceKey>& parsed_face_keys, mhs::core::BcType other_bc_enum, uint16_t other_bc_idx,
        std::vector<mhs::core::FaceBC>& face_bcs);

    // ── SmartMacro block coupling (face-level, BC-agnostic) ────────────────
    //
    // After assign_cell_layers + resolve_boundary_patches, build the SmartBlock
    // coupling data for every SmartMacro block. For each port face (boundary
    // face of the block) the function determines:
    //   - Active neighbor → C_env = k_n*A/h_n, coupled to neighbor cell DOF
    //   - Domain boundary  → (C_env, T_ref, Q_ext) from face BC match
    //
    // The function does NOT bake any BCs into the modal data — all BC effects
    // enter through the environment parameters (C_env, T_ref, Q_ext) which
    // are assembled into the extended system at runtime.
    void build_smart_block_coupling(const std::vector<ResolvedLayerGeometry>& resolved_layers,
        const mhs::core::MeshGeometry& mesh, const mhs::core::CellFields& cells,
        const std::vector<mhs::core::SmartMacroModelData>& trained_models,
        const std::vector<ParsedFaceKey>& parsed_face_keys, mhs::core::BcType other_bc_enum, uint16_t other_bc_idx,
        mhs::core::Model& model);

} // namespace mhs::sim
