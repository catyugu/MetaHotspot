#pragma once

#include "data/io_structure.hpp"
#include "data/model.hpp"
#include "expr/expr.hpp"
#include "face_key_processor.hpp"
#include "layer_processor.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace mhs::sim::detail {

    struct BuildContext {
        const mhs::core::IOStructure& input;
        mhs::core::SymbolTable symbols;
        double si_scale = 1.0;
    };

    struct MaterialCatalog {
        std::vector<std::string> names;
        std::unordered_map<std::string, size_t> name_to_index;
    };

    struct BoundaryCatalog {
        std::vector<mhs::sim::ParsedFaceKey> explicit_patches;
        mhs::sim::OtherBC fallback;
    };

    BuildContext make_build_context(const mhs::core::IOStructure& input);
    void copy_study_config(mhs::core::Model& model, const mhs::core::IOStructure& input);
    mhs::core::MeshGeometry build_mesh(const BuildContext& context);
    std::vector<mhs::core::ProbePoint> build_observation_points(const BuildContext& context);

    MaterialCatalog build_material_catalog(mhs::core::Model& model, const BuildContext& context,
        const std::vector<mhs::sim::ResolvedLayerGeometry>& resolved_layers);

    std::vector<std::vector<uint16_t>> build_heat_source_catalog(mhs::core::Model& model, const BuildContext& context,
        const std::vector<mhs::sim::ResolvedLayerGeometry>& resolved_layers);

    BoundaryCatalog build_boundary_catalog(mhs::core::Model& model, const BuildContext& context);

} // namespace mhs::sim::detail
