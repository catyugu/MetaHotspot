#pragma once

#include "core/model.hpp"
#include "core/model_definition.hpp"
namespace mhs::sim {

    /// Convert a model definition into the internal SoA representation.
    mhs::core::Model build_model(const mhs::model::ModelDefinition& definition);

    // External geometry is already in SI. No mesh generation or neighbor discovery.
    mhs::core::Model build_model(const mhs::model::ModelDefinition& definition, const mhs::core::DiscreteMesh& mesh);

} // namespace mhs::sim
