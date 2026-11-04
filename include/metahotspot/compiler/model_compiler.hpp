#pragma once

#include "model/model.hpp"
#include "model/model_definition.hpp"
namespace mhs::compiler {

    /// Convert a model definition into the compiled runtime model.
    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition);

    // External geometry is already in SI. No mesh generation or neighbor discovery.
    // Face signs are derived from face coordinates and owner centers.
    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition, const mhs::model::DiscreteMesh& mesh);

} // namespace mhs::compiler
