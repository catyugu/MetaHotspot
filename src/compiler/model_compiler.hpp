#pragma once

#include "core/model.hpp"
#include "core/model_definition.hpp"
namespace mhs::sim {

    /// Convert a model definition into the internal SoA representation.
    mhs::core::Model build_model(const mhs::model::ModelDefinition& definition);

} // namespace mhs::sim
