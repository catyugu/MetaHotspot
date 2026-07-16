#pragma once

#include "data/model_definition.hpp"
#include "data/model.hpp"
namespace mhs::sim {

    /// Convert a model definition into the internal SoA representation.
    mhs::core::Model build_model(const mhs::core::ModelDefinition& definition);

} // namespace mhs::sim
