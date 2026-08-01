#pragma once

#include "common/model.hpp"
#include "common/model_definition.hpp"
namespace mhs::sim {

    /// Convert a model definition into the internal SoA representation.
    mhs::core::Model build_model(const mhs::model::ModelDefinition& definition);

} // namespace mhs::sim
