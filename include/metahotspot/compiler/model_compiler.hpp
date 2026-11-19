#pragma once

#include "model/fom.hpp"
#include "model/model_definition.hpp"
namespace mhs::compiler {

    /// Convert a model definition into the compiled runtime model.
    mhs::model::Fom build_model(const mhs::model::ModelDefinition& definition, const mhs::model::MeshSpec& mesh);

} // namespace mhs::compiler
