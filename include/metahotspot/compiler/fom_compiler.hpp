#pragma once

#include "model/fom.hpp"
#include "model/fom_definition.hpp"
namespace mhs::compiler {

    /// Convert a model definition into the compiled runtime model.
    mhs::model::Fom compile_fom(const mhs::model::FomDefinition& definition, const mhs::model::MeshSpec& mesh);

} // namespace mhs::compiler
