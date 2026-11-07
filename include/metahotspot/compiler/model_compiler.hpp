#pragma once

#include "model/model.hpp"
#include "model/model_definition.hpp"
namespace mhs::compiler {

    /// Evaluate constants and register canonical model functions for expression parsing.
    mhs::expression::SymbolTable build_symbol_table(const mhs::model::ModelDefinition& definition);

    /// Convert a model definition into the compiled runtime model.
    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition);

    // External geometry is already in SI. No mesh generation or neighbor discovery.
    // Face orientation follows minus -> plus along its axis; one side is invalid on exterior patches.
    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition, const mhs::model::DiscreteMesh& mesh);

} // namespace mhs::compiler
