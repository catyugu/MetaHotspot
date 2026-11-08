#pragma once

#include "model/model.hpp"
#include "model/model_definition.hpp"
namespace mhs::compiler {

    /// Evaluate constants and register canonical model functions for expression parsing.
    mhs::expression::SymbolTable build_symbol_table(const mhs::model::ModelDefinition& definition);

    /// Convert a model definition into the compiled runtime model.
    mhs::model::CompiledModel build_model(const mhs::model::ModelDefinition& definition);

} // namespace mhs::compiler
