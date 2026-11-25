#pragma once

#include "compiler/fom_compiler.hpp"

namespace mhs::compiler {
    // Extraction compiles solid geometry and field laws without evaluating source
    // densities or rejecting Dirichlet T dependence; the caller freezes all laws.
    model::Fom compile_fom_for_extraction(const model::FomDefinition& definition, const model::MeshSpec& mesh);
} // namespace mhs::compiler
