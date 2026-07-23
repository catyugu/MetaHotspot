#pragma once

#include "runtime/model.hpp"
#include "solver/operator_assembly.hpp"

namespace mhs::sim {

    OperatorContribution assemble_cell_domain(const mhs::core::Model& model, const AssembleContext& context);

} // namespace mhs::sim
