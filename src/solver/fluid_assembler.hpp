#pragma once

#include "runtime/model.hpp"
#include "solver/operator_assembly.hpp"

namespace mhs::sim::fluid {

    /// Return fluid-only corrections in global state coordinates.
    OperatorContribution assemble_operator(const mhs::core::Model& model, const AssembleContext& context);

} // namespace mhs::sim::fluid
