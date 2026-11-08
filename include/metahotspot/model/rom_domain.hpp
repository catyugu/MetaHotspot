#pragma once

#include "model/model_definition.hpp"

namespace mhs::model {
    // Immutable compiled snapshot: contains only reduced operators, ports and
    // observation mappings; no cell mesh or reconstructed FOM volume data.
    using RomDomain = RomDomainDefinition;
} // namespace mhs::model
