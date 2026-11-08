#pragma once
#include "model/model.hpp"

namespace mhs::compiler {
    void compile_connections(const model::ModelDefinition& definition, model::CompiledModel& compiled);
}
