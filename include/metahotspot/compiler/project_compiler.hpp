#pragma once

#include "model/project.hpp"

namespace mhs::compiler {
    model::Project compile_project(model::ProjectDefinition definition);
} // namespace mhs::compiler
