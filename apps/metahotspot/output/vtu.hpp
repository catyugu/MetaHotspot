#pragma once

#include "model/mesh.hpp"

#include <span>
#include <string>

namespace mhs::app::output {

    void write_vtu(const std::string& path, const mhs::model::Mesh& mesh, std::span<const double> cell_temperature);

} // namespace mhs::app::output
