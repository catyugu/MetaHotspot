#pragma once

#include "model/conformal_mesh.hpp"

#include <span>
#include <string>

namespace mhs::app::output {

    void write_vtu(
        const std::string& path, const mhs::model::ConformalMesh& mesh, std::span<const double> cell_temperature);

} // namespace mhs::app::output
