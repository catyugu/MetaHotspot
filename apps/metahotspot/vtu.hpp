#pragma once

#include "model/model.hpp"

#include <span>
#include <string>

namespace mhs::app {

    void write_vtu(
        const std::string& path, const mhs::model::DiscreteMesh& mesh, std::span<const double> cell_temperature);

} // namespace mhs::app
