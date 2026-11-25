#pragma once

#include "erom/extraction.hpp"

namespace mhs::app::erom {
    struct Input {
        model::FomDefinition model;
        model::MeshSpec mesh;
        HtcRanges ranges;
        double tolerance = 1e-3;
    };

    Input read_input(const std::filesystem::path& path);
} // namespace mhs::app::erom
