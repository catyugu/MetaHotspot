#pragma once

#include "model/erom.hpp"
#include <filesystem>
#include <memory>

namespace mhs::io {
    // Read and validate a native EROM XML asset.
    std::shared_ptr<const model::Erom> read_erom(const std::filesystem::path& path);
} // namespace mhs::io
