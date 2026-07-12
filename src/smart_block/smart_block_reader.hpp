#pragma once

#include <string>

#include "data/model.hpp"

namespace mhs::core {

    /// Load a trained SmartMacro model from disk.
    ///
    /// Reads the tiny XML + sibling .data binary (see SmartMacroModelData for the
    /// binary layout). Throws on any I/O or format error.
    SmartMacroModelData read_smart_macro_model(const std::string& xml_path);

} // namespace mhs::core
