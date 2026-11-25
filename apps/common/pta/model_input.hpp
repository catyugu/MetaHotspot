#pragma once

#include "model/fom_definition.hpp"

namespace tinyxml2 {
    class XMLElement;
}

namespace mhs::app::pta {

    struct ModelInput {
        mhs::model::FomDefinition model;
        mhs::model::MeshSpec mesh;
    };

    // Parse and normalize the physical model from an already loaded Structure.
    ModelInput read_model(const tinyxml2::XMLElement& structure);

} // namespace mhs::app::pta
