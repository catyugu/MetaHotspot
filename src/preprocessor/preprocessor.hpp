#pragma once

#include <memory>
#include <string>

#include "model/internal_model.hpp"
#include "model/io_model.hpp"

namespace mhs {

/**
 * @brief Reads IOStructure and converts to internal SoA representation
 */
class Preprocessor {
public:
    Preprocessor() = default;
    ~Preprocessor() = default;

    std::unique_ptr<model::InternalModel> load(const model::IOStructure& ioStructure);

    static void parse_face_key(const std::string& face_key,
                               const model::MeshGeometry& mesh,
                               model::FaceBCFields& face_bcs,
                               BcType bc_type,
                               size_t param_idx);
};

} // namespace mhs
