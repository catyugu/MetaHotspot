#pragma once

#include "model/erom.hpp"
#include "model/fom_definition.hpp"
#include <map>

namespace mhs::app::erom {
    struct HtcRange {
        double minimum, maximum;
    };
    using HtcRanges = std::map<model::BoundaryGroupId, HtcRange>;

    struct Statistics {
        int shifts = 0;
        int response_solves = 0;
        int response_amg_setups = 0;
        std::size_t cg_iterations = 0;
        double lambda_min = 0, lambda_max = 0;
        double output_error = 0;
    };

    struct Extraction {
        model::EromAsset asset;
        Statistics statistics;
    };

    // Group IDs are indices into thermal_boundaries, followed by the default group.
    // Trained facets must lie on the faces of the active FOM bounding box.
    // HTC ranges are positive and in W/(m^2 K). Tolerance is dimensionless and in (0, 1).
    Extraction extract(const model::FomDefinition& definition, const model::MeshSpec& mesh, const HtcRanges& htc,
        double tolerance = 1e-3);
    void write_erom(const std::filesystem::path& path, const model::EromAsset& asset);
} // namespace mhs::app::erom
