#pragma once

#include "core/matrix.hpp"
#include "model/surface.hpp"
#include <filesystem>
#include <memory>
#include <optional>

namespace mhs::model {
    struct EromSource {
        SourceId id;
        std::string name;
        double default_power = 0; // W
    };

    struct EromOutput {
        OutputId id;
        std::string name;
        std::optional<Vec3> point;
        SourceId source = core::invalid_id; // volume-average temperature for this source
    };

    struct EromSurfaceTraining {
        double minimum, maximum;
        double coefficient = 0; // default environment after embedding
        double ambient = 300;
    };

    struct Erom {
        core::Matrix<double> K, C, B, L, Vb;
        core::Vector<double> R;
        core::Vector<Facet> facets; // first Vb.rows() are connectable, the rest describe the outline
        core::Vector<Surface> surfaces;
        core::Vector<std::optional<EromSurfaceTraining>> surface_training; // parallel to surfaces
        core::Vector<EromSource> sources;
        core::Vector<EromOutput> outputs;
        double reference_temperature = 300;
        core::Vector<double> uniform_temperature_mode; // q_initial = (T_initial - T_ref) * mode
    };

    void validate_erom(const Erom& erom);
    std::shared_ptr<const Erom> read_erom(const std::filesystem::path& path);
} // namespace mhs::model
