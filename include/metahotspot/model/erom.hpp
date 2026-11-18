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

    enum class EromOutputKind : std::uint8_t { PointTemperature, SourceTemperature };

    struct EromOutput {
        OutputId id;
        std::string name;
        EromOutputKind kind;
        double offset = 300;
        std::optional<Vec3> point;
        SourceId source = core::invalid_id; // volume-average temperature for this source
    };

    struct EromHtc {
        double minimum, maximum;
        double coefficient = 0; // default environment after embedding
        double ambient = 300;
    };

    struct Erom {
        core::Matrix<double> K, C, B, L, Vb;
        core::Vector<double> R;
        core::Vector<Facet> facets; // first Vb.rows() are connectable, the rest describe the outline
        core::Vector<Surface> surfaces;
        core::Vector<std::optional<EromHtc>> htc; // parallel to surfaces
        core::Vector<EromSource> sources;
        core::Vector<EromOutput> outputs;
        double reference_temperature = 300;
        double initial_temperature = 300;
        core::Vector<double> initial_state;
        core::Vector<double> uniform_temperature_mode; // d(q_initial)/d(T_initial), optional
        double training_tolerance = 0; // 0 denotes unspecified
    };

    void validate_erom(const Erom& erom);
    std::shared_ptr<const Erom> read_erom(const std::filesystem::path& path);
} // namespace mhs::model
