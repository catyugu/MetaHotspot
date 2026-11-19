#pragma once

#include "core/matrix.hpp"
#include "model/surface.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <variant>

namespace mhs::model {
    struct EromSource {
        std::string name;
        double default_power = 0; // W
    };

    struct EromOutput {
        std::string name;
        std::variant<Vec3, SourceId> location = core::invalid_id; // point or source-volume average
    };

    struct EromSurface {
        std::string name;
        double minimum, maximum;
        double coefficient = 0; // default environment after embedding
        double ambient = 300;
    };

    struct EromInterface {
        Facet facet;
        double resistance; // m² K/W
        SurfaceId surface;
    };

    struct EromOutline {
        std::string name;
        core::Vector<Facet> facets;
    };

    struct Erom {
        core::Matrix<double> K, C, B, L, Vb;
        core::Vector<EromInterface> interfaces; // indexed by Vb row
        core::Vector<EromSurface> surfaces; // connectable HTC groups, indexed by SurfaceId
        core::Vector<EromOutline> outline; // untrained groups used only for contact diagnostics
        core::Vector<EromSource> sources;
        core::Vector<EromOutput> outputs;
        double reference_temperature = 300;
        core::Vector<double> uniform_temperature_mode; // q_initial = (T_initial - T_ref) * mode
    };

    void validate_erom(const Erom& erom);
    std::shared_ptr<const Erom> read_erom(const std::filesystem::path& path);
} // namespace mhs::model
