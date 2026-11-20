#pragma once

#include "core/matrix.hpp"
#include "model/surface.hpp"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace mhs::model {
    struct EromSource {
        std::string name;
        double default_power = 0; // W
    };

    struct EromOutput {
        std::string name;
        std::optional<Vec3> point; // component coordinates; absent for averaged outputs
    };

    struct EromBoundaryGroup {
        std::string name;
        double minimum, maximum;
        double coefficient = 0; // default environment after embedding
        double ambient = 300;
    };

    struct EromInterface {
        Facet facet;
        double resistance; // m² K/W
        BoundaryGroupId boundary_group;
    };

    struct EromOutline {
        std::string name;
        core::Vector<Facet> facets;
    };

    struct Erom {
        core::Matrix<double> K, C, B, L, Vb;
        core::Vector<EromInterface> interfaces; // indexed by Vb row
        core::Vector<EromBoundaryGroup> boundary_groups; // connectable HTC groups, indexed by BoundaryGroupId
        core::Vector<EromOutline> outline; // untrained groups used only for contact diagnostics
        core::Vector<EromSource> sources;
        core::Vector<EromOutput> outputs;
        double reference_temperature = 300;
        core::Vector<double> uniform_temperature_mode; // q_initial = (T_initial - T_ref) * mode
    };

    void validate_erom(const Erom& erom);
    std::shared_ptr<const Erom> read_erom(const std::filesystem::path& path);
} // namespace mhs::model
