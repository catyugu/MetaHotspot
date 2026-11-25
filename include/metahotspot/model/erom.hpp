#pragma once

#include "core/matrix.hpp"
#include "model/surface.hpp"
#include <optional>
#include <string>

namespace mhs::model {
    struct EromSource {
        std::string name;
    };

    struct EromOutput {
        std::string name;
        std::optional<Vec3> point; // component coordinates; absent for averaged outputs
    };

    struct EromBoundaryGroup {
        std::string name;
        double minimum, maximum;
    };

    struct EromInterface {
        Facet facet;
        double resistance; // m² K/W
        BoundaryGroupId boundary_group;
        double default_coefficient = 0; // W/(m² K)
        double default_ambient = 300; // K
    };

    struct EromOutline {
        std::string name;
        core::Vector<Facet> facets;
    };

    struct Erom {
        core::Matrix<double> K, C, B, L, Vb;
        core::Vector<double> f; // projected fixed thermal load
        core::Vector<double> output_offset; // affine temperature reconstruction offset
        core::Vector<EromInterface> interfaces; // indexed by Vb row
        core::Vector<EromBoundaryGroup> boundary_groups; // connectable HTC groups, indexed by BoundaryGroupId
        core::Vector<EromOutline> outline; // untrained groups used only for contact diagnostics
        core::Vector<EromSource> sources;
        core::Vector<EromOutput> outputs;
        double reference_temperature = 300;
        core::Vector<double> uniform_temperature_mode; // q_initial = (T_initial - T_ref) * mode
    };

    void validate_erom(const Erom& erom);
} // namespace mhs::model
