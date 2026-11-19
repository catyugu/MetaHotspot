#pragma once

#include "model/erom.hpp"
#include "model/fom.hpp"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>

namespace mhs::model {
    struct Project;
    struct ProjectDefinition;
}

namespace mhs::compiler {
    model::Project compile_project(model::ProjectDefinition definition);
}

namespace mhs::model {
    using BoundaryOverrides = std::map<SurfaceId, BoundaryCondition>;

    struct ComponentInstance {
        ComponentId id = core::invalid_id;
        Vec3 offset {};
        std::variant<std::shared_ptr<const Fom>, std::shared_ptr<const Erom>> asset;
        // EROM source schedules in W replace the selected defaults.
        std::map<SourceId, std::function<double(double)>> powers;
        BoundaryOverrides boundaries; // replaces the law on uncovered surface patches
    };

    struct ProjectDefinition {
        core::Vector<ComponentInstance> components;
        // Area-specific contact resistance (m² K/W) keyed by the two component IDs.
        std::map<std::pair<ComponentId, ComponentId>, double> contact_resistance;
    };

    struct ContactSide {
        std::size_t component; // index in Project::components()
        FacetId facet;
    };

    struct SurfacePatch {
        struct Contact {
            ContactSide left, right;
            double resistance; // m² K/W
        };
        std::array<double, 4> bounds; // world coordinates for contact, component coordinates for exposure
        std::optional<Contact> contact;

        double area() const { return (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]); }
    };

    struct Project {
        auto components() const { return std::span<const ComponentInstance>(components_); }
        auto patches() const { return std::span<const SurfacePatch>(patches_); }
        std::size_t dof_count() const { return dof_offsets_.back(); }
        std::size_t dof_offset(std::size_t component) const { return dof_offsets_.at(component); }
        std::size_t dof_size(std::size_t component) const { return dof_offset(component + 1) - dof_offset(component); }

        // An empty span means the original facet is fully exposed.
        std::span<const std::size_t> facet_patches(std::size_t component, FacetId facet) const;
        std::optional<Vec3> output_position(ComponentId component, OutputId output) const;

    private:
        friend Project compiler::compile_project(model::ProjectDefinition definition);
        core::Vector<ComponentInstance> components_;
        core::Vector<SurfacePatch> patches_;
        core::Vector<std::size_t> dof_offsets_ {0};
        core::Vector<std::map<FacetId, core::Vector<std::size_t>>> facet_patches_;
    };
} // namespace mhs::model
