#pragma once

#include "model/erom.hpp"
#include "model/fom.hpp"
#include <functional>
#include <map>
#include <memory>
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

    struct ContactPatch {
        std::size_t left, right; // indices in Project::components()
        FacetId left_facet, right_facet;
        std::array<double, 4> overlap; // world tangential bounds
        double area, resistance;
    };

    using UncoveredPatches = std::map<FacetId, core::Vector<Facet>>;

    struct Project {
        auto components() const { return std::span<const ComponentInstance>(components_); }
        auto contacts() const { return std::span<const ContactPatch>(contacts_); }
        auto warnings() const { return std::span<const std::string>(warnings_); }
        std::size_t dof_count() const { return dof_offsets_.back(); }
        std::size_t dof_offset(std::size_t component) const { return dof_offsets_.at(component); }
        std::size_t dof_size(std::size_t component) const { return dof_offset(component + 1) - dof_offset(component); }

        // Remaining patches, in component-local coordinates. Only contacted facets have entries.
        const UncoveredPatches& uncovered(std::size_t component) const { return uncovered_.at(component); }
        std::optional<Vec3> output_position(ComponentId component, OutputId output) const;

    private:
        friend Project compiler::compile_project(model::ProjectDefinition definition);
        core::Vector<ComponentInstance> components_;
        core::Vector<ContactPatch> contacts_;
        core::Vector<std::string> warnings_;
        core::Vector<std::size_t> dof_offsets_ {0};
        core::Vector<UncoveredPatches> uncovered_;
    };
} // namespace mhs::model
