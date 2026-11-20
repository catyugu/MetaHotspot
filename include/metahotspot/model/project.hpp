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
    using BoundaryOverrides = std::map<BoundaryGroupId, BoundaryCondition>;

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
        std::size_t component_index; // index in Project::components()
        core::EntityIndex face_index; // FOM facet index or EROM interface index
    };

    struct SurfacePatch {
        struct Contact {
            ContactSide left, right;
            double resistance; // m² K/W
        };
        Rect bounds; // world coordinates for both contact and exposure
        std::optional<Contact> contact;

        double area() const { return rect_area(bounds); }
    };

    struct Project {
        auto components() const { return std::span<const ComponentInstance>(components_); }
        auto patches() const { return std::span<const SurfacePatch>(patches_); }
        std::size_t dof_count() const { return dof_offsets_.back(); }
        std::size_t dof_offset(std::size_t component_index) const { return dof_offsets_.at(component_index); }
        std::size_t dof_size(std::size_t component_index) const
        {
            return dof_offset(component_index + 1) - dof_offset(component_index);
        }

        // An empty span means the original FOM facet or EROM interface is fully exposed.
        std::span<const std::size_t> face_patches(std::size_t component_index, core::EntityIndex face_index) const;
        std::optional<Vec3> output_position(ComponentId component_id, OutputId output) const;

    private:
        friend Project compiler::compile_project(model::ProjectDefinition definition);
        core::Vector<ComponentInstance> components_;
        core::Vector<SurfacePatch> patches_;
        core::Vector<std::size_t> dof_offsets_ {0};
        core::Vector<std::map<core::EntityIndex, core::Vector<std::size_t>>> face_patches_;
    };
} // namespace mhs::model
