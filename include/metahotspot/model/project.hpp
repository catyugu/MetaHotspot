#pragma once

#include "model/erom.hpp"
#include "model/fom.hpp"
#include <functional>
#include <limits>
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
    using BoundaryOverrides = core::Vector<std::optional<BoundaryCondition>>;

    struct ComponentInstance {
        Vec3 offset {};
        std::variant<std::shared_ptr<const Fom>, std::shared_ptr<const Erom>> asset;
        // Indexed by SourceId; empty schedules supply zero power.
        core::Vector<std::function<double(double)>> powers;
        BoundaryOverrides boundaries; // indexed by BoundaryGroupId; unset entries retain asset laws
    };

    struct ProjectDefinition {
        core::Vector<ComponentInstance> components;
        // Area-specific contact resistance (m² K/W) keyed by the two component indices.
        std::map<std::pair<std::size_t, std::size_t>, double> contact_resistance;
    };

    struct ContactSide {
        std::size_t component_index; // index in Project::components()
        core::EntityIndex face_index; // FOM facet index or EROM interface index
    };

    struct Contact {
        ContactSide left, right;
        double area;
        double resistance; // m² K/W
    };

    struct FacePatch {
        static constexpr auto no_contact = std::numeric_limits<std::size_t>::max();

        Rect bounds; // component coordinates
        std::size_t contact = no_contact; // index in Project::contacts()

        bool is_contact() const { return contact != no_contact; }
    };

    struct Project {
        auto components() const { return std::span<const ComponentInstance>(components_); }
        auto contacts() const { return std::span<const Contact>(contacts_); }
        std::size_t dof_count() const { return dof_offsets_.back(); }
        std::size_t dof_offset(std::size_t component_index) const { return dof_offsets_.at(component_index); }
        std::size_t dof_size(std::size_t component_index) const
        {
            return dof_offset(component_index + 1) - dof_offset(component_index);
        }

        // Contact patches precede exposed patches. An empty span means the whole face is exposed.
        std::span<const FacePatch> face_patches(std::size_t component_index, core::EntityIndex face_index) const;
        std::optional<Vec3> output_position(std::size_t component_index, OutputId output) const;

    private:
        friend Project compiler::compile_project(model::ProjectDefinition definition);
        core::Vector<ComponentInstance> components_;
        core::Vector<Contact> contacts_;
        core::Vector<std::size_t> dof_offsets_ {0};
        core::Vector<std::map<core::EntityIndex, core::Vector<FacePatch>>> face_patches_;
    };
} // namespace mhs::model
