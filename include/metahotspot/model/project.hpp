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
    using BoundaryOverrides = core::Vector<std::optional<BoundaryCondition>>;

    struct ComponentInstance {
        Vec3 offset {};
        std::variant<std::shared_ptr<const Fom>, std::shared_ptr<const Erom>> asset;
        // Indexed by SourceId; empty schedules retain the EROM defaults.
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
        Rect bounds; // world coordinates
        double resistance; // m² K/W

        double area() const { return rect_area(bounds); }
    };

    struct FacePartition {
        core::Vector<std::size_t> contacts; // indices in Project::contacts()
        core::Vector<Rect> exposed; // component coordinates; empty means fully covered
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

        // A null partition means the original face is fully exposed.
        const FacePartition* face_partition(std::size_t component_index, core::EntityIndex face_index) const;
        std::optional<Vec3> output_position(std::size_t component_index, OutputId output) const;

    private:
        friend Project compiler::compile_project(model::ProjectDefinition definition);
        core::Vector<ComponentInstance> components_;
        core::Vector<Contact> contacts_;
        core::Vector<std::size_t> dof_offsets_ {0};
        core::Vector<std::map<core::EntityIndex, FacePartition>> face_partitions_;
    };
} // namespace mhs::model
