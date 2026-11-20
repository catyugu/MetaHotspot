#pragma once

#include "model/erom.hpp"
#include "model/fom.hpp"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <variant>

namespace mhs::model {
    struct Project;
    struct ProjectDefinition;
}

namespace mhs::compiler {
    model::Project compile_project(model::ProjectDefinition definition);
}

namespace mhs::model {
    struct FomInstance {
        std::shared_ptr<const Fom> asset;
        // FOM BoundaryGroupId. Unset entries retain the asset law.
        core::Vector<std::optional<BoundaryCondition>> boundaries {};
    };

    struct EromInstance {
        EromAsset asset;
        core::Vector<std::function<double(double)>> powers {}; // EROM SourceId; empty retains the default power
        core::Vector<std::optional<Convection>> htc {}; // EROM BoundaryGroupId; unset retains the default HTC
    };

    struct ComponentInstance {
        Vec3 offset {};
        std::variant<FomInstance, EromInstance> model;
    };

    struct ProjectDefinition {
        core::Vector<ComponentInstance> components;
        // Area-specific contact resistance (m² K/W) keyed by the two component indices.
        std::map<std::pair<std::size_t, std::size_t>, double> contact_resistance;
    };

    // FacetId is local to a FOM mesh; EromInterfaceId is local to an EROM asset.
    using ComponentFace = std::variant<FacetId, EromInterfaceId>;

    struct ContactSide {
        std::size_t component_index; // index in Project::components()
        ComponentFace face;
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
        const FacePartition* face_partition(std::size_t component_index, ComponentFace face) const;
        std::optional<Vec3> output_position(std::size_t component_index, OutputId output) const;

    private:
        friend Project compiler::compile_project(model::ProjectDefinition definition);
        core::Vector<ComponentInstance> components_;
        core::Vector<Contact> contacts_;
        core::Vector<std::size_t> dof_offsets_ {0};
        core::Vector<std::map<ComponentFace, FacePartition>> face_partitions_;
    };
} // namespace mhs::model
