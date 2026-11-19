#pragma once

#include "model/erom.hpp"
#include "model/fom.hpp"
#include <functional>
#include <map>
#include <memory>
#include <span>

namespace mhs::model {
    struct Component {
        ComponentId id = core::invalid_id;
        Vec3 offset {};
        std::variant<std::shared_ptr<const Fom>, std::shared_ptr<const Erom>> asset;
        std::map<SourceId, std::function<double(double)>> powers; // EROM source schedules, W
        std::map<SurfaceId, BoundaryCondition> boundaries;
    };

    struct ProjectDefinition {
        core::Vector<Component> components;
        // Area-specific contact resistance (m² K/W) keyed by the two component IDs.
        std::map<std::pair<ComponentId, ComponentId>, double> contact_resistance;
    };

    struct Connection {
        std::size_t left, right; // indices in Project::components()
        FacetId left_facet, right_facet;
        std::array<double, 4> overlap; // world tangential bounds
        double area, resistance;
    };

    using ExposedFacets = std::map<FacetId, core::Vector<Facet>>;

    struct Project {
        auto components() const { return std::span<const Component>(components_); }
        auto connections() const { return std::span<const Connection>(connections_); }
        auto warnings() const { return std::span<const std::string>(warnings_); }
        std::size_t dof_count() const { return dof_offsets_.back(); }
        std::size_t dof_offset(std::size_t component) const { return dof_offsets_.at(component); }
        std::size_t dof_size(std::size_t component) const { return dof_offset(component + 1) - dof_offset(component); }

        // Remaining patches, in component-local coordinates. Only contacted facets have entries.
        const ExposedFacets& exposed(std::size_t component) const { return exposed_.at(component); }
        std::optional<Vec3> output_position(ComponentId component, OutputId output) const;

    private:
        friend Project compile_project(ProjectDefinition definition);
        core::Vector<Component> components_;
        core::Vector<Connection> connections_;
        core::Vector<std::string> warnings_;
        core::Vector<std::size_t> dof_offsets_ {0};
        core::Vector<ExposedFacets> exposed_;
    };

    Project compile_project(ProjectDefinition definition);
} // namespace mhs::model
