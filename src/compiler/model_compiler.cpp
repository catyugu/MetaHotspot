#include "compiler/model_compiler.hpp"
#include "compiler/connections.hpp"
#include "compiler/geometry_compiler.hpp"
#include "compiler/model_functions.hpp"
#include "core/csr_matrix.hpp"
#include "core/vector.hpp"
#include "expression/expression.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mhs::compiler {
    namespace {
        template <typename... Ts> struct overloaded : Ts... {
            using Ts::operator()...;
        };
        template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

        mhs::model::BoundaryCondition compile_thermal_boundary(
            const mhs::model::ThermalBoundary& condition, const mhs::expression::SymbolTable& symbols)
        {
            using namespace mhs::core;
            using namespace mhs::model;
            using namespace mhs::expression;
            auto compile = [&](const auto& expression) { return parse(expression, symbols); };
            auto result = std::visit(
                overloaded {
                    [&](const mhs::model::DirichletBoundary& value) -> BoundaryCondition {
                        auto temperature = compile(value.temperature);
                        // A trace that prescribes itself is not a Dirichlet
                        // condition; require an explicit temperature.
                        if (temperature.depends_on_temperature())
                            throw std::invalid_argument("Dirichlet boundary temperature must not depend on T");
                        return Dirichlet {std::move(temperature)};
                    },
                    [&](const mhs::model::NeumannBoundary& value) -> BoundaryCondition {
                        return Neumann {compile(value.heat_flux)};
                    },
                    [&](const mhs::model::ConvectionBoundary& value) -> BoundaryCondition {
                        return Convection {compile(value.coefficient), compile(value.ambient_temperature)};
                    },
                },
                condition);
            return result;
        }

    } // namespace

    mhs::model::CompiledModel build_model(const mhs::model::ModelDefinition& definition)
    {
        mhs::model::CompiledModel compiled;
        auto& compiled_library = compiled.library();
        auto append_field = [&](model::DomainId domain, model::FieldQuantity quantity, std::size_t count) {
            core::detail::checked_sparse_size(compiled.layout.state_count + count);
            const auto offset = static_cast<model::DofId>(compiled.layout.state_count);
            compiled.layout.fields.push_back({domain, quantity, offset, count});
            compiled.layout.state_count += count;
        };
        if (definition.components.empty())
            throw std::invalid_argument("model must contain at least one component");
        if (!definition.library_handle)
            throw std::invalid_argument("model requires a Library");
        const auto& library = definition.library();
        auto symbols = build_symbol_table(definition);
        compiled_library.material_table.resize(library.materials().size());
        for (std::size_t m = 0; m < library.materials().size(); ++m) {
            const auto& mat = library.materials()[m].value;
            auto compile = [&](const std::string& expr) { return mhs::expression::parse(expr, symbols); };
            auto& props = compiled_library.material_table[m];
            props.kx = compile(mat.conductivity_x);
            props.ky = compile(mat.conductivity_y);
            props.kz = compile(mat.conductivity_z);
            props.rho = compile(mat.density);
            props.c = compile(mat.specific_heat);
            if (mat.dynamic_viscosity) {
                if (mat.dynamic_viscosity->empty())
                    throw std::runtime_error(
                        "fluid material '" + library.materials()[m].name + "' is missing DynamicViscosity");
                props.dynamic_viscosity = compile(*mat.dynamic_viscosity);
            }
        }
        for (const auto& source : library.sources()) {
            compiled_library.heat_source_table.push_back(mhs::expression::parse(source, symbols));
        }
        for (const auto& boundary : library.thermal_boundaries())
            compiled_library.boundary_conditions.push_back(compile_thermal_boundary(boundary, symbols));
        compiled_library.fluid_conditions = library.fluid_boundaries();
        for (std::size_t domain_id = 0; domain_id < definition.components.size(); ++domain_id) {
            const auto& component = definition.components[domain_id];
            if (const auto* value = std::get_if<model::RomDomainDefinition>(&component.physics)) {
                auto rom = *value;
                const auto n = rom.order;
                if (!n || n > std::numeric_limits<core::SparseIndex>::max() / n || rom.K.size() != n * n
                    || rom.C.size() != n * n || rom.f.size() != n
                    || (!rom.initial_state.empty() && rom.initial_state.size() != n))
                    throw std::invalid_argument("ROM operators or initial state have invalid dimensions");
                auto finite = [](double value) { return std::isfinite(value); };
                if (!std::all_of(rom.K.begin(), rom.K.end(), finite) || !std::all_of(rom.C.begin(), rom.C.end(), finite)
                    || !std::all_of(rom.f.begin(), rom.f.end(), finite)
                    || !std::all_of(rom.initial_state.begin(), rom.initial_state.end(), finite))
                    throw std::invalid_argument("ROM operators and initial state must be finite");
                if (rom.initial_state.empty())
                    rom.initial_state.assign(n, 0.0);
                for (std::size_t j = 0; j < rom.affine_terms.size(); ++j) {
                    const auto& term = rom.affine_terms[j];
                    if (term.name.empty() || !finite(term.value) || term.K.size() != n * n || term.f.size() != n
                        || !std::all_of(term.K.begin(), term.K.end(), finite)
                        || !std::all_of(term.f.begin(), term.f.end(), finite))
                        throw std::invalid_argument("ROM affine term has invalid dimension or values");
                    for (std::size_t previous = 0; previous < j; ++previous)
                        if (rom.affine_terms[previous].name == term.name)
                            throw std::invalid_argument("ROM parameter names must be unique within a domain");
                }
                for (const auto& obs : component.observations) {
                    const auto& point = obs.point;
                    if (obs.weights.size() != n || !std::all_of(obs.weights.begin(), obs.weights.end(), finite)
                        || !finite(obs.offset) || point.name.empty() || !finite(point.x) || !finite(point.y)
                        || !finite(point.z))
                        throw std::invalid_argument("reduced observation has invalid dimension or values");
                }
                append_field(static_cast<model::DomainId>(domain_id), model::FieldQuantity::ReducedCoordinate, n);
                compiled.components.push_back({std::move(rom), component.ports, component.observations});
                continue;
            }

            const auto& domain = std::get<model::FomDomainDefinition>(component.physics);
            for (const auto& obs : component.observations)
                if (!obs.weights.empty() || obs.offset != 0.0 || obs.point.name.empty() || !std::isfinite(obs.point.x)
                    || !std::isfinite(obs.point.y) || !std::isfinite(obs.point.z))
                    throw std::invalid_argument("Cartesian observation requires only a finite physical point");
            compiled.components.push_back({model::FomDomain {}, component.ports, component.observations});
            auto& model = compiled.fom(static_cast<model::DomainId>(domain_id));
            model.flow_mode = domain.flow_mode;
            for (const auto& volume : domain.volumes) {
                (void)library.material(volume.material);
                if (volume.source != core::invalid_id)
                    (void)library.source(volume.source);
                for (const auto& box : volume.geometry)
                    for (int a = 0; a < 3; ++a)
                        if (!std::isfinite(box.min[a]) || !std::isfinite(box.max[a]) || box.min[a] >= box.max[a])
                            throw std::invalid_argument("box bounds must be finite and increasing");
            }

            for (const auto& boundary : component.thermal_boundaries)
                (void)library.thermal_boundary(boundary.condition);
            if (domain.default_thermal_boundary != core::invalid_id)
                (void)library.thermal_boundary(domain.default_thermal_boundary);

            compile_cartesian_mesh(model, domain.mesh, domain.volumes);
            project_cell_regions(model, domain.volumes);
            for (const auto material : model.cell_material)
                if (material >= compiled_library.material_table.size())
                    throw std::out_of_range("cell has no valid material assignment");

            // Fluid membership and boundary projection are immutable; no pressure solve here.
            auto& fluid = model.fluid;
            fluid.global_to_fluid.assign(model.discrete.cells.size(), mhs::core::invalid_id);
            for (std::size_t c = 0; c < model.discrete.cells.size(); ++c) {
                if (!compiled_library.material_table[model.cell_material[c]].dynamic_viscosity)
                    continue;
                fluid.global_to_fluid[c] = static_cast<mhs::model::FluidCellId>(fluid.fluid_to_global.size());
                fluid.fluid_to_global.push_back(static_cast<mhs::model::CellId>(c));
            }
            if (fluid.fluid_to_global.empty())
                fluid.global_to_fluid.clear();
            else {
                // Determine hydraulic components once from the compiled face adjacency.
                fluid.component_by_fluid_cell.assign(fluid.fluid_to_global.size(), core::invalid_id);
                for (std::size_t start = 0; start < fluid.fluid_to_global.size(); ++start) {
                    if (fluid.component_by_fluid_cell[start] != core::invalid_id)
                        continue;
                    const auto fluid_component = static_cast<core::EntityIndex>(fluid.component_count++);
                    fluid.component_by_fluid_cell[start] = fluid_component;
                    core::Vector<model::FluidCellId> pending {static_cast<model::FluidCellId>(start)};
                    while (!pending.empty()) {
                        const auto current = pending.back();
                        pending.pop_back();
                        const auto cell = fluid.fluid_to_global[current];
                        for (const auto face_id : model.discrete.incident_faces(cell)) {
                            const auto& face = model.discrete.faces[face_id];
                            if (face.exterior())
                                continue;
                            const auto neighbor = face.minus == cell ? face.plus : face.minus;
                            const auto adjacent = fluid.global_to_fluid[neighbor];
                            if (adjacent != core::invalid_id
                                && fluid.component_by_fluid_cell[adjacent] == core::invalid_id) {
                                fluid.component_by_fluid_cell[adjacent] = fluid_component;
                                pending.push_back(adjacent);
                            }
                        }
                    }
                }
                mhs::core::Vector<TaggedSurfaceRegion> regions;
                for (const auto& boundary : domain.fluid_boundaries) {
                    (void)library.fluid_boundary(boundary.condition);
                    const auto condition = static_cast<model::FluidBoundaryIndex>(boundary.condition);
                    const auto assignment = static_cast<model::Id>(model.fluid_assignments.size());
                    model.fluid_assignments.push_back({condition, 0.0});
                    for (const auto& region : boundary.regions)
                        regions.push_back({region, assignment});
                }
                for (const auto& patch : project_exterior_surface_regions(model.discrete, regions))
                    if (fluid.global_to_fluid[patch.cell] != core::invalid_id) {
                        model.fluid_boundaries.push_back({patch.cell, patch.axis, patch.bounds, patch.tag});
                        model.fluid_assignments[patch.tag].area += surface_area(patch.bounds);
                    }
            }

            // Only temperature is an unknown in frozen-flow mode. Hydraulic diagnostics
            // are solve workspace data, not fictitious pressure/velocity state blocks.
            const auto n = model.discrete.cells.size();
            append_field(static_cast<model::DomainId>(domain_id), model::FieldQuantity::Temperature, n);

            if (model.flow_mode == model::FlowMode::Stokes && !fluid.fluid_to_global.empty()) {
                fluid.face_assignments.assign(model.discrete.faces.size(), core::invalid_id);
                for (const auto& patch : model.fluid_boundaries) {
                    mhs::model::FaceId matched = core::invalid_id;
                    for (const auto id : model.discrete.incident_faces(patch.cell)) {
                        const auto& face = model.discrete.faces[id];
                        if (face.exterior() && face.axis == patch.axis && face.bounds == patch.bounds) {
                            matched = id;
                            break;
                        }
                    }
                    if (matched == core::invalid_id || fluid.face_assignments[matched] != core::invalid_id)
                        throw std::invalid_argument(
                            "staggered Stokes requires each fluid boundary to cover exactly one whole face");
                    fluid.face_assignments[matched] = patch.assignment;
                }
                core::Vector<bool> pressure_references(fluid.component_count, false);
                for (std::size_t id = 0; id < model.discrete.faces.size(); ++id) {
                    const auto& face = model.discrete.faces[id];
                    const auto minus = face.minus;
                    const auto i = minus == core::invalid_id ? core::invalid_id : fluid.global_to_fluid[minus];
                    const auto j = face.plus == core::invalid_id ? core::invalid_id : fluid.global_to_fluid[face.plus];
                    if (i != core::invalid_id && j != core::invalid_id) {
                        fluid.velocity_faces.push_back(static_cast<model::FaceId>(id));
                    }
                    else if (face.exterior() && (i != core::invalid_id || j != core::invalid_id)
                        && fluid.face_assignments[id] != core::invalid_id) {
                        const auto assignment = fluid.face_assignments[id];
                        const auto condition = model.fluid_assignments[assignment].condition;
                        const auto& bc = compiled_library.fluid_conditions[condition];
                        if (std::holds_alternative<model::PressureInlet>(bc)
                            || std::holds_alternative<model::PressureOutlet>(bc)) {
                            fluid.velocity_faces.push_back(static_cast<model::FaceId>(id));
                            const auto fluid_cell = i != core::invalid_id ? i : j;
                            pressure_references[fluid.component_by_fluid_cell[fluid_cell]] = true;
                        }
                    }
                }
                if (std::find(pressure_references.begin(), pressure_references.end(), false)
                    != pressure_references.end())
                    throw std::invalid_argument(
                        "staggered Stokes requires a pressure boundary for every connected fluid component");

                append_field(static_cast<model::DomainId>(domain_id), model::FieldQuantity::Pressure,
                    fluid.fluid_to_global.size());
                append_field(static_cast<model::DomainId>(domain_id), model::FieldQuantity::Velocity,
                    fluid.velocity_faces.size());
            }
        }
        compile_connections(definition, compiled);

        // Boundary laws on exposed ROM ports are assembled through their
        // physical trace maps, not through HTC-specific affine ROM operators.
        // Subdivide each port patch by the ordinary surface-condition regions.
        for (std::size_t d = 0; d < definition.components.size(); ++d) {
            const auto* value = std::get_if<model::RomDomainDefinition>(&definition.components[d].physics);
            if (!value)
                continue;
            const auto& ports = definition.components[d].ports;
            const auto& boundaries = definition.components[d].thermal_boundaries;
            const auto domain_id = static_cast<model::DomainId>(d);
            core::Vector<core::Vector<TaggedBounds>> per_port(ports.size());
            for (const auto& boundary : boundaries) {
                (void)library.thermal_boundary(boundary.condition);
                const auto condition = static_cast<model::ThermalBoundaryIndex>(boundary.condition);
                for (const auto& region : boundary.regions) {
                    const auto rectangle = surface_bounds(region.rectangle);
                    validate_surface_region(region);
                    // One declared boundary rectangle may cover multiple
                    // adjacent ROM ports on the same physical plane.
                    double covered = 0.0;
                    for (std::size_t p = 0; p < ports.size(); ++p) {
                        const auto& port = ports[p];
                        if (region.axis != port.region.axis || region.coordinate != port.region.coordinate)
                            continue;
                        const auto outer = surface_bounds(port.region.rectangle);
                        const auto intersection = intersect_bounds(rectangle, outer);
                        if (!positive_bounds(intersection))
                            continue;
                        for (const auto& connection : definition.connections)
                            if ((connection.a.domain == domain_id && connection.a.port == p)
                                || (connection.b.domain == domain_id && connection.b.port == p))
                                throw std::invalid_argument("ROM thermal boundary overlaps a connected port");
                        per_port[p].push_back({intersection, condition});
                        covered += surface_area(intersection);
                    }
                    const double requested = surface_area(rectangle);
                    if (std::abs(covered - requested) > 1e-10 * requested)
                        throw std::invalid_argument("ROM thermal boundary must be covered by exposed ports");
                }
            }

            for (std::size_t p = 0; p < ports.size(); ++p)
                for (std::size_t j = 0; j < ports[p].patches.size(); ++j) {
                    const auto original = surface_bounds(ports[p].patches[j].rectangle);
                    for (const auto& tagged : partition_surface_bounds(original, per_port[p]))
                        compiled.rom_boundary_terms.push_back(
                            {domain_id, static_cast<model::PortId>(p), j, tagged.tag, tagged.bounds});
                }
        }
        return compiled;
    }

} // namespace mhs::compiler
