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

        mhs::model::BoundaryConditionId compile_thermal_boundary(const mhs::model::ThermalBoundary& condition,
            mhs::core::Vector<mhs::model::BoundaryCondition>& conditions, const mhs::expression::SymbolTable& symbols)
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
            const auto index = static_cast<BoundaryConditionId>(conditions.size());
            conditions.push_back(std::move(result));
            return index;
        }

        mhs::core::Vector<TaggedSurfaceRegion> compile_thermal_boundaries(
            const mhs::core::Vector<mhs::model::ThermalBoundarySpec>& boundaries,
            mhs::core::Vector<mhs::model::BoundaryCondition>& conditions, const mhs::expression::SymbolTable& symbols)
        {
            mhs::core::Vector<TaggedSurfaceRegion> compiled;
            for (const auto& boundary : boundaries) {
                const auto condition = compile_thermal_boundary(boundary.condition, conditions, symbols);
                for (const auto& region : boundary.regions)
                    compiled.push_back({region, condition});
            }
            return compiled;
        }

    } // namespace

    mhs::model::CompiledModel build_model(const mhs::model::ModelDefinition& definition)
    {
        mhs::model::CompiledModel compiled;
        auto append_field = [&](model::DomainId domain, model::FieldQuantity quantity, std::size_t count) {
            core::detail::checked_sparse_size(compiled.layout.state_count + count);
            const auto offset = static_cast<model::DofId>(compiled.layout.state_count);
            compiled.layout.fields.push_back({domain, quantity, offset, count});
            compiled.layout.state_count += count;
        };
        if (definition.domains.empty())
            throw std::invalid_argument("model must contain a FOM or ROM domain");

        auto symbols = build_symbol_table(definition);
        compiled.point_probes = definition.point_probes;

        for (std::size_t domain_id = 0; domain_id < definition.domains.size(); ++domain_id) {
            const auto& value = definition.domains[domain_id];
            if (std::holds_alternative<model::RomDomainDefinition>(value)) {
                auto rom = std::get<model::RomDomainDefinition>(value);
                const auto n = rom.order;
                if (!n || n > std::numeric_limits<core::SparseIndex>::max() / n || rom.K.size() != n * n
                    || rom.C.size() != n * n || rom.f.size() != n
                    || (!rom.initial_state.empty() && rom.initial_state.size() != n))
                    throw std::invalid_argument("ROM operators or initial state have invalid dimensions");
                auto finite = [](double value) { return std::isfinite(value); };
                if (!std::all_of(rom.K.begin(), rom.K.end(), finite)
                    || !std::all_of(rom.C.begin(), rom.C.end(), finite)
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
                for (const auto& obs : rom.observations) {
                    if (obs.weights.size() != n || !std::all_of(obs.weights.begin(), obs.weights.end(), finite)
                        || !finite(obs.offset) || !finite(obs.point.x) || !finite(obs.point.y) || !finite(obs.point.z))
                        throw std::invalid_argument("ROM observation has invalid dimension or values");
                }
                append_field(static_cast<model::DomainId>(domain_id), model::FieldQuantity::ReducedCoordinate, n);
                compiled.domains.emplace_back(std::move(rom));
                continue;
            }

            const auto& domain = std::get<model::FomDomainDefinition>(value);
            compiled.domains.emplace_back(model::FomDomain {});
            auto& model = compiled.fom(static_cast<model::DomainId>(domain_id));
            model.flow_mode = domain.flow_mode;
            // Compile material properties in ModelDefinition::materials order.
            model.material_table.resize(definition.materials.size());

            for (size_t m = 0; m < definition.materials.size(); m++) {
                const auto& mat = definition.materials[m].value;
                auto compile_mat = [&](const std::string& expr) { return mhs::expression::parse(expr, symbols); };
                auto& props = model.material_table[m];
                props.kx = compile_mat(mat.conductivity_x);
                props.ky = compile_mat(mat.conductivity_y);
                props.kz = compile_mat(mat.conductivity_z);
                props.rho = compile_mat(mat.density);
                props.c = compile_mat(mat.specific_heat);
                if (mat.dynamic_viscosity.has_value()) {
                    if (mat.dynamic_viscosity->empty()) {
                        throw std::runtime_error(
                            "fluid material '" + definition.materials[m].name + "' is missing DynamicViscosity");
                    }
                    props.dynamic_viscosity = compile_mat(*mat.dynamic_viscosity);
                }
            }

            for (const auto& source : definition.sources)
                model.heat_source_table.push_back(mhs::expression::parse(source, symbols));
            for (const auto& volume : domain.volumes) {
                if (volume.material >= definition.materials.size())
                    throw std::out_of_range("volume material ID out of range");
                if (volume.source != mhs::core::invalid_id && volume.source >= definition.sources.size())
                    throw std::out_of_range("volume source ID out of range");
                for (const auto& box : volume.geometry)
                    for (int a = 0; a < 3; ++a)
                        if (!std::isfinite(box.min[a]) || !std::isfinite(box.max[a]) || box.min[a] >= box.max[a])
                            throw std::invalid_argument("box bounds must be finite and increasing");
            }

            // Compile boundary patches.
            auto& boundary_conditions = model.boundary_conditions;
            auto compiled_boundaries
                = compile_thermal_boundaries(domain.thermal_boundaries, boundary_conditions, symbols);

            auto default_thermal_boundary
                = compile_thermal_boundary(domain.default_thermal_boundary, boundary_conditions, symbols);

            compile_cartesian_mesh(model, domain.mesh, domain.volumes);
            project_cell_regions(model, domain.volumes);
            for (const auto material : model.cell_material)
                if (material >= model.material_table.size())
                    throw std::out_of_range("cell has no valid material assignment");
            for (const auto& patch :
                project_exterior_surface_regions(model.discrete, compiled_boundaries, default_thermal_boundary))
                model.boundary_faces.push_back({patch.cell, patch.axis, patch.sign,
                    static_cast<mhs::model::BoundaryConditionId>(patch.tag), patch.bounds});

            // Fluid membership and boundary projection are immutable; no pressure solve here.
            auto& fluid = model.fluid;
            fluid.global_to_fluid.assign(model.discrete.cells.size(), mhs::core::invalid_id);
            for (std::size_t c = 0; c < model.discrete.cells.size(); ++c) {
                if (!model.material_table[model.cell_material[c]].dynamic_viscosity)
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
                    const auto component = static_cast<core::EntityIndex>(fluid.component_count++);
                    fluid.component_by_fluid_cell[start] = component;
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
                                fluid.component_by_fluid_cell[adjacent] = component;
                                pending.push_back(adjacent);
                            }
                        }
                    }
                }
                mhs::core::Vector<TaggedSurfaceRegion> regions;
                for (const auto& boundary : domain.fluid_boundaries) {
                    const auto id = model.fluid_conditions.size();
                    model.fluid_conditions.push_back(boundary.condition);
                    for (const auto& region : boundary.regions)
                        regions.push_back({region, id});
                }
                for (const auto& patch : project_exterior_surface_regions(model.discrete, regions))
                    if (fluid.global_to_fluid[patch.cell] != mhs::core::invalid_id)
                        model.fluid_boundaries.push_back({patch.cell, patch.axis, patch.bounds, patch.tag});
            }

            // Only temperature is an unknown in frozen-flow mode. Hydraulic diagnostics
            // are solve workspace data, not fictitious pressure/velocity state blocks.
            const auto n = model.discrete.cells.size();
            append_field(static_cast<model::DomainId>(domain_id), model::FieldQuantity::Temperature, n);

            if (model.flow_mode == model::FlowMode::Stokes && !fluid.fluid_to_global.empty()) {
                fluid.face_conditions.assign(model.discrete.faces.size(), core::invalid_id);
                for (const auto& patch : model.fluid_boundaries) {
                    mhs::model::FaceId matched = core::invalid_id;
                    for (const auto id : model.discrete.incident_faces(patch.cell)) {
                        const auto& face = model.discrete.faces[id];
                        if (face.exterior() && face.axis == patch.axis && face.bounds == patch.bounds) {
                            matched = id;
                            break;
                        }
                    }
                    if (matched == core::invalid_id || fluid.face_conditions[matched] != core::invalid_id)
                        throw std::invalid_argument(
                            "staggered Stokes requires each fluid boundary to cover exactly one whole face");
                    fluid.face_conditions[matched] = static_cast<core::EntityIndex>(patch.condition);
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
                        && fluid.face_conditions[id] != core::invalid_id) {
                        const auto& bc = model.fluid_conditions[fluid.face_conditions[id]];
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
        for (std::size_t d = 0; d < definition.domains.size(); ++d) {
            const auto* value = std::get_if<model::RomDomainDefinition>(&definition.domains[d]);
            if (!value)
                continue;
            const auto& rom = *value;
            const auto domain_id = static_cast<model::DomainId>(d);
            using Bounds = std::array<double, 4>;
            using RegionLaw = std::pair<Bounds, model::BoundaryConditionId>;
            core::Vector<core::Vector<RegionLaw>> per_port(rom.ports.size());
            const auto bounds_of = [](const auto& r) -> Bounds { return {r.a_min, r.a_max, r.b_min, r.b_max}; };
            const auto overlaps = [](const Bounds& a, const Bounds& b) {
                return a[1] > b[0] && b[1] > a[0] && a[3] > b[2] && b[3] > a[2];
            };
            for (const auto& boundary : rom.thermal_boundaries) {
                const auto condition
                    = compile_thermal_boundary(boundary.condition, compiled.rom_boundary_conditions, symbols);
                for (const auto& region : boundary.regions) {
                    const auto rectangle = bounds_of(region.rectangle);
                    if (static_cast<unsigned>(region.axis) > 2 || !std::isfinite(region.coordinate)
                        || !std::all_of(rectangle.begin(), rectangle.end(), [](double v) { return std::isfinite(v); })
                        || rectangle[0] >= rectangle[1] || rectangle[2] >= rectangle[3])
                        throw std::invalid_argument("invalid ROM thermal boundary region");
                    // One declared boundary rectangle may cover multiple
                    // adjacent ROM ports on the same physical plane.
                    double covered = 0.0;
                    for (std::size_t p = 0; p < rom.ports.size(); ++p) {
                        const auto& port = rom.ports[p];
                        if (region.axis != port.region.axis || region.coordinate != port.region.coordinate)
                            continue;
                        const auto outer = bounds_of(port.region.rectangle);
                        Bounds intersection {};
                        intersection[0] = std::max(rectangle[0], outer[0]);
                        intersection[1] = std::min(rectangle[1], outer[1]);
                        intersection[2] = std::max(rectangle[2], outer[2]);
                        intersection[3] = std::min(rectangle[3], outer[3]);
                        if (!overlaps(intersection, outer))
                            continue;
                        for (const auto& connection : definition.connections)
                            if ((connection.a.domain == domain_id && connection.a.port == p)
                                || (connection.b.domain == domain_id && connection.b.port == p))
                                throw std::invalid_argument("ROM thermal boundary overlaps a connected port");
                        per_port[p].push_back({intersection, condition});
                        covered += (intersection[1] - intersection[0]) * (intersection[3] - intersection[2]);
                    }
                    const double requested = (rectangle[1] - rectangle[0]) * (rectangle[3] - rectangle[2]);
                    if (std::abs(covered - requested) > 1e-10 * requested)
                        throw std::invalid_argument("ROM thermal boundary must be covered by exposed ports");
                }
            }

            for (std::size_t p = 0; p < rom.ports.size(); ++p)
                for (std::size_t j = 0; j < rom.ports[p].patches.size(); ++j) {
                    const auto original = bounds_of(rom.ports[p].patches[j].rectangle);
                    core::Vector<double> a {original[0], original[1]}, b {original[2], original[3]};
                    for (const auto& tagged : per_port[p]) {
                        const auto& region = tagged.first;
                        if (!overlaps(original, region))
                            continue;
                        a.push_back(std::max(original[0], region[0]));
                        a.push_back(std::min(original[1], region[1]));
                        b.push_back(std::max(original[2], region[2]));
                        b.push_back(std::min(original[3], region[3]));
                    }
                    auto unique = [](auto& values) {
                        std::sort(values.begin(), values.end());
                        values.erase(std::unique(values.begin(), values.end()), values.end());
                    };
                    unique(a);
                    unique(b);
                    for (std::size_t u = 1; u < a.size(); ++u)
                        for (std::size_t v = 1; v < b.size(); ++v) {
                            const double x = (a[u - 1] + a[u]) / 2, y = (b[v - 1] + b[v]) / 2;
                            auto condition = core::invalid_id;
                            for (const auto& [region, id] : per_port[p])
                                if (x >= region[0] && x <= region[1] && y >= region[2] && y <= region[3])
                                    condition = id; // Last declared law wins.
                            if (condition != core::invalid_id) {
                                model::RomBoundaryTerm term;
                                term.domain = domain_id;
                                term.port = static_cast<model::PortId>(p);
                                term.patch = j;
                                term.condition = condition;
                                term.bounds = {a[u - 1], a[u], b[v - 1], b[v]};
                                compiled.rom_boundary_terms.push_back(term);
                            }
                        }
                }
        }
        return compiled;
    }

} // namespace mhs::compiler
