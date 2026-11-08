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
        compiled.study = definition.study;
        if (definition.domains.empty() && definition.rom_domains.empty())
            throw std::invalid_argument("model must contain a FOM or ROM domain");

        auto symbols = build_symbol_table(definition);
        compiled.point_probes = definition.point_probes;

        for (std::size_t domain_id = 0; domain_id < definition.domains.size(); ++domain_id) {
            const auto& domain = definition.domains[domain_id];
            compiled.domains.emplace_back();
            auto& model = compiled.domains.back();
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
            mhs::model::FieldLayout temperature;
            temperature.domain = static_cast<mhs::model::DomainId>(domain_id);
            const auto n = model.discrete.cells.size();
            mhs::core::detail::checked_sparse_size(compiled.layout.state_count + n);
            temperature.entities.resize(n);
            temperature.dofs.resize(n);
            for (std::size_t c = 0; c < n; ++c) {
                temperature.entities[c] = static_cast<mhs::model::CellId>(c);
                temperature.dofs[c] = static_cast<mhs::model::DofId>(compiled.layout.state_count + c);
            }
            compiled.layout.state_count += n;
            compiled.layout.fields.push_back(std::move(temperature));
        }
        // A ROM has algebraic coordinates and physical boundary maps but no
        // Cartesian cell mesh or fabricated material table.
        for (std::size_t r = 0; r < definition.rom_domains.size(); ++r) {
            auto rom = definition.rom_domains[r];
            const auto n = rom.order;
            if (!n || n > std::numeric_limits<core::SparseIndex>::max() / n
                || rom.K.size() != n * n || rom.C.size() != n * n || rom.f.size() != n
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
            for (const auto& obs : rom.observations) {
                if (obs.weights.size() != n || !std::all_of(obs.weights.begin(), obs.weights.end(), finite)
                    || !finite(obs.offset) || !finite(obs.point.x) || !finite(obs.point.y)
                    || !finite(obs.point.z))
                    throw std::invalid_argument("ROM observation has invalid dimension or values");
            }
            const auto global_id = static_cast<model::DomainId>(compiled.domains.size() + r);
            model::FieldLayout field;
            field.domain = global_id;
            field.quantity = model::FieldQuantity::ReducedCoordinate;
            field.support = model::EntitySupport::Reduced;
            core::detail::checked_sparse_size(compiled.layout.state_count + n);
            field.entities.resize(n);
            field.dofs.resize(n);
            for (std::size_t j = 0; j < n; ++j) {
                field.entities[j] = static_cast<core::EntityIndex>(j);
                field.dofs[j] = static_cast<model::DofId>(compiled.layout.state_count + j);
            }
            compiled.layout.state_count += n;
            compiled.layout.fields.push_back(std::move(field));
            compiled.rom_domains.push_back(std::move(rom));
        }
        compile_connections(definition, compiled);
        return compiled;
    }

} // namespace mhs::compiler
