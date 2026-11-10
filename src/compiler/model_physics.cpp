#include "compiler/model_physics.hpp"
#include "compiler/model_functions.hpp"
#include "expression/expression.hpp"

#include <cmath>
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

        void validate_surface_regions(const mhs::model::ModelDefinition& definition)
        {
            auto check = [](const mhs::model::SurfaceRegion& region) {
                const auto& r = region.rectangle;
                if (static_cast<unsigned>(region.axis) > 2 || !std::isfinite(region.coordinate)
                    || !std::isfinite(r.a_min) || !std::isfinite(r.a_max) || !std::isfinite(r.b_min)
                    || !std::isfinite(r.b_max) || r.a_min >= r.a_max || r.b_min >= r.b_max)
                    throw std::invalid_argument("surface region requires a valid axis and finite, increasing bounds");
            };
            for (const auto& boundary : definition.thermal_boundaries)
                for (const auto& region : boundary.regions)
                    check(region);
            for (const auto& boundary : definition.fluid_boundaries)
                for (const auto& region : boundary.regions)
                    check(region);
        }

    } // namespace

    CompiledPhysics compile_model_physics(mhs::model::Model& model, const mhs::model::ModelDefinition& definition)
    {
        if (!std::isfinite(definition.reference_temperature))
            throw std::invalid_argument("reference temperature must be finite");
        validate_surface_regions(definition);

        CompiledPhysics result;
        model.reference_temperature = definition.reference_temperature;

        mhs::expression::SymbolTable symbols;
        register_all_functions(symbols, definition.functions);

        // Compile material properties in ModelDefinition::materials order.
        model.material_table.resize(definition.materials.size());
        result.initial_viscosity.resize(definition.materials.size());

        for (size_t m = 0; m < definition.materials.size(); m++) {
            const auto& mat = definition.materials[m].value;
            auto compile_mat = [&](const std::string& expr) { return mhs::expression::parse(expr, symbols); };
            auto& props = model.material_table[m];
            props.kind = mat.kind;
            props.kx = compile_mat(mat.conductivity_x);
            props.ky = compile_mat(mat.conductivity_y);
            props.kz = compile_mat(mat.conductivity_z);
            props.rho = compile_mat(mat.density);
            props.c = compile_mat(mat.specific_heat);
            if (mat.kind == mhs::model::MaterialKind::Fluid) {
                if (!mat.dynamic_viscosity || mat.dynamic_viscosity->empty())
                    throw std::runtime_error(
                        "fluid material '" + definition.materials[m].name + "' is missing DynamicViscosity");
                result.initial_viscosity[m]
                    = compile_mat(*mat.dynamic_viscosity).eval({0, 0, 0, model.reference_temperature, 0});
            }
            else if (mat.dynamic_viscosity) {
                throw std::invalid_argument(
                    "solid material '" + definition.materials[m].name + "' must not define DynamicViscosity");
            }
        }

        for (const auto& source : definition.sources)
            model.heat_source_table.push_back(mhs::expression::parse(source, symbols));
        for (const auto& volume : definition.volumes) {
            if (volume.material >= definition.materials.size())
                throw std::out_of_range("volume material ID out of range");
            if (volume.source != mhs::core::invalid_id && volume.source >= definition.sources.size())
                throw std::out_of_range("volume source ID out of range");
        }

        // Compile boundary patches.
        auto& boundary_conditions = model.boundary_conditions;
        result.thermal_regions
            = compile_thermal_boundaries(definition.thermal_boundaries, boundary_conditions, symbols);

        result.default_thermal_boundary
            = compile_thermal_boundary(definition.default_thermal_boundary, boundary_conditions, symbols);

        for (std::size_t id = 0; id < definition.fluid_boundaries.size(); ++id)
            for (const auto& region : definition.fluid_boundaries[id].regions)
                result.fluid_regions.push_back({region, static_cast<SurfaceTag>(id)});
        return result;
    }

    void bind_cell_physics(mhs::model::Model& model, std::span<const mhs::model::VolumeSpec> volumes,
        std::span<const RegionId> cell_regions)
    {
        model.cell_material.reserve(cell_regions.size());
        for (std::size_t c = 0; c < cell_regions.size(); ++c) {
            const auto& volume = volumes[cell_regions[c]];
            const auto& cell = model.discrete.cells[c];
            model.cell_material.push_back(volume.material);
            if (volume.source != mhs::core::invalid_id)
                model.source_terms.push_back(
                    {static_cast<mhs::model::CellId>(c), volume.source, cell.volume(), cell.center()});
        }
    }
} // namespace mhs::compiler
