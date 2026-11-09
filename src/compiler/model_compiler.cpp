#include "compiler/model_compiler.hpp"
#include "compiler/fluid_preprocessor.hpp"
#include "compiler/geometry_compiler.hpp"
#include "compiler/model_functions.hpp"
#include "core/vector.hpp"
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

    mhs::model::Model build_model(const mhs::model::ModelDefinition& definition)
    {
        if (!std::isfinite(definition.reference_temperature))
            throw std::invalid_argument("reference temperature must be finite");
        validate_surface_regions(definition);

        mhs::model::Model model;
        model.reference_temperature = definition.reference_temperature;

        auto symbols = build_symbol_table(definition);
        for (const auto& p : definition.point_probes)
            model.point_probes.push_back({p.name, {p.x, p.y, p.z}});

        // Compile material properties in ModelDefinition::materials order.
        model.material_table.resize(definition.materials.size());
        mhs::compiler::fluid::FluidMaterialData fluid_materials;
        fluid_materials.initial_viscosity.resize(definition.materials.size());

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
                fluid_materials.initial_viscosity[m]
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
            for (const auto& box : volume.geometry)
                for (int a = 0; a < 3; ++a)
                    if (!std::isfinite(box.min[a]) || !std::isfinite(box.max[a]) || box.min[a] >= box.max[a])
                        throw std::invalid_argument("box bounds must be finite and increasing");
        }

        // Compile boundary patches.
        auto& boundary_conditions = model.boundary_conditions;
        auto compiled_boundaries
            = compile_thermal_boundaries(definition.thermal_boundaries, boundary_conditions, symbols);

        auto default_thermal_boundary
            = compile_thermal_boundary(definition.default_thermal_boundary, boundary_conditions, symbols);

        compile_cartesian_mesh(model, definition.mesh, definition.volumes);
        project_cell_regions(model, definition.volumes);
        for (const auto material : model.cell_material)
            if (material >= model.material_table.size())
                throw std::out_of_range("cell has no valid material assignment");

        // Canonical geometric partition shared by all boundary physics.
        mhs::core::Vector<mhs::model::SurfaceRegion> cuts;
        for (const auto& boundary : compiled_boundaries)
            cuts.push_back(boundary.region);
        mhs::core::Vector<TaggedSurfaceRegion> fluid_regions;
        for (std::size_t id = 0; id < definition.fluid_boundaries.size(); ++id)
            for (const auto& region : definition.fluid_boundaries[id].regions) {
                cuts.push_back(region);
                fluid_regions.push_back({region, id});
            }
        refine_exterior_facets(model.discrete, cuts);
        build_cell_facet_incidence(model.discrete);

        auto thermal_tags = assign_exterior_regions(model.discrete, compiled_boundaries, default_thermal_boundary);
        model.facet_thermal_condition.reserve(thermal_tags.size());
        for (auto tag : thermal_tags)
            model.facet_thermal_condition.push_back(static_cast<mhs::model::BoundaryConditionId>(tag));

        // Hydraulic boundary assignments share the same canonical FacetIds.
        auto fluid_tags = assign_exterior_regions(model.discrete, fluid_regions);
        mhs::compiler::fluid::build_domain(model, definition.fluid_boundaries, fluid_tags, fluid_materials);

        return model;
    }

} // namespace mhs::compiler
