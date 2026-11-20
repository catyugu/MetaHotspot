#include "compiler/fom_compiler.hpp"
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

        mhs::model::BoundaryCondition compile_thermal_boundary(
            const mhs::model::ThermalBoundary& condition, const mhs::expression::SymbolTable& symbols)
        {
            using namespace mhs::model;
            using namespace mhs::expression;
            auto compile = [&](const auto& expression) { return parse(expression, symbols); };
            return std::visit(
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
        }

        void validate_surface_regions(const mhs::model::FomDefinition& definition)
        {
            auto check = [](const mhs::model::SurfaceRegion& region) {
                const auto& r = region.bounds;
                if (static_cast<unsigned>(region.axis) > 2 || !std::isfinite(region.coordinate)
                    || !std::ranges::all_of(r, [](double x) { return std::isfinite(x); })
                    || !mhs::model::positive_rect(r))
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

    mhs::model::Fom compile_fom(const mhs::model::FomDefinition& definition, const mhs::model::MeshSpec& mesh)
    {
        if (!std::isfinite(definition.reference_temperature))
            throw std::invalid_argument("reference temperature must be finite");
        validate_surface_regions(definition);

        mhs::model::Fom model;
        model.reference_temperature = definition.reference_temperature;

        mhs::expression::SymbolTable symbols;
        register_all_functions(symbols, definition.functions);

        // Compile material properties in FomDefinition::materials order.
        model.material_table.resize(definition.materials.size());
        mhs::core::Vector<double> initial_viscosity(definition.materials.size());

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
                initial_viscosity[m]
                    = compile_mat(*mat.dynamic_viscosity).eval({0, 0, 0, model.reference_temperature, 0});
            }
            else if (mat.dynamic_viscosity) {
                throw std::invalid_argument(
                    "solid material '" + definition.materials[m].name + "' must not define DynamicViscosity");
            }
        }

        for (const auto& source : definition.sources)
            model.heat_source_table.push_back(mhs::expression::parse(source.density, symbols));
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
        compile_conformal_mesh(model, mesh, definition.volumes, definition.point_probes);
        for (const auto material : model.cell_material)
            if (material >= model.material_table.size())
                throw std::out_of_range("cell has no valid material assignment");

        // Canonical geometric partition shared by all boundary physics.
        mhs::core::Vector<TaggedSurfaceRegion> surface_regions;
        for (std::size_t i = 0; i < definition.thermal_boundaries.size(); ++i) {
            const auto& boundary = definition.thermal_boundaries[i];
            const auto id = static_cast<mhs::model::BoundaryGroupId>(i);
            auto name = boundary.name.empty() ? "boundary " + std::to_string(i) : boundary.name;
            model.boundary_groups.push_back({std::move(name), compile_thermal_boundary(boundary.condition, symbols)});
            for (const auto& region : boundary.regions)
                surface_regions.push_back({region, id});
        }
        const auto fallback_boundary = static_cast<mhs::model::BoundaryGroupId>(model.boundary_groups.size());
        model.boundary_groups.push_back(
            {"default", compile_thermal_boundary(definition.default_thermal_boundary, symbols)});
        mhs::core::Vector<TaggedSurfaceRegion> fluid_regions;
        for (std::size_t id = 0; id < definition.fluid_boundaries.size(); ++id)
            for (const auto& region : definition.fluid_boundaries[id].regions)
                fluid_regions.push_back({region, static_cast<SurfaceTag>(id)});
        auto [thermal_tags, fluid_tags]
            = compile_exterior_facets(model.discrete, surface_regions, fluid_regions, fallback_boundary);
        model.facet_boundary_group = std::move(thermal_tags);
        mhs::compiler::fluid::build_domain(model, definition.fluid_boundaries, fluid_tags, initial_viscosity);

        return model;
    }

} // namespace mhs::compiler
