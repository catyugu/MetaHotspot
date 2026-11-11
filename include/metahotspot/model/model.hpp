#pragma once

#include "expression/expression.hpp"
#include "model/fluid_model.hpp"
#include "model/mesh.hpp"
#include "model/types.hpp"

#include "core/vector.hpp"
#include <string>
#include <variant>

namespace mhs::model {

    struct MaterialProps {
        MaterialKind kind = MaterialKind::Solid;
        mhs::expression::CompiledExpression kx;
        mhs::expression::CompiledExpression ky;
        mhs::expression::CompiledExpression kz;
        mhs::expression::CompiledExpression rho;
        mhs::expression::CompiledExpression c;
    };

    struct Dirichlet {
        mhs::expression::CompiledExpression temperature;
    };

    struct Neumann {
        mhs::expression::CompiledExpression heat_flux;
    };

    struct Convection {
        mhs::expression::CompiledExpression coefficient;
        mhs::expression::CompiledExpression ambient_temperature;
    };

    // Compiled laws are assigned independently to exterior BoundaryFacetIds.
    using BoundaryCondition = std::variant<Dirichlet, Neumann, Convection>;

    // Midpoint quadrature on a physical source subvolume.
    struct SourceTerm {
        CellId cell = mhs::core::invalid_id;
        SourceId source = mhs::core::invalid_id;
        double weight;
        Vec3 point;
    };

    struct PointProbe {
        std::string name;
        Vec3 point {};
        CellId cell = mhs::core::invalid_id;
    };

    struct Cell {
        MaterialId material = core::invalid_id;
    };

    struct Model {
        Mesh mesh;
        core::Vector<Cell> cells;
        core::Vector<BoundaryFacet> boundary_facets;

        mhs::core::Vector<MaterialProps> material_table;
        mhs::core::Vector<mhs::expression::CompiledExpression> heat_source_table;
        mhs::core::Vector<BoundaryCondition> boundary_conditions;
        mhs::core::Vector<SourceTerm> source_terms;
        mhs::core::Vector<BoundaryConditionId> boundary_thermal_condition; // by BoundaryFacetId
        FluidModel fluid;

        double reference_temperature = 300.0;

        mhs::core::Vector<PointProbe> point_probes;
    };

} // namespace mhs::model
