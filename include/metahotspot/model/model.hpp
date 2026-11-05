#pragma once

#include "expression/expression.hpp"
#include "model/discrete_mesh.hpp"
#include "model/fluid_model.hpp"
#include "model/output_mesh.hpp"
#include "model/types.hpp"

#include "core/vector.hpp"
#include <string>
#include <variant>

namespace mhs::model {

    // Projected external boundary integration term; independent of mesh face IDs.
    struct BoundaryFace {
        CellId cell = mhs::core::invalid_id;
        Axis axis = Axis::X;
        std::int8_t sign = 1;
        BoundaryConditionId condition = mhs::core::invalid_id; // index in the condition table used for this projection
        std::array<double, 4> bounds; // tangential min/max pairs
    };

    struct MaterialProps {
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

    // Shared compiled laws; per-face BoundaryFace records store only an index.
    using BoundaryCondition = std::variant<Dirichlet, Neumann, Convection>;

    // Midpoint quadrature on a physical source subvolume.
    struct SourceTerm {
        CellId cell = mhs::core::invalid_id;
        SourceId source = mhs::core::invalid_id;
        double weight;
        Vec3 point;
    };

    struct ProbePoint {
        std::string name;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct Model {
        DiscreteMesh discrete;
        OutputMesh output;

        mhs::core::Vector<MaterialId> cell_material;
        mhs::core::Vector<MaterialProps> material_table;
        mhs::core::Vector<mhs::expression::CompiledExpression> heat_source_table;
        mhs::core::Vector<BoundaryCondition> boundary_conditions;
        mhs::core::Vector<SourceTerm> source_terms;
        mhs::core::Vector<BoundaryFace> boundary_faces; // sorted by CellId
        FluidModel fluid;

        double initial_temperature = 300.0;
        StudyType study_type = StudyType::Steady;
        double transient_duration = 0.0;
        double transient_output_interval = 1.0;

        mhs::core::Vector<ProbePoint> observation_points;
    };

} // namespace mhs::model
