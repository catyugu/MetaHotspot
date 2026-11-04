#pragma once

#include "core/discrete_mesh.hpp"
#include "core/fluid_domain.hpp"
#include "core/output_mesh.hpp"
#include "core/types.hpp"
#include "numerics/expression/expr.hpp"

#include "core/vector.hpp"
#include <string>
#include <variant>

namespace mhs::core {

    // Projected integration patch on an external topological face.
    struct BoundaryFace {
        FaceId parent;
        TableIndex condition; // index in the condition table used for this projection
        std::array<double, 4> bounds; // tangential min/max pairs
    };

    struct MaterialProps {
        mhs::numerics::CompiledExpression kx;
        mhs::numerics::CompiledExpression ky;
        mhs::numerics::CompiledExpression kz;
        mhs::numerics::CompiledExpression rho;
        mhs::numerics::CompiledExpression c;
    };

    struct Dirichlet {
        mhs::numerics::CompiledExpression temperature;
    };

    struct Neumann {
        mhs::numerics::CompiledExpression heat_flux;
    };

    struct Convection {
        mhs::numerics::CompiledExpression coefficient;
        mhs::numerics::CompiledExpression ambient_temperature;
    };

    // Shared compiled laws; per-face BoundaryFace records store only an index.
    using BoundaryCondition = std::variant<Dirichlet, Neumann, Convection>;

    // Midpoint quadrature on a physical source subvolume.
    struct SourceTerm {
        CellId cell;
        TableIndex source;
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

        mhs::core::Vector<TableIndex> cell_material;
        mhs::core::Vector<MaterialProps> material_table;
        mhs::core::Vector<mhs::numerics::CompiledExpression> heat_source_table;
        mhs::core::Vector<BoundaryCondition> boundary_conditions;
        mhs::core::Vector<SourceTerm> source_terms;
        mhs::core::Vector<BoundaryFace> boundary_faces; // sorted by parent FaceId
        FluidDomain fluid;

        double initial_temperature = 300.0;
        StudyType study_type = StudyType::Steady;
        double transient_duration = 0.0;
        double transient_time_step = 1.0;

        mhs::core::Vector<ProbePoint> observation_points;
    };

} // namespace mhs::core
