#pragma once

#include "core/discrete_mesh.hpp"
#include "core/fluid_domain.hpp"
#include "core/output_mesh.hpp"
#include "core/types.hpp"
#include "numerics/expression/expr.hpp"

#include <string>
#include <variant>
#include <vector>

namespace mhs::core {

    // Boundary integration rectangle on an unchanged topological face.
    struct BoundaryTerm {
        FaceId face;
        TableIndex condition; // index in the condition table used for this projection
        std::array<double, 4> bounds; // tangential min/max pairs
    };

    struct MaterialProps {
        CompiledExpression kx;
        CompiledExpression ky;
        CompiledExpression kz;
        CompiledExpression rho;
        CompiledExpression c;
    };

    struct Dirichlet {
        CompiledExpression temperature;
    };

    struct Neumann {
        CompiledExpression heat_flux;
    };

    struct Convection {
        CompiledExpression coefficient;
        CompiledExpression ambient_temperature;
    };

    // Shared compiled laws; per-face BoundaryTerm records store only an index.
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

        std::vector<TableIndex> cell_material;
        std::vector<MaterialProps> material_table;
        std::vector<CompiledExpression> heat_source_table;
        std::vector<BoundaryCondition> boundary_conditions;
        std::vector<SourceTerm> source_terms;
        std::vector<BoundaryTerm> boundary_terms;
        FluidDomain fluid;

        double initial_temperature = 300.0;
        StudyType study_type = StudyType::Steady;
        double transient_duration = 0.0;
        double transient_time_step = 1.0;

        std::vector<ProbePoint> observation_points;
    };

} // namespace mhs::core
