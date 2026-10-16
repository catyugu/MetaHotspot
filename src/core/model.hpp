#pragma once

#include "core/discrete_mesh.hpp"
#include "core/fluid_domain.hpp"
#include "core/output_mesh.hpp"
#include "core/types.hpp"
#include "numerics/expression/expr.hpp"

#include <string>
#include <type_traits>
#include <vector>

namespace mhs::core {

    // Boundary integration rectangle on an unchanged topological face.
    struct BoundaryTerm {
        FaceId face;
        TableIndex parameter;
        std::array<double, 4> bounds; // tangential min/max pairs
    };

    static_assert(std::is_trivially_copyable_v<BoundaryTerm> && std::is_standard_layout_v<BoundaryTerm>);

    inline FacePatch projected_face(const DiscreteMesh& mesh, const BoundaryTerm& term)
    {
        auto f = mesh.faces[term.face.value];
        const int a = f.axis == 0 ? 1 : 0, b = f.axis == 2 ? 1 : 2;
        f.min[a] = term.bounds[0];
        f.max[a] = term.bounds[1];
        f.min[b] = term.bounds[2];
        f.max[b] = term.bounds[3];
        f.area
            = (f.max[(f.axis + 1) % 3] - f.min[(f.axis + 1) % 3]) * (f.max[(f.axis + 2) % 3] - f.min[(f.axis + 2) % 3]);
        return f;
    }

    // ── Material properties ──────────────────────────────────────────────
    struct MaterialProps {
        CompiledExpression kx;
        CompiledExpression ky;
        CompiledExpression kz;
        CompiledExpression rho;
        CompiledExpression c;
    };

    // Value is temperature, heat flux or HTC according to type.
    // Robin ambient temperature shares the same record and index as HTC.
    struct BoundaryParameters {
        BcType type;
        CompiledExpression value;
        CompiledExpression ambient;
    };

    // Midpoint quadrature on a physical source subvolume.
    struct SourceTerm {
        CellId cell;
        TableIndex source;
        double weight;
        Vec3 point;
    };
    static_assert(std::is_trivially_copyable_v<SourceTerm> && std::is_standard_layout_v<SourceTerm>);

    // ── Probe / observation point ────────────────────────────────────────
    struct ProbePoint {
        std::string name;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    // ── Top-level model ──────────────────────────────────────────────────
    struct Model {
        DiscreteMesh discrete;
        OutputMesh output;
        std::vector<BoundaryTerm> boundary_terms;
        std::vector<TableIndex> cell_material;
        std::vector<SourceTerm> source_terms;

        std::vector<BoundaryParameters> boundary_table;

        std::vector<MaterialProps> material_table;

        std::vector<CompiledExpression> heat_source_table;

        double initial_temperature = 300.0;
        StudyType study_type = StudyType::Steady;
        double transient_duration = 0.0;
        double transient_time_step = 1.0;

        std::vector<ProbePoint> observation_points;

        mhs::core::FluidDomain fluid;
    };

    // Terms are sorted by parent FaceId; lookup needs no per-face cache.
    inline std::span<const BoundaryTerm> boundary_terms_on_face(const Model& model, FaceId face)
    {
        const auto& terms = model.boundary_terms;
        auto first = std::lower_bound(
            terms.begin(), terms.end(), face, [](const BoundaryTerm& t, FaceId f) { return t.face.value < f.value; });
        auto last = first;
        while (last != terms.end() && last->face == face)
            ++last;
        return {first, last};
    }

} // namespace mhs::core
