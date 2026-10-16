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

    // ── A per-face BC record ─────────────────────────────────
    struct FaceBC {
        BcType type = BcType::None;
        TableIndex param_idx = 0;
    };

    // Boundary integration rectangle on an unchanged topological face.
    struct BoundaryTerm {
        FaceId face;
        FaceBC bc;
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

    // ── BC parameter table ───────────────────────────────────────────────
    struct BCParamTable {
        std::vector<CompiledExpression> dirichlet_T;
        std::vector<CompiledExpression> neumann_q;
        std::vector<CompiledExpression> cauchy_h;
        std::vector<CompiledExpression> cauchy_T_inf;
    };

    // Physical assignments in CellId order.
    struct CellFields {
        std::vector<RegionId> region_id;

        std::vector<TableIndex> material_id;

        std::vector<TableIndex> layer_id;
        std::vector<TableIndex> block_id;
    };

    // Exact geometric subvolume membership, independent of material and DOFs.
    // Expressions are integrated by midpoint quadrature on each projected piece.
    struct VolumeAssignment {
        CellId cell;
        RegionId region;
        TableIndex source;
        double volume;
        Vec3 centroid;
    };

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
        DofMap dofs;
        std::vector<BoundaryTerm> boundary_terms;
        CellFields cells;
        std::vector<VolumeAssignment> volume_projection;

        BCParamTable bc_params;

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
