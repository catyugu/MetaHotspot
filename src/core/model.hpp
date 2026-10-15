#pragma once

#include "core/discrete_mesh.hpp"
#include "core/fluid_domain.hpp"
#include "core/output_mesh.hpp"
#include "core/types.hpp"
#include "numerics/expression/expr.hpp"

#include <string>
#include <vector>

namespace mhs::core {

    // ── A per-face BC record ─────────────────────────────────
    struct FaceBC {
        BcType type = BcType::None;
        TableIndex param_idx = 0;
    };

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

    // ── Cell topology and compact per-active-cell fields ────────────────
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
        std::vector<FaceBC> patch_bcs;
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

} // namespace mhs::core
