#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "expr/expr.hpp"
#include "fluid_domain.hpp"
#include "types.hpp"

namespace mhs::core {

    // ── A per-face BC record ─────────────────────────────────
    // Every cell has exactly 6 faces, stored as a flat array
    // [N_active * 6] in row-major order (dir 0..5 per cell).
    struct FaceBC {
        BcType type = BcType::None; // None = internal face or adiabatic
        uint16_t param_idx = 0; // → BCParamTable
    };

    // ── Structured mesh geometry ─────────────────────────────────────────
    struct MeshGeometry {
        mhs::Index nx = 0, ny = 0, nz = 0;

        std::vector<double> dx;
        std::vector<double> dy;
        std::vector<double> dz;

        std::vector<double> cx;
        std::vector<double> cy;
        std::vector<double> cz;
    };

    // ── Material properties ──────────────────────────────────────────────
    struct MaterialProps {
        CompiledExpression kx;
        CompiledExpression ky;
        CompiledExpression kz;
        CompiledExpression rho;
        CompiledExpression c;
        CompiledExpression dynamic_viscosity; // μ; 非 fluid = make_constant(0)

        bool is_fluid = false;
    };

    // ── BC parameter table (per-BC-type expression vectors) ──────────────
    struct BCParamTable {
        std::vector<CompiledExpression> dirichlet_T;
        std::vector<CompiledExpression> neumann_q;
        std::vector<CompiledExpression> cauchy_h;
        std::vector<CompiledExpression> cauchy_T_inf;
    };

    // ── Per-cell fields (compact, N_active entries) ──────────────────────
    struct CellFields {
        std::vector<uint16_t> material_id; // index into material_table
        std::vector<uint16_t> heat_source_idx; // index into heat_source_table
        std::vector<Index> index_map; // old grid index → compact;
                                      // invalidIndex = virtual / inactive
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
        MeshGeometry mesh;
        CellFields cells;

        // Face-level BC storage: flat array [N_active * 6].
        // face_bcs[c * 6 + dir] gives the BC for cell c's face `dir`.
        std::vector<FaceBC> face_bcs;
        BCParamTable bc_params;

        std::vector<MaterialProps> material_table;

        std::vector<CompiledExpression> heat_source_table;

        double initial_temperature = 300.0;
        StudyType study_type = StudyType::Steady;
        double transient_duration = 0.0;
        double transient_time_step = 1.0;

        std::vector<ProbePoint> observation_points;

        // Fluid-solid coupled heat-transfer subsystem
        mhs::core::FluidDomain fluid;
    };

} // namespace mhs::core
