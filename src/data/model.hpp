#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "expr/expr.hpp"
#include "fluid_domain.hpp"
#include "types.hpp"

namespace mhs::core {

    inline constexpr uint32_t invalidIndex = std::numeric_limits<uint32_t>::max();

    // ── A per-face BC record ─────────────────────────────────
    // Every cell has exactly 6 faces, stored as a flat array
    // [N_active * 6] in row-major order (dir 0..5 per cell).
    struct FaceBC {
        BcType type = BcType::None;  // None = internal face or adiabatic
        uint16_t param_idx = 0;      // → BCParamTable
    };

    // ── Structured mesh geometry ─────────────────────────────────────────
    struct MeshGeometry {
        int nx = 0, ny = 0, nz = 0;

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
        std::vector<uint32_t> index_map; // old grid index → compact;
                                         // invalidIndex = virtual / inactive
    };

    // ── Probe / observation point ────────────────────────────────────────
    struct ProbePoint {
        std::string name;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    // ── Smart macro-model (POD-based extended system) ────────────────

    /// A trained POD-reduced macro model ready for assembly.
    /// Port faces on the block boundary are reduced via POD. Each port face
    /// `f` maps to an owner cell inside the block and has environment
    /// parameters (C_env_f, T_ref_f, Q_ext_f) from either an active
    /// neighbor cell or a domain-boundary BC.
    ///
    /// At assembly time, the extended system scattering uses aggregated
    /// per-neighbor-cell coupling data (`coupled_cells`/`coupled_C`/`coupled_phi`)
    /// for compact triplet generation.
    struct SmartBlockModel {
        std::string name;

        // Face-level environment parameters [N_faces].
        Eigen::VectorXd C_env_vec;
        Eigen::VectorXd T_ref_vec;
        Eigen::VectorXd Q_ext_vec;

        // POD basis [N_faces x n_modes] and modal operators.
        Eigen::MatrixXd phi_basis;
        Eigen::MatrixXd K_modal_eff;       // [n_modes x n_modes] = K_modal + Φᵀ·C_env·Φ (all faces)

        // Aggregated coupling for active-neighbor cells.
        std::vector<uint32_t> coupled_cells;     // [n_coupled] — compact cell indices
        Eigen::VectorXd coupled_C;               // [n_coupled] — Σ C_env_f per cell
        Eigen::MatrixXd coupled_phi;             // [n_coupled x n_modes] — Σ C_env_f * φ(f,k)

        int modal_start_idx = 0;
        int n_faces = 0;
        int n_modes = 0;
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

        // Smart macro-model blocks (POD-based extended system)
        std::vector<SmartBlockModel> smart_blocks;

        // ── DOF accounting helpers ──────────────────────────────────────
        int physical_dofs() const { return static_cast<int>(cells.material_id.size()); }
        int total_modal_dofs() const {
            int s = 0;
            for (const auto& sb : smart_blocks) s += sb.n_modes;
            return s;
        }
        int total_dofs() const {
            return physical_dofs() + total_modal_dofs();
        }
    };

} // namespace mhs::core
