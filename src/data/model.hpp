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
        BcType type = BcType::None; // None = internal face or adiabatic
        uint16_t param_idx = 0; // → BCParamTable
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

    /// Per-face invariant data for a SmartMacro port.
    /// Everything here is purely geometric/topological — independent of temperature and time.
    struct PortFaceInfo {
        // Owner cell grid coordinates (inside the SmartMacro block)
        int ix = 0, iy = 0, iz = 0;
        mhs::core::FaceDir dir = mhs::core::FaceDir::XM;

        // Precomputed face geometry
        double A_f = 0.0; // face area [m²]

        // ── Active neighbor coupling ──
        bool has_neighbor = false;
        uint32_t neighbor_c = invalidIndex; // compact index of the neighbor cell
        double half_dist_nbr = 0.0; // neighbor half-length along dir [m]

        // ── Domain boundary BC (used only when !has_neighbor) ──
        mhs::core::BcType bc_type = mhs::core::BcType::None;
        uint16_t bc_param_idx = 0;
    };

    /// A trained POD-reduced macro model ready for assembly.
    /// Port faces on the block boundary are reduced via POD. Each port face
    /// `f` has invariant geometry/topology stored in `faces[f]`.
    ///
    /// At assembly time, the environment parameters (C_env, T_ref, Q_ext),
    /// cell-aggregated coupling, and K_modal_eff are computed on-the-fly
    /// from ctx.T and ctx.current_time — supporting nonlinear neighbour
    /// materials (k(T)) and time-varying BCs.
    struct SmartBlockModel {
        std::string name;

        // Invariant modal data (from training).
        Eigen::MatrixXd K_modal; // [n_modes x n_modes] — raw modal stiffness
        Eigen::MatrixXd phi_basis; // [n_faces x n_modes] — POD basis

        // Per-face invariant geometry & topology.
        std::vector<PortFaceInfo> faces;

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
        int total_modal_dofs() const
        {
            int s = 0;
            for (const auto& sb : smart_blocks)
                s += sb.n_modes;
            return s;
        }
        int total_dofs() const { return physical_dofs() + total_modal_dofs(); }
    };

} // namespace mhs::core
