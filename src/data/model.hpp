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

    // ── Smart macro-model (DtN) support ──────────────────────────────────
    /// Records one port (interface face) of a SmartBlock.
    /// Maps a smart block interface face to the adjacent active cell.
    struct SmartPortRecord {
        uint32_t active_cell_idx = invalidIndex;
        mhs::core::FaceDir dir = FaceDir::XM;
        uint16_t port_idx = 0;   // index into K_port
        double C = 0.0;          // coupling conductance [W/K]
    };

    /// A trained DtN macro model.
    struct SmartBlockModel {
        std::string name;
        Eigen::MatrixXd K_port;               // dense [N_ports x N_ports] DtN matrix
        Eigen::VectorXd f_port;               // RHS vector from BCs on smart block (size N_ports)
        std::vector<SmartPortRecord> ports;
        Eigen::SparseMatrix<double> K_eff;    // pre-computed effective stiffness contribution
        Eigen::VectorXd rhs_eff;              // pre-computed effective RHS contribution (size N_ports)
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

        // Smart macro-model blocks (DtN-based)
        std::vector<SmartBlockModel> smart_blocks;
    };

} // namespace mhs::core
