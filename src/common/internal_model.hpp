#pragma once
#include "expr/expr.hpp"
#include "types.hpp"
#include <vector>

namespace mhs {

    using CompiledExpression = expr::CompiledExpression;

    // Per-cell per-face BC: type + parameter index into BCParamTable
    struct CellBC {
        std::array<BcType, FACE_COUNT> types;
        std::array<uint16_t, FACE_COUNT> param_idxs;
    };

    struct MeshGeometry {
        int nx = 0, ny = 0, nz = 0;
        int total_cell_count = 0; // nx * ny * nz

        std::vector<double> vertex_x;
        std::vector<double> vertex_y;
        std::vector<double> vertex_z;

        std::vector<double> dx;
        std::vector<double> dy;
        std::vector<double> dz;

        std::vector<double> cx;
        std::vector<double> cy;
        std::vector<double> cz;
    };

    struct MaterialProps {
        CompiledExpression k;
        CompiledExpression rho;
        CompiledExpression c;
    };

    struct CellFields {
        int cell_count = 0; // = N_active (valid cell count)

        // Full-grid size (nx*ny*nz): virtual + active
        std::vector<size_t> index_map; // Maps old grid index → compact active index. SIZE_MAX = virtual
        std::vector<uint8_t> valid_mask; // 1 = active cell, 0 = virtual
        std::vector<size_t> material_id; // Full grid size
        std::vector<size_t> layer_id; // Full grid size

        // Compact size (N_active): active cells only
        std::vector<CellBC> cell_bcs;
        std::vector<CompiledExpression> heat_source;
    };

    struct BCParamTable {
        std::vector<CompiledExpression> dirichlet_T;
        std::vector<CompiledExpression> neumann_q;
        std::vector<CompiledExpression> cauchy_h;
        std::vector<CompiledExpression> cauchy_T_inf;
    };

    struct GlobalState {
        int cell_count = 0; // = N_active
        double current_time = 0.0;
        int time_step = 0;
        double dt = 0.0; // current time step size for transient assembly

        std::vector<double> T; // size = N_active
        std::vector<double> T_prev; // size = N_active
        std::vector<double> residual; // size = N_active
    };

    struct InternalModel {
        MeshGeometry mesh;
        CellFields cells;
        BCParamTable bc_params;

        std::vector<MaterialProps> material_table;

        double initial_temperature = 300.0;
        double ambient_temperature = 300.0;
        StudyType study_type = StudyType::Steady;
        double transient_duration = 0.0;
        double transient_time_step = 1.0;
    };

} // namespace mhs