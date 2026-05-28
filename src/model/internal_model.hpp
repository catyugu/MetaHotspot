#pragma once

#include "expr/expr.hpp"
#include "types.hpp"
#include <deque>
#include <vector>

namespace mhs::model {

    // Type alias for convenience
    using CompiledExpression = expr::CompiledExpression;

    struct MeshGeometry {
        int nx = 0, ny = 0, nz = 0;
        int cell_count = 0;

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
        CompiledExpression k; // 导热系数
        CompiledExpression rho; // 密度
        CompiledExpression c; // 比热容
    };

    struct CellFields {
        int cell_count = 0;

        std::vector<size_t> material_id;
        std::vector<size_t> layer_id;

        std::vector<CompiledExpression> heat_source;

        std::vector<uint8_t> bc_flags;
    };

    struct BCParamTable {
        std::vector<CompiledExpression> dirichlet_T;
        std::vector<CompiledExpression> neumann_q;
        std::vector<CompiledExpression> cauchy_h;
        std::vector<CompiledExpression> cauchy_T_inf;
    };

    struct FaceBCFields {
        std::vector<BcType> bc_type_zm;
        std::vector<uint16_t> bc_param_idx_zm;
        std::vector<BcType> bc_type_zp;
        std::vector<uint16_t> bc_param_idx_zp;
        std::vector<BcType> bc_type_ym;
        std::vector<uint16_t> bc_param_idx_ym;
        std::vector<BcType> bc_type_yp;
        std::vector<uint16_t> bc_param_idx_yp;
        std::vector<BcType> bc_type_xm;
        std::vector<uint16_t> bc_param_idx_xm;
        std::vector<BcType> bc_type_xp;
        std::vector<uint16_t> bc_param_idx_xp;
    };

    struct GlobalState {
        int cell_count = 0;
        double current_time = 0.0;
        int time_step = 0;
        ConvergenceStatus status = ConvergenceStatus::Running;

        std::vector<double> T; // current temperature
        std::vector<double> T_prev; // previous time step
        std::vector<double> residual; // current residual

        // Ring buffers (configurable capacity, default 5)
        std::deque<std::vector<double>> T_history; // past time steps
        std::deque<std::vector<double>> nl_history; // non-linear iteration snapshots
        std::deque<double> dt_history; // past dt values
    };

    struct InternalModel {
        MeshGeometry mesh;
        CellFields cells;
        FaceBCFields face_bcs;
        BCParamTable bc_params;

        std::vector<MaterialProps> material_table;

        double initial_temperature = 300.0;
        double ambient_temperature = 300.0;
        StudyType study_type = StudyType::Steady;
        double transient_duration = 0.0;
        double transient_time_step = 1.0;
    };

} // namespace mhs::model