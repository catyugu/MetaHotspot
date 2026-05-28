#pragma once

#include "types.hpp"
#include <vector>

namespace mhs::model {

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
        FieldExpression k; // 导热系数
        FieldExpression rho; // 密度
        FieldExpression c; // 比热容
    };

    struct CellFields {
        int cell_count = 0;

        std::vector<size_t> material_id;
        std::vector<size_t> layer_id;

        std::vector<FieldExpression> heat_source;

        std::vector<uint8_t> bc_flags;
    };

    struct BCParamTable {
        std::vector<FieldEvaluator> dirichlet_T;
        std::vector<FieldEvaluator> neumann_q;
        std::vector<FieldEvaluator> cauchy_h;
        std::vector<FieldEvaluator> cauchy_T_inf;
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

        std::vector<double> T;
        std::vector<double> T_prev;
        std::vector<double> residual;
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
