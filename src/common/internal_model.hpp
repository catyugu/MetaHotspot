#pragma once
#include <string>
#include <vector>

#include "types.hpp"

namespace mhs {

    struct CellBC {
        std::array<BcType, FACE_COUNT> types;
        std::array<uint16_t, FACE_COUNT> param_idxs;
    };

    struct MeshGeometry {
        int nx = 0, ny = 0, nz = 0;
        int total_cell_count = 0;

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
        int cell_count = 0;

        std::vector<size_t> index_map;
        std::vector<uint8_t> valid_mask;
        std::vector<size_t> material_id;
        std::vector<size_t> layer_id;

        std::vector<CellBC> cell_bcs;

        // 降维为 16 位整型字典索引，实现极速的连续内存读取
        std::vector<uint16_t> heat_source_idx;
    };

    struct BCParamTable {
        std::vector<CompiledExpression> dirichlet_T;
        std::vector<CompiledExpression> neumann_q;
        std::vector<CompiledExpression> cauchy_h;
        std::vector<CompiledExpression> cauchy_T_inf;
    };

    struct GlobalState {
        int cell_count = 0;
        double current_time = 0.0;
        int time_step = 0;
        double dt = 0.0;

        std::vector<double> T;
        std::vector<double> T_prev;
        std::vector<double> residual;
    };

    // 内部探针点：用户坐标系下的固定位置（已求值到 SI 单位），求解器在每个时间步记录该点温度。
    // 与 io_model::ObservationPoint3D（表达式字符串）一一对应，由 preprocessor 转换生成。
    struct ProbePoint {
        std::string name;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct InternalModel {
        MeshGeometry mesh;
        CellFields cells;
        BCParamTable bc_params;

        std::vector<MaterialProps> material_table;

        std::vector<CompiledExpression> heat_source_table;

        double initial_temperature = 300.0;
        double ambient_temperature = 300.0;
        StudyType study_type = StudyType::Steady;
        double transient_duration = 0.0;
        double transient_time_step = 1.0;

        // 用户坐标系下的 3D 观察点列表（来自 IOStructure，已求值到 SI 单位）。
        // 探针不参与方程求解，仅用于输出温度时间序列。
        std::vector<ProbePoint> observation_points;
    };

} // namespace mhs
