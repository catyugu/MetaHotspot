#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "data/solution_history.hpp"
#include "expr/expr.hpp"
#include "types.hpp"

namespace mhs::core {

    inline constexpr uint32_t invalidIndex = std::numeric_limits<uint32_t>::max();

    struct CellBC {
        std::array<BcType, FACE_COUNT> types;
        std::array<uint16_t, FACE_COUNT> param_idxs;
    };

    struct MeshGeometry {
        int nx = 0, ny = 0, nz = 0;

        std::vector<double> dx;
        std::vector<double> dy;
        std::vector<double> dz;

        std::vector<double> cx;
        std::vector<double> cy;
        std::vector<double> cz;
    };

    struct MaterialProps {
        CompiledExpression kx;
        CompiledExpression ky;
        CompiledExpression kz;
        CompiledExpression rho;
        CompiledExpression c;
    };

    struct CellFields {
        std::vector<uint16_t> material_id; // index into material_table
        std::vector<uint16_t> heat_source_idx; // index into heat_source_table
        std::vector<CellBC> cell_bcs;
        std::vector<uint32_t> index_map; // old grid index → compact; invalidIndex = virtual
    };

    struct BCParamTable {
        std::vector<CompiledExpression> dirichlet_T;
        std::vector<CompiledExpression> neumann_q;
        std::vector<CompiledExpression> cauchy_h;
        std::vector<CompiledExpression> cauchy_T_inf;
    };

    /// Mutable, per-step state owned by Scheduler::run().
    /// Invariant: state.T is the most recent accepted solution; it mirrors
    /// accepted.current() at the end of every accepted step.
    struct GlobalState {
        double current_time = 0.0;
        int time_step = 0;
        double dt = 0.0;

        // BDF-k history buffer.  accepted.current() == T (after accept).
        // The buffer capacity matches the time scheme's max_order (typically
        // 2 for BDF2 / AdaptiveBdf).  Populated via accepted.initialize(T)
        // before the first step.
        SolutionHistory accepted {0, 1};

        // Active temperature field, length = N_active (== cells.cell_bcs.size()).
        std::vector<double> T;
    };

    // 内部探针点：用户坐标系下的固定位置（已求值到 SI 单位），求解器在每个时间步记录该点温度。
    // 与 IOStructure::ObservationPoint3D（表达式字符串）一一对应，由 preprocessor 转换生成。
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
        StudyType study_type = StudyType::Steady;
        double transient_duration = 0.0;
        double transient_time_step = 1.0;

        // 用户坐标系下的 3D 观察点列表（来自 IOStructure，已求值到 SI 单位）。
        // 探针不参与方程求解，仅用于输出温度时间序列。
        std::vector<ProbePoint> observation_points;
    };

} // namespace mhs::core
