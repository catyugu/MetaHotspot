#pragma once
#include <array>
#include <cstdint>
#include <limits>
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
        // 流体-固体耦合扩展
        bool is_fluid = false;
        CompiledExpression dynamic_viscosity; // μ; 非 fluid = make_constant(0)
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
        // 流体-固体耦合扩展: 压力边界参数值 (不需要表达式, 直接 double)
        std::vector<double> pressure_bc_values; // index by PressureBC idx
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

        // ============================================================
        // 流体-固体耦合传热 (fluid-algorithm) 扩展字段
        // 所有字段零初始化；无 overlay 时全部为空/零值，不参与求解。
        //
        // 流体属性数组为 [n_fluid] 紧凑长度，通过 global_to_fluid 索引：
        //   f_idx = global_to_fluid[c_idx]  (或 -1 为非流体)
        //   c_idx = fluid_to_global[f_idx]
        // ============================================================
        int n_fluid = 0;                              // 流体单元数
        std::vector<int> fluid_to_global;             // [n_fluid] → N_active compact index
        std::vector<int> global_to_fluid;             // [N_active] → n_fluid fluid index, -1 = 非流体

        std::vector<uint8_t> is_fluid;                // [N_active] 标记流体 cell（用于快速判别）
        std::vector<double> dynamic_viscosity;        // [n_fluid] μ [Pa·s]
        std::vector<double> pressure;                 // [n_fluid] 压力场
        std::vector<int8_t> flow_axes;                // [n_fluid] 主导流轴
        std::vector<double> hydroC_x;                 // [n_fluid] hydraulic conductance X
        std::vector<double> hydroC_y;                 // [n_fluid] hydraulic conductance Y
        std::vector<double> hydroC_z;                 // [n_fluid] hydraulic conductance Z
        std::vector<uint8_t> is_pressure_boundary;    // [n_fluid] 压力边界标记
        std::vector<double> boundary_pressure;        // [n_fluid] 压力边界值 [Pa]
        std::vector<double> boundary_temperature_fluid;// [n_fluid] 入口温度 [K]；非入口 = NaN
        std::vector<double> hydraulic_diameter;        // [n_fluid] 水力直径 [m]
        std::vector<double> channel_width;             // [n_fluid] 通道宽度 [m]
        std::vector<double> channel_height;            // [n_fluid] 通道高度 [m]
    };

} // namespace mhs::core
