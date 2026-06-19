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
        std::vector<uint16_t> fluid_material_id; // fluid material index; max() when non-fluid
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
        // ============================================================
        std::vector<uint8_t> is_fluid;                // [N_active] 标记流体 cell
        std::vector<double> dynamic_viscosity;        // [N_active] 流体 cell 的 μ [Pa·s]；非 fluid = 0
        std::vector<double> pressure;                 // [N_active] 求解后的压力场；非 fluid = 0
        std::vector<int8_t> flow_axes;                // [N_active] 主导流轴 [0=X,1=Y,2=Z]；非 fluid = -1
        std::vector<double> hydroC_x;                 // [N_active] hydraulic conductance 沿 X
        std::vector<double> hydroC_y;                 // [N_active] hydraulic conductance 沿 Y
        std::vector<double> hydroC_z;                 // [N_active] hydraulic conductance 沿 Z
        std::vector<uint8_t> is_pressure_boundary;    // [N_active] 压力边界标记
        std::vector<double> boundary_pressure;        // [N_active] 压力边界值 [Pa]
        std::vector<double> boundary_temperature_fluid;// [N_active] 流体入口温度 [K]；非入口 = NaN
        std::vector<double> hydraulic_diameter;        // [N_active] 水力直径 [m]；非 fluid = 0
    };

} // namespace mhs::core
