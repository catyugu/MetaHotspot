#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "expr/expr.hpp"
#include "fluid_domain.hpp"
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
        CompiledExpression dynamic_viscosity; // μ; 非 fluid = make_constant(0)
        // 流体-固体耦合扩展
        bool is_fluid = false;
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

    // 内部探针点：用户坐标系下的固定位置（已求值到 SI 单位），求解器在每个时间步记录该点温度。
    // 与 IOStructure::ObservationPoint3D（表达式字符串）一一对应，由 preprocessor 转换生成。
    struct ProbePoint {
        std::string name;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct Model {
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
        // 流体-固体耦合传热 (fluid-algorithm) 子系统
        // 类型定义见 src/data/fluid_domain.hpp (mhs::core::FluidDomain)。
        // 零初始化；无 overlay 时全部为空/零值，不参与求解。
        //
        // 流体属性数组为 [n_fluid] 紧凑长度，通过 global_to_fluid 索引：
        //   f_idx = fluid.global_to_fluid[c_idx]  (或 -1 为非流体)
        //   c_idx = fluid.fluid_to_global[f_idx]
        // ============================================================
        mhs::core::FluidDomain fluid;
    };

} // namespace mhs::core
