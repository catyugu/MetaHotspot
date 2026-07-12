#pragma once

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "types.hpp"

namespace mhs::core {

    struct Variable {
        std::string name;
        std::string value;
    };

    struct Rect {
        bool add_sub;
        std::string width_expr;
        std::string height_expr;
        std::string x_expr;
        std::string y_expr;
        std::string x_size_expr;
        std::string y_size_expr;
        std::string x_interval_expr;
        std::string y_interval_expr;
        std::string name;
    };

    enum class BlockType { Normal, SmartMacro };

    // 找到 struct Block 定义并修改：
    struct Block {
        std::vector<Rect> all_rects;
        std::string material_name;
        std::string x_offset_expr;
        std::string y_offset_expr;
        std::string ti_reyuan_expr; // 体热源表达式 [W/m³]
        std::string name;
        std::string thickness_expr; // 新增：Block 自己的厚度表达式
        BlockType block_type = BlockType::Normal;
        std::string model_file; // 仅 SmartMacro: 训练模型 XML 路径
    };
    struct Layer {
        std::vector<Block> blocks;
        std::string name;
        std::string thickness_expr;
        std::string x_offset_expr;
        std::string y_offset_expr;
        std::string period_width_expr;
        int period_width = 10;
        bool is_top_layer = false;
    };

    enum class BoundaryCategory { Electrical };

    enum class ThermalBCType { FirstType, SecondType, ThirdType };

    struct FirstTypeThermalBC {
        std::string temperature = "300.0";
    };

    struct SecondTypeThermalBC {
        std::string heat_flux = "0.0";
    };

    struct ThirdTypeThermalBC {
        std::string convection_coeff = "0.0";
        std::string T_inf = "300.0";
    };

    struct Boundary {
        BoundaryCategory category;
        std::string name;
        std::vector<std::string> face_keys;
        ThermalBCType bc_type;
        FirstTypeThermalBC first;
        SecondTypeThermalBC second;
        ThirdTypeThermalBC third;
    };

    struct Material {
        std::string name;
        std::string kx = "0.0"; // 导热系数 X 方向
        std::string ky = "0.0"; // 导热系数 Y 方向
        std::string kz = "0.0"; // 导热系数 Z 方向
        std::string midu = "0.0"; // 密度 rho
        std::string bi_rerong = "0.0"; // 比热容 c
    };

    // 单变元函数类别（5 类），XML <Functions> 块解析为这些 POD 类型。
    // 仅在 IO 层使用：preprocessor 收到后注册到 expr 全局注册表，Model
    // 不再持有这些类型，保持模块解耦。
    enum class FunctionType { Expression, DoubleExponential, Gauss, Sine, PieceWise };

    struct ExpressionFunction {
        std::string expression; // muparser 字符串，自变量名 `x`
        double draw_min_x = 0.0;
        double draw_max_x = 100.0;
    };

    struct DoubleExponentialFunction {
        double a = 0.0;
        double alpha = 0.0;
        double beta = 0.0;
        double draw_min_x = 0.0;
        double draw_max_x = 100.0;
    };

    struct GaussFunction {
        double a = 0.0;
        double tau = 0.0;
        double x0 = 0.0;
        double draw_min_x = 0.0;
        double draw_max_x = 100.0;
    };

    struct SineFunction {
        double a = 0.0;
        double omega = 0.0;
        double phi = 0.0;
        double draw_min_x = 0.0;
        double draw_max_x = 100.0;
    };

    struct PieceWiseFunction {
        struct Point {
            double x = 0.0;
            double y = 0.0;
        };
        std::vector<Point> points;
        double draw_min_x = 0.0;
        double draw_max_x = 100.0;
    };

    struct Function {
        FunctionType type = FunctionType::Expression;
        ExpressionFunction expression;
        DoubleExponentialFunction double_exp;
        GaussFunction gauss;
        SineFunction sine;
        PieceWiseFunction piecewise;
    };

    enum class LengthUnit { M, Mm, Um, Nm, Inch, Mil };

    enum class Dimension { Dimension2D, Dimension3D };

    // 3D 探针（观察点）：用户坐标系下的固定位置，坐标以 muparser 表达式形式给出
    // （如 "chip_w/2 + 0.1"），由 preprocessor 在加载时一次性求值到 Model。
    // 求解器在每个时间步记录该点温度。
    struct ObservationPoint3D {
        std::string name;
        std::string x;
        std::string y;
        std::string z;
    };

    // ── SmartMacro model (IO-payload, no Eigen types) ──────────────────
    /// POD-based reduced model loaded from disk: modal DtN + modal RHS + POD basis.
    /// K_modal is stored as dense row-major [n_modes x n_modes].
    /// f_modal is stored as a flat vector (length n_modes).
    /// phi_basis is row-major [N_ports x n_modes].
    /// Consumers wrap with Eigen::Map for linear algebra.
    struct SmartMacroModelData {
        std::string name;
        std::vector<double> K_modal;           // row-major [n_modes x n_modes]
        std::vector<double> f_modal;           // RHS vector (size n_modes)
        std::vector<double> phi_basis;         // row-major [N_ports x n_modes]
        std::vector<int> port_ix;
        std::vector<int> port_iy;
        std::vector<int> port_iz;
        int n_modes = 0;
    };

    // 探针温度时间序列：与 ObservationPoint3D::name 一一对应，times/values 等长。
    struct ProbeTrace {
        std::string name;
        std::vector<double> times;
        std::vector<double> values;
    };

    // 流体-固体耦合: 流体 overlay 类型 (从额外 XML 解析, 不侵入主 IOStructure)
    struct FluidMaterialOverlay {
        std::string name; // 与 IOStructure::materials 中已有材料同名
        std::string dynamic_viscosity; // 动力粘度 μ [Pa·s], 表达式字符串
    };

    // 流体边界：单一 value 字段 + kind 决定物理量语义。
    // 字典化 schema; 三种 kind 互斥:
    //   PressureType     — value [Pa]    Dirichlet on p
    //   MassFlowRateType — value [kg/s]  Neumann (energy 侧覆盖 netOutflux)
    //   VelocityType     — value [m/s]   Neumann, normal to face (energy 侧覆盖)
    struct FluidBoundaryOverlay {
        std::string name;
        std::vector<std::string> face_keys; // 同格式: X|E|8|...
        FluidBCType kind = FluidBCType::None;
        double value = 0.0;
        double inlet_temperature = std::numeric_limits<double>::quiet_NaN(); // [K], NaN=未指定
    };

    struct FluidOverlay {
        std::vector<FluidMaterialOverlay> fluid_materials;
        std::vector<FluidBoundaryOverlay> boundaries;
    };

    struct IOStructure {
        StudyType study_type;
        Dimension dimension;
        LengthUnit length_unit;
        double initial_temperature = 300.0;

        std::vector<Variable> variables;
        std::vector<Layer> layers;
        std::unordered_map<std::string, Material> materials;
        std::vector<Boundary> boundaries;

        double transient_duration = 0.0;
        double transient_time_step = 1.0;
        std::string transient_time_unit = "s";

        ThermalBCType other_bc_type = ThermalBCType::SecondType;
        FirstTypeThermalBC other_bc_first;
        SecondTypeThermalBC other_bc_second;
        ThirdTypeThermalBC other_bc_third;

        // Mesh vertex coordinates from XML (Results[0].Mesh.XArray/YArray/ZArray)
        std::vector<double> mesh_vertex_x;
        std::vector<double> mesh_vertex_y;
        std::vector<double> mesh_vertex_z;

        std::unordered_map<std::string, Function> functions;

        // 3D 观察点（探针）列表，默认空：稳态 case 不会有此项。
        std::vector<ObservationPoint3D> observation_points;
    };

} // namespace mhs::core
