#pragma once

#include <limits>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "types.hpp"

namespace mhs::core {

    struct Variable {
        std::string name;
        std::string value;
    };

    struct Rect {
        bool add_sub = true;
        std::string width_expr;
        std::string height_expr;
        std::string x_expr;
        std::string y_expr;
    };

    struct Block {
        std::vector<Rect> all_rects;
        std::string material_name;
        std::string x_offset_expr;
        std::string y_offset_expr;
        std::string ti_reyuan_expr; // 体热源表达式 [W/m³]
        std::string thickness_expr;
    };

    struct Layer {
        std::vector<Block> blocks;
        std::string thickness_expr;
        std::string x_offset_expr;
        std::string y_offset_expr;
    };

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
        std::vector<std::string> face_keys;
        std::variant<FirstTypeThermalBC, SecondTypeThermalBC, ThirdTypeThermalBC> bc;
    };

    struct Material {
        std::string kx = "0.0"; // 导热系数 X 方向
        std::string ky = "0.0"; // 导热系数 Y 方向
        std::string kz = "0.0"; // 导热系数 Z 方向
        std::string midu = "0.0"; // 密度 rho
        std::string bi_rerong = "0.0"; // 比热容 c
        // Empty means solid. A non-empty expression marks this material as fluid.
        std::string dynamic_viscosity;
    };

    // 单变元函数类别（5 类）。XML 可解析为这些 POD，外部代码也可直接构造；
    // build_model() 将其注册到本次构建的本地 SymbolTable。
    struct ExpressionFunction {
        std::string expression; // muparser 字符串，自变量名 `x`
    };

    struct DoubleExponentialFunction {
        double a = 0.0;
        double alpha = 0.0;
        double beta = 0.0;
    };

    struct GaussFunction {
        double a = 0.0;
        double tau = 0.0;
        double x0 = 0.0;
    };

    struct SineFunction {
        double a = 0.0;
        double omega = 0.0;
        double phi = 0.0;
    };

    struct PieceWiseFunction {
        struct Point {
            double x = 0.0;
            double y = 0.0;
        };
        std::vector<Point> points;
    };

    using Function
        = std::variant<ExpressionFunction, DoubleExponentialFunction, GaussFunction, SineFunction, PieceWiseFunction>;

    enum class LengthUnit { M, Mm, Um, Nm, Inch, Mil };

    // 3D 探针（观察点）：用户坐标系下的固定位置，坐标以 muparser 表达式形式给出
    // （如 "chip_w/2 + 0.1"），由 build_model() 一次性求值到 Model。
    // 求解器在每个时间步记录该点温度。
    struct ObservationPoint3D {
        std::string name;
        std::string x;
        std::string y;
        std::string z;
    };

    // 流体边界：单一 value 字段 + kind 决定物理量语义。
    // 字典化 schema; 三种 kind 互斥:
    //   PressureType     — value [Pa]    Dirichlet on p
    //   MassFlowRateType — value [kg/s]  Neumann (energy 侧覆盖 netOutflux)
    //   VelocityType     — value [m/s]   Neumann, normal to face (energy 侧覆盖)
    struct FluidBoundary {
        std::vector<std::string> face_keys; // 同格式: X|E|8|...
        FluidBCType kind = FluidBCType::None;
        double value = 0.0;
        double inlet_temperature = std::numeric_limits<double>::quiet_NaN(); // [K], NaN=未指定
    };

    struct ModelDefinition {
        StudyType study_type = StudyType::Steady;
        LengthUnit length_unit = LengthUnit::M;
        double initial_temperature = 300.0;

        std::vector<Variable> variables;
        std::vector<Layer> layers;
        std::unordered_map<std::string, Material> materials;
        std::vector<Boundary> boundaries;

        double transient_duration = 0.0;
        double transient_time_step = 1.0;

        std::variant<FirstTypeThermalBC, SecondTypeThermalBC, ThirdTypeThermalBC> other_bc = SecondTypeThermalBC {};

        // Input mesh vertex coordinates in length_unit.
        std::vector<double> mesh_vertex_x;
        std::vector<double> mesh_vertex_y;
        std::vector<double> mesh_vertex_z;

        std::unordered_map<std::string, Function> functions;

        // 3D 观察点（探针）列表，默认空：稳态 case 不会有此项。
        std::vector<ObservationPoint3D> observation_points;

        // Fluid data is part of the model definition. XML overlays merge into
        // materials.dynamic_viscosity and this boundary list before build_model().
        std::vector<FluidBoundary> fluid_boundaries;
    };

} // namespace mhs::core
