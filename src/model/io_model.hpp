#pragma once

#include "types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace mhs::model {

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

    struct Block {
        std::vector<Rect> all_rects;
        std::string material_name;
        std::string thickness_expr;
        std::string mesh_size_x_expr;
        std::string mesh_size_y_expr;
        std::string mesh_size_z_expr;
        std::string x_offset_expr;
        std::string y_offset_expr;
        std::string z_offset_expr;
        std::string ti_reyuan_expr; // 体热源表达式 [W/m³]
        std::string name;
        bool is_normal_material = true;
    };

    struct Layer {
        std::vector<Block> blocks;
        std::string name;
        std::string thickness_expr;
        std::string mesh_size_x_expr;
        std::string mesh_size_y_expr;
        std::string mesh_size_z_expr;
        std::string x_offset_expr;
        std::string y_offset_expr;
        std::string period_width_expr;
        int period_width = 10;
        bool is_top_layer = false;
    };

    enum class BoundaryCategory { Electrical };

    enum class ThermalBCType { FirstType,
        SecondType,
        ThirdType };

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
        std::string daore_xishu = "0.0"; // 导热系数 k
        std::string midu = "0.0"; // 密度 rho
        std::string bi_rerong = "0.0"; // 比热容 c
    };

    enum class LengthUnit { M,
        Mm,
        Um,
        Nm,
        Inch,
        Mil
    };

    enum class Dimension { Dimension2D,
        Dimension3D };

    struct IOStructure {
        StudyType study_type;
        Dimension dimension;
        LengthUnit length_unit;
        double initial_temperature = 300.0;
        double ambient_temperature = 300.0;

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

        std::vector<double> result_values;
        std::vector<double> result_x;
        std::vector<double> result_y;
        std::vector<double> result_z;
    };

    // Expression functions
    enum class FunctionType { Expression,
        DoubleExponential,
        Gauss,
        Sine,
        PieceWise };

    struct ExpressionFunction {
        std::string expression;
        double draw_min_x = 0.0;
        double draw_max_x = 100.0;
    };

    struct DoubleExponentialFunction {
        double a = 0.0, alpha = 0.0, beta = 0.0;
        double draw_min_x = 0.0, draw_max_x = 100.0;
    };

    struct GaussFunction {
        double a = 0.0, tau = 0.0, x0 = 0.0;
        double draw_min_x = 0.0, draw_max_x = 100.0;
    };

    struct SineFunction {
        double a = 0.0, omega = 0.0, phi = 0.0;
        double draw_min_x = 0.0, draw_max_x = 100.0;
    };

    struct PieceWiseFunction {
        struct Point {
            double x = 0.0, y = 0.0;
        };
        std::vector<Point> points;
        double draw_min_x = 0.0, draw_max_x = 100.0;
    };

    struct Function {
        std::string key;
        FunctionType type;
        ExpressionFunction expression;
        DoubleExponentialFunction double_exp;
        GaussFunction gauss;
        SineFunction sine;
        PieceWiseFunction piecewise;
    };

} // namespace mhs::model
