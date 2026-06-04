#include "function_helpers.hpp"

#include "expr/expr.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace mhs::preprocessor {

    // ---- 5 类闭包构造器 ---------------------------------------------------

    FieldEvaluator make_expression_evaluator(const std::string& inner_expr)
    {
        // 内层 parse：自变量名仍叫 x（用户写的就是 x）。
        // 闭包内：把 ctx.t 喂给内层 x 槽——这是 ExpressionFunction 的设计约定，
        // 引用处字面 x 已被 preprocessor 替换为 t 或 T，exprtk 会绑到对应槽，
        // 但 native 闭包只看 ctx.t。
        auto ce = expr::parse(inner_expr);
        return [ce](const expr::FieldContext& ctx) {
            // FieldContext 顺序: (x, y, z, T, t)
            expr::FieldContext inner {ctx.t, 0.0, 0.0, 0.0, ctx.t};
            return ce.eval(inner);
        };
    }

    FieldEvaluator make_double_exp_evaluator(double A, double alpha, double beta)
    {
        return [A, alpha, beta](const expr::FieldContext& c) {
            return A * (std::exp(alpha * c.t) - std::exp(beta * c.t));
        };
    }

    FieldEvaluator make_gauss_evaluator(double A, double tau, double x0)
    {
        return [A, tau, x0](const expr::FieldContext& c) {
            double u = (c.t - x0) / tau;
            return A * std::exp(-u * u);
        };
    }

    FieldEvaluator make_sine_evaluator(double A, double omega, double phi)
    {
        return [A, omega, phi](const expr::FieldContext& c) {
            return A * std::sin(omega * c.t + phi);
        };
    }

    FieldEvaluator make_piecewise_evaluator(const std::vector<PieceWiseFunction::Point>& pts_in)
    {
        // 按 X 升序排；首尾延伸用首/末值
        std::vector<PieceWiseFunction::Point> pts = pts_in;
        std::sort(pts.begin(), pts.end(),
            [](const PieceWiseFunction::Point& a, const PieceWiseFunction::Point& b) { return a.x < b.x; });
        return [pts = std::move(pts)](const expr::FieldContext& c) {
            double x = c.t;
            if (pts.empty())
                return 0.0;
            if (x <= pts.front().x)
                return pts.front().y;
            if (x >= pts.back().x)
                return pts.back().y;
            auto it = std::upper_bound(pts.begin(), pts.end(), x,
                [](double v, const PieceWiseFunction::Point& p) { return v < p.x; });
            const auto& p1 = *(it - 1);
            const auto& p2 = *it;
            double t = (x - p1.x) / (p2.x - p1.x);
            return p1.y + t * (p2.y - p1.y);
        };
    }

    // ---- 字面替换 ---------------------------------------------------------

    namespace {

        bool is_id_char(char c)
        {
            return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
        }

        bool is_digit_char(char c)
        {
            return c >= '0' && c <= '9';
        }

        // 扫描 expr_str 中所有形如 name(...) 的引用，name 必须在 fns 中存在。
        // 返回 false 表示至少有一个未注册（已经 panic 过）。
        // 算法：从左到右找 `name(` 模式——name 是 [A-Za-z_][A-Za-z0-9_]*，前面是非 id 字符或字符串首。
        bool validate_function_names(const std::string& expr_str,
            const std::unordered_map<std::string, Function>& fns, const std::string& argname)
        {
            const size_t n = expr_str.size();
            size_t i = 0;
            while (i < n) {
                char c = expr_str[i];
                if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_') {
                    size_t start = i;
                    while (i < n && (is_id_char(expr_str[i]) || is_digit_char(expr_str[i])))
                        i++;
                    std::string name = expr_str.substr(start, i - start);
                    if (i < n && expr_str[i] == '(') {
                        if (name != argname && name != "x" && name != "T" && name != "t"
                            && name != "pi" && name != "e" && fns.find(name) == fns.end()) {
                            throw std::runtime_error("unknown function '" + name
                                + "' referenced in '" + expr_str
                                + "': must be declared in <Functions>");
                        }
                    }
                }
                else {
                    i++;
                }
            }
            return true;
        }

    } // namespace

    std::string substitute_function_args(const std::string& expr_str, const std::string& argname,
        const std::unordered_map<std::string, Function>& fns)
    {
        // 1) 校验所有 name(...) 引用
        validate_function_names(expr_str, fns, argname);

        // 2) 单次扫描，记录每个"孤立 x"位置
        const size_t n = expr_str.size();
        std::string out;
        out.reserve(n + 8);
        std::vector<size_t> x_positions;
        for (size_t i = 0; i < n; i++) {
            if (expr_str[i] == 'x') {
                bool left_ok = (i == 0) || (!is_id_char(expr_str[i - 1]));
                bool right_ok = (i + 1 == n) || (!is_id_char(expr_str[i + 1]));
                if (left_ok && right_ok) {
                    x_positions.push_back(i);
                }
            }
        }

        // 3) 倒序把孤立 x 替换为 argname（避免位置偏移）
        out = expr_str;
        for (auto it = x_positions.rbegin(); it != x_positions.rend(); ++it) {
            out.replace(*it, 1, argname);
        }
        return out;
    }

    // ---- 注册全部 native --------------------------------------------------

    std::unordered_map<std::string, std::string> register_all_functions(
        const std::unordered_map<std::string, Function>& fns)
    {
        std::unordered_map<std::string, std::string> names;
        names.reserve(fns.size());
        for (const auto& [key, fn] : fns) {
            FieldEvaluator ev = nullptr;
            switch (fn.type) {
            case FunctionType::Expression:
                ev = make_expression_evaluator(fn.expression.expression);
                break;
            case FunctionType::DoubleExponential:
                ev = make_double_exp_evaluator(fn.double_exp.a, fn.double_exp.alpha, fn.double_exp.beta);
                break;
            case FunctionType::Gauss:
                ev = make_gauss_evaluator(fn.gauss.a, fn.gauss.tau, fn.gauss.x0);
                break;
            case FunctionType::Sine:
                ev = make_sine_evaluator(fn.sine.a, fn.sine.omega, fn.sine.phi);
                break;
            case FunctionType::PieceWise:
                ev = make_piecewise_evaluator(fn.piecewise.points);
                break;
            }
            expr::register_native(key, std::move(ev));
            names[key] = "native";
        }
        return names;
    }

} // namespace mhs::preprocessor
