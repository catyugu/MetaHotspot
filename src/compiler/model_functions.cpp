#include "compiler/model_functions.hpp"
#include "logging/logger.hpp"
#include "numerics/expression/expr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace mhs::sim {

    // ---- 5 类闭包构造器 ---------------------------------------------------

    // 闭包对每次 muparser 调用的约定：
    //   - args 是 muparser 已求好值的实参数组（nargs >= 1），单变元函数族从 args[0] 读自变量
    //   - ctx 是当前物理上下文（x, y, z, T, t 的真实值），仅作为参考
    // 单变元函数族（除 make_expression_evaluator 外）不参考 ctx。

    mhs::core::FieldEvaluator make_expression_evaluator(
        const std::string& inner_expr, const mhs::core::SymbolTable& symbols)
    {
        auto ce = mhs::core::parse(inner_expr, symbols);
        return [ce](const double* args, int nargs, const mhs::core::FieldContext& ctx) {
            mhs::core::FieldContext effective_ctx = ctx;
            if (nargs > 0) {
                effective_ctx.x = args[0];
            }
            return ce.eval(effective_ctx);
        };
    }
    mhs::core::FieldEvaluator make_double_exp_evaluator(double A, double alpha, double beta)
    {
        return [A, alpha, beta](const double* args, int /*nargs*/, const mhs::core::FieldContext& /*c*/) {
            double t_val = args[0];
            return A * (std::exp(alpha * t_val) - std::exp(beta * t_val));
        };
    }

    mhs::core::FieldEvaluator make_gauss_evaluator(double A, double tau, double x0)
    {
        return [A, tau, x0](const double* args, int /*nargs*/, const mhs::core::FieldContext& /*c*/) {
            double t_val = args[0];
            double u = (t_val - x0) / tau;
            return A * std::exp(-u * u);
        };
    }

    mhs::core::FieldEvaluator make_sine_evaluator(double A, double omega, double phi)
    {
        return [A, omega, phi](const double* args, int /*nargs*/, const mhs::core::FieldContext& /*c*/) {
            double t_val = args[0];
            return A * std::sin(omega * t_val + phi);
        };
    }

    mhs::core::FieldEvaluator make_piecewise_evaluator(std::vector<mhs::model::PiecewiseFunctionSpec::Point> pts)
    {
        // Points are pre-sorted by X in the IO parser, so no resort needed.
        return [pts = std::move(pts)](const double* args, int /*nargs*/, const mhs::core::FieldContext& /*c*/) {
            double x = args[0];
            if (pts.empty())
                return 0.0;
            if (x <= pts.front().x)
                return pts.front().y;
            if (x >= pts.back().x)
                return pts.back().y;
            auto it = std::upper_bound(pts.begin(), pts.end(), x,
                [](double v, const mhs::model::PiecewiseFunctionSpec::Point& p) { return v < p.x; });
            const auto& p1 = *(it - 1);
            const auto& p2 = *it;
            double t = (x - p1.x) / (p2.x - p1.x);
            return p1.y + t * (p2.y - p1.y);
        };
    }

    mhs::core::FieldEvaluator make_periodic_piecewise_constant_evaluator(double period, std::vector<double> values)
    {
        return [period, values = std::move(values)](
                   const double* args, int /*nargs*/, const mhs::core::FieldContext& /*c*/) {
            double t_val = args[0];
            if (values.empty())
                return 0.0;
            const std::size_t n = values.size();
            if (period <= 0.0 || n == 1)
                return values.front();

            // 每个完整周期 period 对应一个 values[i]，
            // 第 i 个周期 [i*period, (i+1)*period) 取 values[i % n]
            // cycle = floor(t / period)，C++ 整数除法向零截断不适合负数，
            // 所以手动用 std::floor。
            const double ratio = t_val / period;
            const auto cycle = static_cast<std::int64_t>(std::floor(ratio));
            auto idx = cycle % static_cast<std::int64_t>(n);
            if (idx < 0)
                idx += static_cast<std::int64_t>(n);
            return values[static_cast<std::size_t>(idx)];
        };
    }

    // ---- 字面替换 ---------------------------------------------------------

    namespace {

        // muparser 内建符号（变量 + 常量），引用处出现这些名字不算"未注册函数"。
        // 仅作宽松白名单；真正的语法校验由后续 mhs::core::parse 负责。
        bool is_known_builtin(std::string_view name)
        {
            return name == "x" || name == "y" || name == "z" || name == "T" || name == "t" || name == "pi"
                || name == "e";
        }

        // identifier-char 判定：[A-Za-z0-9_]
        bool is_id_char(char c)
        {
            return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
        }

        // identifier-start 判定：[A-Za-z_]（与 is_id_char 的区别在数字）
        bool is_id_start(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; }

        bool has_function(const std::vector<mhs::model::NamedFunction>& functions, std::string_view name)
        {
            return std::any_of(functions.begin(), functions.end(),
                [&](const mhs::model::NamedFunction& function) { return function.name == name; });
        }

    } // namespace

    std::string substitute_function_args(const std::string& expr_str, const std::string& argname,
        const std::vector<mhs::model::NamedFunction>& functions)
    {
        // 单次扫描：找到每个 identifier-start → 读到 identifier 末尾 →
        //   1) 若紧跟 `(` 且不在白名单也不在 fns → panic
        //   2) 否则是裸 x 且 argname 槽匹配 → 写入 argname
        //   3) 其余原样拷贝
        const size_t n = expr_str.size();
        std::string out;
        out.reserve(n);
        size_t i = 0;
        while (i < n) {
            char c = expr_str[i];
            if (is_id_start(c)) {
                size_t start = i;
                while (i < n && is_id_char(expr_str[i]))
                    i++;
                std::string_view name(expr_str.data() + start, i - start);
                if (i < n && expr_str[i] == '(') {
                    if (!is_known_builtin(name) && !has_function(functions, name)) {
                        MHS_LOG_WARN(
                            "unknown function {} referenced in {}. returning raw string", std::string(name), expr_str);
                        return std::string(expr_str);
                    }
                    out.append(expr_str, start, i - start);
                }
                else if (name == "x") {
                    out += argname;
                }
                else {
                    out.append(expr_str, start, i - start);
                }
            }
            else {
                out += c;
                i++;
            }
        }
        return out;
    }

    // ---- 写入 SymbolTable -------------------------------------------------

    void register_all_functions(
        mhs::core::SymbolTable& symbols, const std::vector<mhs::model::NamedFunction>& functions)
    {
        for (const auto& function : functions) {
            mhs::core::FieldEvaluator ev = nullptr;
            std::visit(
                [&](const auto& variant) {
                    using T = std::decay_t<decltype(variant)>;
                    if constexpr (std::is_same_v<T, mhs::model::ExpressionFunctionSpec>) {
                        ev = make_expression_evaluator(variant.expression, symbols);
                    }
                    else if constexpr (std::is_same_v<T, mhs::model::DoubleExponentialFunctionSpec>) {
                        ev = make_double_exp_evaluator(variant.amplitude, variant.alpha, variant.beta);
                    }
                    else if constexpr (std::is_same_v<T, mhs::model::GaussFunctionSpec>) {
                        ev = make_gauss_evaluator(variant.amplitude, variant.tau, variant.center);
                    }
                    else if constexpr (std::is_same_v<T, mhs::model::SineFunctionSpec>) {
                        ev = make_sine_evaluator(variant.amplitude, variant.angular_frequency, variant.phase);
                    }
                    else if constexpr (std::is_same_v<T, mhs::model::PiecewiseFunctionSpec>) {
                        ev = make_piecewise_evaluator(variant.points);
                    }
                    else if constexpr (std::is_same_v<T, mhs::model::PeriodicPiecewiseConstantFunctionSpec>) {
                        ev = make_periodic_piecewise_constant_evaluator(variant.period, variant.values);
                    }
                },
                function.value);
            symbols.natives[function.name] = std::move(ev);
        }
    }

} // namespace mhs::sim
