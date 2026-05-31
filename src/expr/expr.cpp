#include "expr.hpp"
#define exprtk_disable_caseinsensitivity
#include <exprtk/exprtk.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mhs::expr {

    // Internal registry (thread-safe)
    namespace registry {
        std::mutex& mutex()
        {
            static std::mutex m;
            return m;
        }

        std::unordered_map<std::string, double>& variables()
        {
            static std::unordered_map<std::string, double> vars;
            return vars;
        }

        std::unordered_map<std::string, FieldEvaluator>& native_functions()
        {
            static std::unordered_map<std::string, FieldEvaluator> funcs;
            return funcs;
        }

        std::unordered_map<std::string, std::string>& user_functions()
        {
            static std::unordered_map<std::string, std::string> funcs;
            return funcs;
        }
    } // namespace registry

    // Pre-compiled exprtk expression (thread-safe, reusable)
    class ExprTKCompiled {
    public:
        explicit ExprTKCompiled(const std::string& formula)
        {
            using namespace exprtk;

            sym_table_ = std::make_unique<symbol_table<double>>();
            expr_ = std::make_unique<expression<double>>();
            parser_ = std::make_unique<parser<double>>();

            // Add context variables
            sym_table_->add_variable("x", x_);
            sym_table_->add_variable("y", y_);
            sym_table_->add_variable("z", z_);
            sym_table_->add_variable("T", T_);
            sym_table_->add_variable("t", t_);

            expr_->register_symbol_table(*sym_table_);

            if (!parser_->compile(formula, *expr_)) {
                valid_ = false;
            }
            else {
                valid_ = true;
            }
        }

        bool valid() const { return valid_; }

        double eval(const FieldContext& ctx)
        {
            if (!valid_) {
                return 0.0;
            }
            // Not thread-safe. For parallel evaluation, create per-thread instances.
            x_ = ctx.x;
            y_ = ctx.y;
            z_ = ctx.z;
            T_ = ctx.T;
            t_ = ctx.t;
            return expr_->value();
        }

    private:
        bool valid_ = true;
        double x_ = 0, y_ = 0, z_ = 0, T_ = 0, t_ = 0;
        std::unique_ptr<exprtk::symbol_table<double>> sym_table_;
        std::unique_ptr<exprtk::expression<double>> expr_;
        std::unique_ptr<exprtk::parser<double>> parser_;
    };

    // Thread-safe registry operations
    void set_variable(const std::string& name, double value)
    {
        std::lock_guard<std::mutex> lock(registry::mutex());
        registry::variables()[name] = value;
    }

    void register_native(const std::string& name, FieldEvaluator func)
    {
        std::lock_guard<std::mutex> lock(registry::mutex());
        registry::native_functions()[name] = std::move(func);
    }

    void register_function(const std::string& name, const std::string& expression)
    {
        std::lock_guard<std::mutex> lock(registry::mutex());
        registry::user_functions()[name] = expression;
    }

    FieldEvaluator get_native(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(registry::mutex());
        auto it = registry::native_functions().find(name);
        if (it != registry::native_functions().end()) {
            return it->second;
        }
        return nullptr;
    }

    void clear_registry()
    {
        std::lock_guard<std::mutex> lock(registry::mutex());
        registry::variables().clear();
        registry::native_functions().clear();
        registry::user_functions().clear();
    }

    // Internal: pre-compiled exprtk expression holder
    struct PrecompiledExpr {
        std::unique_ptr<ExprTKCompiled> impl;
        bool is_const = false;
        double const_val = 0.0;
    };

    // Cache for compiled expressions (thread-safe)
    static std::unordered_map<std::string, PrecompiledExpr>& expr_cache()
    {
        static std::unordered_map<std::string, PrecompiledExpr> cache;
        return cache;
    }

    static std::mutex& cache_mutex()
    {
        static std::mutex m;
        return m;
    }

    CompiledExpression parse(const std::string& formula)
    {
        // Quick check for constant
        {
            char* end = nullptr;
            double val = std::strtod(formula.c_str(), &end);
            if (end != formula.c_str() && *end == '\0') {
                return CompiledExpression::make_constant(val);
            }
        }

        std::lock_guard<std::mutex> lock(cache_mutex());

        // Check cache
        auto it = expr_cache().find(formula);
        if (it != expr_cache().end()) {
            const auto& cached = it->second;
            if (cached.is_const) {
                return CompiledExpression::make_constant(cached.const_val);
            }
            return CompiledExpression::make_evaluator(
                [impl = it->second.impl.get()](const FieldContext& ctx) {
                    return impl ? impl->eval(ctx) : 0.0;
                });
        }

        // Compile and cache
        auto precompiled = std::make_unique<ExprTKCompiled>(formula);
        PrecompiledExpr entry;
        entry.impl = std::move(precompiled);

        expr_cache()[formula] = std::move(entry);
        auto& cached = expr_cache()[formula];

        if (cached.is_const) {
            return CompiledExpression::make_constant(cached.const_val);
        }

        return CompiledExpression::make_evaluator(
            [impl = cached.impl.get()](const FieldContext& ctx) {
                return impl ? impl->eval(ctx) : 0.0;
            });
    }

    double eval_geometry(const std::string& formula)
    {
        std::lock_guard<std::mutex> lock(registry::mutex());

        const auto& vars = registry::variables();

        // Direct variable lookup
        auto var_it = vars.find(formula);
        if (var_it != vars.end()) {
            return var_it->second;
        }

        using namespace exprtk;
        expression<double> exprtk_expr;
        symbol_table<double> sym_table;

        std::vector<std::pair<std::string, double>> active_vars;
        active_vars.reserve(vars.size());

        for (const auto& [name, val] : vars) {
            if (formula.find(name) != std::string::npos) {
                active_vars.emplace_back(name, val);
                sym_table.add_variable(active_vars.back().first, active_vars.back().second);
            }
        }

        exprtk_expr.register_symbol_table(sym_table);

        parser<double> parser;

        // 直接编译未经替换的、原始的 formula 字符串
        if (parser.compile(formula, exprtk_expr)) {
            return exprtk_expr.value();
        }

        return 0.0;
    }

} // namespace mhs::expr