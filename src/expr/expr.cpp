#include "expr.hpp"
#include <exprtk/exprtk.hpp>
#include <mutex>
#include <unordered_map>

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
        }

        bool valid() const { return valid_; }

        double eval(const FieldContext& ctx)
        {
            if (!valid_) {
                return 0.0;
            }
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

        // Try to evaluate as expression with substituted variables
        std::string expr = formula;

        // Sort by length descending to avoid partial replacements
        std::vector<std::pair<std::string, double>> sorted_vars;
        for (const auto& [name, val] : vars) {
            sorted_vars.emplace_back(name, val);
        }
        std::sort(sorted_vars.begin(), sorted_vars.end(),
            [](const auto& a, const auto& b) { return a.first.length() > b.first.length(); });

        for (const auto& [name, val] : sorted_vars) {
            size_t pos;
            while ((pos = expr.find(name)) != std::string::npos) {
                expr.replace(pos, name.length(), std::to_string(val));
            }
        }

        // Try to compile and evaluate with exprtk
        using namespace exprtk;
        expression<double> exprtk_expr;
        symbol_table<double> sym_table;

        // Add variables
        std::vector<std::pair<std::string, double>> active_vars;
        for (const auto& [name, val] : sorted_vars) {
            if (expr.find(name) != std::string::npos) {
                double v = val;
                sym_table.add_variable(name, v);
                active_vars.emplace_back(name, v);
            }
        }

        exprtk_expr.register_symbol_table(sym_table);

        parser<double> parser;
        if (parser.compile(expr, exprtk_expr)) {
            return exprtk_expr.value();
        }

        return 0.0;
    }

} // namespace mhs::expr