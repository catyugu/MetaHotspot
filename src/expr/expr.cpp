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

    // ExprTKCompiled: owns its own symbol table and mutable context variables
    class ExprTKCompiled {
    public:
        explicit ExprTKCompiled(const std::string& formula)
        {
            using namespace exprtk;

            sym_table_ = std::make_unique<symbol_table<double>>();
            expr_ = std::make_unique<expression<double>>();

            sym_table_->add_variable("x", x_);
            sym_table_->add_variable("y", y_);
            sym_table_->add_variable("z", z_);
            sym_table_->add_variable("T", T_);
            sym_table_->add_variable("t", t_);

            expr_->register_symbol_table(*sym_table_);

            parser<double> parser;
            valid_ = parser.compile(formula, *expr_);
        }

        bool valid() const { return valid_; }

        double eval(const FieldContext& ctx)
        {
            if (!valid_)
                return 0.0;
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
    };

    // CompiledExpression implementation
    CompiledExpression::CompiledExpression() : is_const_(true), const_val_(0.0) { }

    CompiledExpression::~CompiledExpression() = default;

    CompiledExpression::CompiledExpression(CompiledExpression&& other) noexcept
        : is_const_(other.is_const_), const_val_(other.const_val_), impl_(std::move(other.impl_))
    {
        other.is_const_ = true;
        other.const_val_ = 0.0;
    }

    CompiledExpression& CompiledExpression::operator=(CompiledExpression&& other) noexcept
    {
        if (this != &other) {
            is_const_ = other.is_const_;
            const_val_ = other.const_val_;
            impl_ = std::move(other.impl_);
            other.is_const_ = true;
            other.const_val_ = 0.0;
        }
        return *this;
    }

    double CompiledExpression::eval(const FieldContext& ctx) const
    {
        if (is_const_)
            return const_val_;
        if (!impl_)
            return 0.0;
        return impl_->eval(ctx);
    }

    CompiledExpression CompiledExpression::make_constant(double value)
    {
        CompiledExpression e;
        e.is_const_ = true;
        e.const_val_ = value;
        return e;
    }

    CompiledExpression CompiledExpression::make_evaluator(std::unique_ptr<ExprTKCompiled> impl)
    {
        CompiledExpression e;
        e.is_const_ = false;
        e.const_val_ = 0.0;
        e.impl_ = std::move(impl);
        return e;
    }

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

        auto impl = std::make_unique<ExprTKCompiled>(formula);
        if (!impl->valid()) {
            return CompiledExpression::make_constant(0.0);
        }
        return CompiledExpression::make_evaluator(std::move(impl));
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
        if (parser.compile(formula, exprtk_expr)) {
            return exprtk_expr.value();
        }

        return 0.0;
    }

} // namespace mhs::expr