#include "expr.hpp"
#define exprtk_disable_caseinsensitivity
#include <tbb/enumerable_thread_specific.h>

#include <exprtk/exprtk.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mhs::core {

    namespace detail { // registry state: cross-file internal within mhs::core
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
    } // namespace detail

    namespace {
        // FieldContext 顺序: (x, y, z, T, t)
        constexpr size_t kFieldSlots = 5;
    }

    template <typename T> class NativeFn : public exprtk::ifunction<T> {
    public:
        explicit NativeFn(FieldEvaluator fe)
            : exprtk::ifunction<T>(1) // 1 argument
            , fe_(std::move(fe))
        {
        }

        T operator()(const T& arg0) override
        {
            // Build a fresh ctx per call with arg0 in every slot. No hidden
            // global / TLS state — the value flows through exprtk's argument
            // mechanism, which is the right depth.
            const T slots[kFieldSlots] = {arg0, arg0, arg0, arg0, arg0};
            FieldContext ctx {slots[0], slots[1], slots[2], slots[3], slots[4]};
            return static_cast<T>(fe_(ctx));
        }

    private:
        FieldEvaluator fe_;
    };

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

            // Bind every registered native function name as a 1-arg exprtk function.
            // Slot lifetime must outlive the symbol table / expression.
            {
                std::lock_guard<std::mutex> lock(detail::mutex());
                for (const auto& [name, fe] : detail::native_functions()) {
                    auto slot = std::make_shared<NativeFn<double>>(fe);
                    native_slots_[name] = slot;
                    sym_table_->add_function(name, *slot);
                }
            }

            expr_->register_symbol_table(*sym_table_);

            parser<double> parser;
            valid_ = parser.compile(formula, *expr_);
        }

        ExprTKCompiled(ExprTKCompiled&&) = default;
        ExprTKCompiled& operator=(ExprTKCompiled&&) = default;

        bool valid() const { return valid_; }

        double eval(const FieldContext& ctx)
        {
            if (!valid_)
                return 0.0;
            // 这里的状态修改现在是线程安全的，因为每个线程拥有自己独立的 ExprTKCompiled 副本
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
        // Owns the NativeFn objects bound into the symbol table. They must live
        // as long as the symbol table is in use.
        std::unordered_map<std::string, std::shared_ptr<NativeFn<double>>> native_slots_;
    };

    // TLS 包装器：确保每个访问表达式的线程都能获得一个独立的 AST
    struct ExprTKCompiledTLS {
        tbb::enumerable_thread_specific<ExprTKCompiled> tls;

        explicit ExprTKCompiledTLS(const std::string& formula)
            : tls([formula]() { return ExprTKCompiled(formula); }) { }
    };

    CompiledExpression::CompiledExpression() : is_const_(true), const_val_(0.0) { }

    CompiledExpression::~CompiledExpression() = default;

    double CompiledExpression::eval(const FieldContext& ctx) const
    {
        if (is_const_)
            return const_val_;
        if (!tls_impl_)
            return 0.0;
        // 无锁获取当前线程的专属 AST 副本进行求值
        return tls_impl_->tls.local().eval(ctx);
    }

    CompiledExpression CompiledExpression::make_constant(double value)
    {
        CompiledExpression e;
        e.is_const_ = true;
        e.const_val_ = value;
        return e;
    }

    CompiledExpression CompiledExpression::make_evaluator(const std::string& formula)
    {
        CompiledExpression e;
        e.is_const_ = false;
        e.tls_impl_ = std::make_shared<ExprTKCompiledTLS>(formula);
        return e;
    }

    void set_variable(const std::string& name, double value)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        detail::variables()[name] = value;
    }

    void register_native(const std::string& name, FieldEvaluator func)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        detail::native_functions()[name] = std::move(func);
    }

    void register_function(const std::string& name, const std::string& expression)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        detail::user_functions()[name] = expression;
    }

    FieldEvaluator get_native(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        auto it = detail::native_functions().find(name);
        if (it != detail::native_functions().end()) {
            return it->second;
        }
        return nullptr;
    }

    void clear_registry()
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        detail::variables().clear();
        detail::native_functions().clear();
        detail::user_functions().clear();
    }

    CompiledExpression parse(const std::string& formula)
    {
        char* end = nullptr;
        double val = std::strtod(formula.c_str(), &end);
        if (end != formula.c_str() && *end == '\0') {
            return CompiledExpression::make_constant(val);
        }

        // 主线程进行一次试编译，尽早捕获语法错误
        {
            ExprTKCompiled test_compile(formula);
            if (!test_compile.valid()) {
                return CompiledExpression::make_constant(0.0);
            }
        }
        return CompiledExpression::make_evaluator(formula);
    }

    double eval_geometry(const std::string& formula)
    {
        std::lock_guard<std::mutex> lock(detail::mutex());
        const auto& vars = detail::variables();

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

} // namespace mhs::core
