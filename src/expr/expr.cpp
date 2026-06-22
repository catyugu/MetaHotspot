// src/expr/expr.cpp
#include "expr.hpp"
#include <tbb/enumerable_thread_specific.h>

#include <memory>
#include <muParser.h>
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

    // Bridge state carried alongside every native function registration.
    // muparser hands us a void* to this struct on every native call.
    struct NativeFnCtx {
        FieldEvaluator fe;
        const FieldContext* ctx_ptr;
    };

    // Bridge entry: muparser evaluates each argument independently, then calls this
    // with (user_data, args_array, nargs). Forward the raw pointer straight to
    // the user's FieldEvaluator — no per-call allocation, muparser's array is
    // already in registers.
    static double native_fn_bridge(void* pUserData, const mu::value_type* args, int nargs)
    {
        auto* ctx = static_cast<NativeFnCtx*>(pUserData);
        return ctx->fe(args, nargs, *ctx->ctx_ptr);
    }

    class MuCompiled {
    public:
        explicit MuCompiled(const std::string& formula)
        {
            // Bind x/y/z/T/t to the addresses of current_ctx_ — muparser dereferences
            // the pointer on every Eval(), so writing current_ctx_ before Eval() updates
            // all five variables in one shot, just like the muparser symbol table did.
            parser_.DefineVar("x", &current_ctx_.x);
            parser_.DefineVar("y", &current_ctx_.y);
            parser_.DefineVar("z", &current_ctx_.z);
            parser_.DefineVar("T", &current_ctx_.T);
            parser_.DefineVar("t", &current_ctx_.t);

            // muparser exposes pi/e as _pi/_e by default. Re-export under the familiar names
            // (matching exprtk's behavior) so existing expressions keep working.
            parser_.DefineConst("pi", mu::MathImpl<mu::value_type>::CONST_PI);
            parser_.DefineConst("e", mu::MathImpl<mu::value_type>::CONST_E);

            {
                std::lock_guard<std::mutex> lock(detail::mutex());
                for (const auto& [name, fe] : detail::native_functions()) {
                    auto slot = std::make_shared<NativeFnCtx>();
                    slot->fe = fe;
                    slot->ctx_ptr = &current_ctx_;
                    native_slots_[name] = slot;
                    // muparser requires user data pointer to be non-null; our slot is.
                    parser_.DefineFunUserData(name, &native_fn_bridge, slot.get(), false);
                }
            }

            try {
                parser_.SetExpr(formula);
                // Force compilation up front so the caller learns about syntax errors
                // synchronously, mirroring the old `parser.compile(...)` bool contract.
                (void)parser_.Eval();
                valid_ = true;
            }
            catch (const mu::ParserError&) {
                valid_ = false;
            }
        }

        // Disable copy/move to guarantee current_ctx_'s address is stable — same
        // contract as before, since native slots hold a raw `FieldContext*`.
        MuCompiled(const MuCompiled&) = delete;
        MuCompiled& operator=(const MuCompiled&) = delete;
        MuCompiled(MuCompiled&&) = delete;
        MuCompiled& operator=(MuCompiled&&) = delete;

        bool valid() const { return valid_; }

        double eval(const FieldContext& ctx)
        {
            if (!valid_)
                return 0.0;
            current_ctx_ = ctx;
            try {
                return parser_.Eval();
            }
            catch (const mu::ParserError&) {
                return 0.0;
            }
        }

    private:
        bool valid_ = true;
        FieldContext current_ctx_; // single TLS-backing state for x/y/z/T/t

        mu::Parser parser_;
        // Heap-allocated so the address we hand to DefineFunUserData stays valid even
        // if `this` moves — though we delete move ops, the slot's identity is independent.
        std::unordered_map<std::string, std::shared_ptr<NativeFnCtx>> native_slots_;
    };

    // TLS wrapper: each thread that touches this expression gets its own MuCompiled.
    // unique_ptr keeps the AST heap address stable across ETS growth, and the formula
    // string is captured by value so the wrapper has no external lifetime dependency.
    struct MuCompiledTLS {
        tbb::enumerable_thread_specific<std::unique_ptr<MuCompiled>> tls;

        explicit MuCompiledTLS(const std::string& formula)
            : tls([formula]() { return std::make_unique<MuCompiled>(formula); })
        {
        }
    };

    CompiledExpression::CompiledExpression() : is_const_(true), const_val_(0.0) { }

    CompiledExpression::~CompiledExpression() = default;

    double CompiledExpression::eval(const FieldContext& ctx) const
    {
        if (is_const_)
            return const_val_;
        if (!tls_impl_)
            return 0.0;
        // Lock-free grab of this thread's dedicated AST
        return tls_impl_->tls.local()->eval(ctx);
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
        e.tls_impl_ = std::make_shared<MuCompiledTLS>(formula);
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

        // Main-thread trial compile: catch syntax errors early
        {
            MuCompiled test_compile(formula);
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

        mu::Parser parser;
        std::vector<std::pair<std::string, double>> active_vars;
        active_vars.reserve(vars.size());

        for (const auto& [name, val] : vars) {
            if (formula.find(name) != std::string::npos) {
                active_vars.emplace_back(name, val);
                parser.DefineVar(active_vars.back().first, &active_vars.back().second);
            }
        }

        try {
            parser.SetExpr(formula);
            return parser.Eval();
        }
        catch (const mu::ParserError&) {
            return 0.0;
        }
    }

} // namespace mhs::core
