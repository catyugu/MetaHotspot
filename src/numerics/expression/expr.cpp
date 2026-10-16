// muparser-backed expression implementation
#include "numerics/expression/expr.hpp"
#include <tbb/enumerable_thread_specific.h>

#include <memory>
#include <muParser.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mhs::core {

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
        // Fixed slots keep muparser's user-data pointers stable for this instance.
        explicit MuCompiled(const std::string& formula, const SymbolTable& symbols)
        {
            // Bind x/y/z/T/t to the addresses of current_ctx_ — muparser dereferences
            // the pointer on every Eval(), so writing current_ctx_ before Eval() updates
            // all five variables in one shot, just like the muparser symbol table did.
            parser_.DefineVar("x", &current_ctx_.x);
            parser_.DefineVar("y", &current_ctx_.y);
            parser_.DefineVar("z", &current_ctx_.z);
            parser_.DefineVar("T", &current_ctx_.T);
            parser_.DefineVar("t", &current_ctx_.t);

            // Re-export pi/e under familiar names (muparser exposes them as _pi/_e by default).
            parser_.DefineConst("pi", mu::MathImpl<mu::value_type>::CONST_PI);
            parser_.DefineConst("e", mu::MathImpl<mu::value_type>::CONST_E);

            native_slots_.reserve(symbols.natives.size());
            for (const auto& [name, fe] : symbols.natives) {
                native_slots_.push_back({fe, &current_ctx_});
                parser_.DefineFunUserData(name, &native_fn_bridge, &native_slots_.back(), false);
            }

            try {
                parser_.SetExpr(formula);
                // muparser compiles on first Eval(). Never publish a failed parser.
                (void)parser_.Eval();
            }
            catch (const mu::ParserError& e) {
                throw std::runtime_error("expression compile error: '" + formula + "': " + e.GetMsg());
            }
        }

        // Disable copy/move to guarantee current_ctx_'s address is stable — same
        // contract as before, since native slots hold a raw `FieldContext*`.
        MuCompiled(const MuCompiled&) = delete;
        MuCompiled& operator=(const MuCompiled&) = delete;
        MuCompiled(MuCompiled&&) = delete;
        MuCompiled& operator=(MuCompiled&&) = delete;

        double eval(const FieldContext& ctx)
        {
            current_ctx_ = ctx;
            try {
                return parser_.Eval();
            }
            catch (const mu::ParserError& e) {
                throw std::runtime_error(std::string("expression eval error: ") + e.GetMsg());
            }
        }

    private:
        FieldContext current_ctx_; // single TLS-backing state for x/y/z/T/t

        std::vector<NativeFnCtx> native_slots_;
        mu::Parser parser_;
    };

    // TLS wrapper: each thread that touches this expression gets its own MuCompiled.
    // unique_ptr keeps the AST heap address stable across ETS growth. The SymbolTable
    // is captured by value so the wrapper has no external lifetime dependency.
    struct MuCompiledTLS {
        SymbolTable symbols;
        tbb::enumerable_thread_specific<std::unique_ptr<MuCompiled>> tls;

        MuCompiledTLS(const std::string& formula, SymbolTable sym)
            : symbols(std::move(sym)), tls([formula, this]() { return std::make_unique<MuCompiled>(formula, symbols); })
        {
            (void)tls.local(); // Validate immediately and reuse this parser on the creating thread.
        }
    };

    CompiledExpression::CompiledExpression() = default;

    CompiledExpression::~CompiledExpression() = default;

    double CompiledExpression::eval(const FieldContext& ctx) const
    {
        if (const auto* constant = std::get_if<double>(&value_))
            return *constant;
        return std::get<std::shared_ptr<MuCompiledTLS>>(value_)->tls.local()->eval(ctx);
    }

    CompiledExpression CompiledExpression::make_constant(double value)
    {
        CompiledExpression e;
        e.value_ = value;
        return e;
    }

    CompiledExpression CompiledExpression::make_evaluator(const std::string& formula, const SymbolTable& symbols)
    {
        CompiledExpression e;
        e.value_ = std::make_shared<MuCompiledTLS>(formula, symbols);
        return e;
    }

    CompiledExpression parse(const std::string& formula, const SymbolTable& symbols)
    {
        if (formula.empty())
            return CompiledExpression::make_constant(0.0);

        char* end = nullptr;
        double val = std::strtod(formula.c_str(), &end);
        if (end != formula.c_str() && *end == '\0') {
            return CompiledExpression::make_constant(val);
        }

        return CompiledExpression::make_evaluator(formula, symbols);
    }

    double eval_geometry(const std::string& formula, const SymbolTable& symbols)
    {
        if (formula.empty())
            return 0.0;

        const auto& vars = symbols.variables;

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
        catch (const mu::ParserError& e) {
            throw std::runtime_error(std::string("geometry eval error: '") + formula + "': " + e.GetMsg());
        }
    }

} // namespace mhs::core
