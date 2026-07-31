// muparser-backed expression API
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace mhs::core {

    // FieldContext is the data contract that the expression engine exposes
    // to its callers: at eval time the caller must supply (x, y, z, T, t).
    struct FieldContext {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double T = 0.0;
        double t = 0.0;
    };
    using FieldEvaluator = std::function<double(const double* args, int nargs, const FieldContext& ctx)>;

    struct SymbolTable {
        std::unordered_map<std::string, double> variables;
        std::unordered_map<std::string, FieldEvaluator> natives;
    };

    // Internal: Thread-Local Storage wrapper for muparser compiled instances
    struct MuCompiledTLS;

    // Precompiled expression (Copyable, thread-safe eval via TLS)
    class CompiledExpression {
    public:
        CompiledExpression();
        ~CompiledExpression();

        // 允许复制和移动，使其在 vector 等容器中成为廉价的句柄
        CompiledExpression(const CompiledExpression& other) = default;
        CompiledExpression& operator=(const CompiledExpression& other) = default;
        CompiledExpression(CompiledExpression&& other) noexcept = default;
        CompiledExpression& operator=(CompiledExpression&& other) noexcept = default;

        double eval(const FieldContext& ctx) const;

        bool is_constant() const { return is_const_; }
        double constant_value() const { return const_val_; }

        static CompiledExpression make_constant(double value);
        static CompiledExpression make_evaluator(const std::string& formula, const SymbolTable& symbols);

    private:
        bool is_const_ = false;
        double const_val_ = 0.0;
        std::shared_ptr<MuCompiledTLS> tls_impl_;
    };

    // Parse a field expression string against the supplied SymbolTable.
    CompiledExpression parse(const std::string& formula, const SymbolTable& symbols);

    // Evaluate a geometry expression against the supplied SymbolTable's variables.
    double eval_geometry(const std::string& formula, const SymbolTable& symbols);

} // namespace mhs::core
