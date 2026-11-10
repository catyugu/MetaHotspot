// muparser-backed expression API
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace mhs::expression {

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

    // Registered under a formula-level name. A callback that is not annotated
    // may read any context field, so it counts as temperature dependent.
    struct NativeFunction {
        FieldEvaluator evaluate;
        bool reads_temperature = true;
    };

    struct SymbolTable {
        std::unordered_map<std::string, NativeFunction> natives;
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

        bool depends_on_temperature() const { return temperature_dependent_; }

        bool is_constant() const { return std::holds_alternative<double>(value_); }
        double constant_value() const { return std::get<double>(value_); }

        static CompiledExpression make_constant(double value);
        static CompiledExpression make_evaluator(const std::string& formula, const SymbolTable& symbols);

    private:
        bool temperature_dependent_ = false;
        std::variant<double, std::shared_ptr<MuCompiledTLS>> value_ {0.0};
    };

    // Parse a field expression string against the supplied SymbolTable.
    CompiledExpression parse(const std::string& formula, const SymbolTable& symbols);


} // namespace mhs::expression
