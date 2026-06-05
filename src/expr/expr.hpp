#pragma once

#include <functional>
#include <memory>
#include <string>

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

    using FieldEvaluator = std::function<double(const FieldContext&)>;

    // Internal: Thread-Local Storage wrapper for ExprTK compiled instances
    struct ExprTKCompiledTLS;

    // Precompiled expression (Copyable, thread-safe eval via TLS)
    // Each thread accessing this expression will instantiate its own ExprTK AST on-demand.
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
        static CompiledExpression make_evaluator(const std::string& formula);

    private:
        bool is_const_ = false;
        double const_val_ = 0.0;
        std::shared_ptr<ExprTKCompiledTLS> tls_impl_;
    };

    // Thread-safe registry operations (mutex-protected)
    void set_variable(const std::string& name, double value);
    void register_native(const std::string& name, FieldEvaluator func);
    void register_function(const std::string& name, const std::string& expression);
    FieldEvaluator get_native(const std::string& name);
    void clear_registry();

    // Parse a field expression string
    CompiledExpression parse(const std::string& formula);

    // Evaluate a geometry expression (uses registered variables)
    double eval_geometry(const std::string& formula);

} // namespace mhs::core
