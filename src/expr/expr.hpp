#pragma once

#include "common/types.hpp"
#include <memory>
#include <string>

namespace mhs::expr {

    // Internal: compiled ExprTK expression (defined in expr.cpp, pimpl)
    class ExprTKCompiled;

    // Precompiled expression (move-only, thread-safe eval)
    // Each instance owns its own ExprTKCompiled — no shared mutable state
    class CompiledExpression {
    public:
        CompiledExpression();
        ~CompiledExpression();

        CompiledExpression(CompiledExpression&& other) noexcept;
        CompiledExpression& operator=(CompiledExpression&& other) noexcept;

        // Non-copyable: each instance owns a unique ExprTKCompiled
        CompiledExpression(const CompiledExpression&) = delete;
        CompiledExpression& operator=(const CompiledExpression&) = delete;

        double eval(const FieldContext& ctx) const;

        bool is_constant() const { return is_const_; }
        double constant_value() const { return const_val_; }

        static CompiledExpression make_constant(double value);
        static CompiledExpression make_evaluator(std::unique_ptr<ExprTKCompiled> impl);

    private:
        bool is_const_ = false;
        double const_val_ = 0.0;
        std::unique_ptr<ExprTKCompiled> impl_;
    };

    // Thread-safe registry operations (mutex-protected)
    void set_variable(const std::string& name, double value);
    void register_native(const std::string& name, FieldEvaluator func);
    void register_function(const std::string& name, const std::string& expression);
    FieldEvaluator get_native(const std::string& name);
    void clear_registry();

    // Parse a field expression string (each call creates a fresh ExprTK instance)
    CompiledExpression parse(const std::string& formula);

    // Evaluate a geometry expression (uses registered variables)
    double eval_geometry(const std::string& formula);

} // namespace mhs::expr