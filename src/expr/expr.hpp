#pragma once

#include "model/types.hpp"
#include <string>

namespace mhs::expr {

    // Precompiled expression (value type, stateless, thread-safe eval)
    class CompiledExpression {
    public:
        CompiledExpression() = default;
        CompiledExpression(const CompiledExpression&) = default;
        CompiledExpression(CompiledExpression&&) = default;
        CompiledExpression& operator=(const CompiledExpression&) = default;
        CompiledExpression& operator=(CompiledExpression&&) = default;
        ~CompiledExpression() = default;

        double eval(const FieldContext& ctx) const
        {
            return is_const_ ? const_val_ : (eval_ ? eval_(ctx) : 0.0);
        }

        bool is_constant() const { return is_const_; }
        double constant_value() const { return const_val_; }

        static CompiledExpression make_constant(double value)
        {
            return CompiledExpression(nullptr, true, value);
        }

        static CompiledExpression make_evaluator(FieldEvaluator eval)
        {
            return CompiledExpression(std::move(eval), false, 0.0);
        }

    private:
        FieldEvaluator eval_;
        bool is_const_ = false;
        double const_val_ = 0.0;

        CompiledExpression(FieldEvaluator eval, bool is_const, double const_val)
            : eval_(std::move(eval)), is_const_(is_const), const_val_(const_val)
        {
        }
    };

    // Thread-safe registry operations (mutex-protected)
    // These are called by ModelBuilder during preprocessing

    // Register a geometry variable (used by eval_geometry)
    void set_variable(const std::string& name, double value);

    // Register a native C++ function
    void register_native(const std::string& name, FieldEvaluator func);

    // Register a user-defined expression function
    void register_function(const std::string& name, const std::string& expression);

    // Clear all registered functions (for testing)
    void clear_registry();

    // Parse a field expression string (thread-safe during compilation)
    CompiledExpression parse(const std::string& formula);

    // Evaluate a geometry expression (no context needed, uses registered variables)
    double eval_geometry(const std::string& formula);

} // namespace mhs::expr