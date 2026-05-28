#include "expr.hpp"
#include <mutex>
#include <unordered_map>
#include <regex>

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

    void clear_registry()
    {
        std::lock_guard<std::mutex> lock(registry::mutex());
        registry::variables().clear();
        registry::native_functions().clear();
        registry::user_functions().clear();
    }

    // Parse field expression string
    // TODO: Replace with exprtk integration
    CompiledExpression parse(const std::string& formula)
    {
        std::lock_guard<std::mutex> lock(registry::mutex());

        // Simple constant detection
        std::regex const_regex(R"(^\s*-?\d+\.?\d*\s*$)");
        if (std::regex_match(formula, const_regex)) {
            double value = std::stod(formula);
            return CompiledExpression::make_constant(value);
        }

        // Return a stub evaluator for now
        return CompiledExpression::make_evaluator([](const FieldContext&) { return 0.0; });
    }

    // Evaluate geometry expression
    double eval_geometry(const std::string& formula)
    {
        std::lock_guard<std::mutex> lock(registry::mutex());

        const auto& vars = registry::variables();
        if (vars.find(formula) != vars.end()) {
            return vars.at(formula);
        }

        // Try to parse simple arithmetic with substituted variables
        std::string expr = formula;
        for (const auto& [name, val] : vars) {
            size_t pos;
            while ((pos = expr.find(name)) != std::string::npos) {
                expr.replace(pos, name.length(), std::to_string(val));
            }
        }

        try {
            return std::stod(expr);
        } catch (...) {
            return 0.0;
        }
    }

} // namespace mhs::expr