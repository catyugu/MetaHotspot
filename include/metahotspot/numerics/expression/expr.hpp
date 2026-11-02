// muparser-backed expression API
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

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

    enum FieldDependency : std::uint8_t {
        DependsX = 1,
        DependsY = 2,
        DependsZ = 4,
        DependsTemperature = 8,
        DependsTime = 16,
        DependsAll = 31
    };

    struct SymbolTable {
        std::unordered_map<std::string, double> variables;
        std::unordered_map<std::string, FieldEvaluator> natives;
        // Unannotated native callbacks may read any context field.
        std::unordered_map<std::string, std::uint8_t> native_dependencies;
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

        std::uint8_t dependencies() const { return dependencies_; }
        bool depends_on_temperature() const { return (dependencies_ & DependsTemperature) != 0; }

        bool is_constant() const { return std::holds_alternative<double>(value_); }
        double constant_value() const { return std::get<double>(value_); }

        static CompiledExpression make_constant(double value);
        static CompiledExpression make_evaluator(const std::string& formula, const SymbolTable& symbols);

    private:
        std::uint8_t dependencies_ = 0;
        std::variant<double, std::shared_ptr<MuCompiledTLS>> value_ {0.0};
    };

    // Parse a field expression string against the supplied SymbolTable.
    CompiledExpression parse(const std::string& formula, const SymbolTable& symbols);

    // Evaluate a geometry expression against the supplied SymbolTable's variables.
    double eval_geometry(const std::string& formula, const SymbolTable& symbols);

} // namespace mhs::core
