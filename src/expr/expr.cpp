#include "expr.hpp"

namespace mhs {

    FieldEvaluator ExprEngine::compile(const std::string& formula)
    {
        (void)formula;
        return [](const FieldContext&) { return 0.0; };
    }

    double ExprEngine::evaluate(const FieldContext& ctx)
    {
        (void)ctx;
        return 0.0;
    }

    void ExprEngine::registerNative(const std::string& name, FieldEvaluator func)
    {
        natives_[name] = std::move(func);
    }

    double evalGeometryExpr(const std::string& formula, const std::unordered_map<std::string, double>& vars)
    {
        (void)formula;
        (void)vars;
        return 0.0;
    }

} // namespace mhs
