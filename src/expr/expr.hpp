#pragma once

#include "model/types.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace mhs {

    class ExprEngine {
    public:
        ExprEngine() = default;
        ~ExprEngine() = default;

        FieldEvaluator compile(const std::string& formula);

        double evaluate(const FieldContext& ctx);

        void registerNative(const std::string& name, FieldEvaluator func);

    private:
        std::unordered_map<std::string, FieldEvaluator> natives_;
        std::unique_ptr<FieldEvaluator> current_;
    };

    double evalGeometryExpr(const std::string& formula, const std::unordered_map<std::string, double>& vars);

} // namespace mhs
