#pragma once

#include "expr/expr.hpp"
#include "model/model_definition.hpp"

#include <string>
#include <vector>

namespace mhs::sim {

    // 闭包构造器。Expression 类型需要传入 SymbolTable 以便内层表达式能正确解析。
    mhs::core::FieldEvaluator make_expression_evaluator(
        const std::string& inner_expr, const mhs::core::SymbolTable& symbols);
    mhs::core::FieldEvaluator make_double_exp_evaluator(double a, double alpha, double beta);
    mhs::core::FieldEvaluator make_gauss_evaluator(double a, double tau, double x0);
    mhs::core::FieldEvaluator make_sine_evaluator(double a, double omega, double phi);
    mhs::core::FieldEvaluator make_piecewise_evaluator(std::vector<mhs::model::PiecewiseFunctionSpec::Point> pts);

    // 字面替换：把字符串中的"孤立 x"（前后都不是字母或下划线）替换为 argname。
    // 同一遍扫描里同时校验所有 name(...) 引用的函数名都在 fns 中已注册；
    // 未注册 → assert。
    std::string substitute_function_args(const std::string& expr_str, const std::string& argname,
        const std::vector<mhs::model::NamedFunction>& functions);

    // 把 mhs::model::ModelDefinition 的所有 NamedFunction 注册为 SymbolTable 的 natives。
    // 不写任何全局状态；调用方持有 SymbolTable，函数闭包按值存于其中。
    void register_all_functions(mhs::core::SymbolTable& symbols,
        const std::vector<mhs::model::NamedFunction>& functions);

} // namespace mhs::sim
