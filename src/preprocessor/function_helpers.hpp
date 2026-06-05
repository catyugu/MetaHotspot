#pragma once

#include "common/io_model.hpp"

#include <string>
#include <unordered_map>

namespace mhs::preprocessor {

    // 5 类单变元函数 → FieldEvaluator 闭包。所有闭包都从 ctx.t 槽取自变量；
    // 引用处的字面 x 在 preprocessor 字面替换阶段被改写为 t 或 T。
    // 表达式函数：内层 parse 自变量名仍为 x，闭包内用 ctx.t 喂入。
    FieldEvaluator make_expression_evaluator(const std::string& inner_expr);
    FieldEvaluator make_double_exp_evaluator(double a, double alpha, double beta);
    FieldEvaluator make_gauss_evaluator(double a, double tau, double x0);
    FieldEvaluator make_sine_evaluator(double a, double omega, double phi);
    FieldEvaluator make_piecewise_evaluator(std::vector<PieceWiseFunction::Point> pts);

    // 字面替换：把字符串中的"孤立 x"（前后都不是字母或下划线）替换为 argname。
    // 同一遍扫描里同时校验所有 name(...) 引用的函数名都在 fns 中已注册；
    // 未注册 → panic。
    std::string substitute_function_args(
        const std::string& expr_str, const std::string& argname, const std::unordered_map<std::string, Function>& fns);

    // 把 IOStructure 的所有 Function 注册为 expr 全局 native。
    void register_all_functions(const std::unordered_map<std::string, Function>& fns);

} // namespace mhs::preprocessor
