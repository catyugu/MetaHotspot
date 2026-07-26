#pragma once

#include "model/model_definition.hpp"
#include "numerics/expression/expr.hpp"

#include <string>
#include <vector>

namespace mhs::sim {

    // 字面替换：把字符串中的"孤立 x"（前后都不是字母或下划线）替换为 argname。
    // 同一遍扫描里同时校验所有 name(...) 引用的函数名都在 fns 中已注册；
    // 未注册 → assert。
    std::string substitute_function_args(const std::string& expr_str, const std::string& argname,
        const std::vector<mhs::model::NamedFunction>& functions);

    // 把 mhs::model::ModelDefinition 的所有 NamedFunction 注册为 SymbolTable 的 natives。
    // 不写任何全局状态；调用方持有 SymbolTable，函数闭包按值存于其中。
    void register_all_functions(
        mhs::core::SymbolTable& symbols, const std::vector<mhs::model::NamedFunction>& functions);

} // namespace mhs::sim
