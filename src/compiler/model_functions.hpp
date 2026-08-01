#pragma once

#include "common/model_definition.hpp"
#include "numerics/expression/expr.hpp"

#include <string>

namespace mhs::sim {

    // 字面替换：把字符串中的裸 x 替换为 argname。
    // 更完整的语法校验由后续 mhs::core::parse 负责。
    std::string substitute_function_args(const std::string& expr_str, const std::string& argname);

    // 把 mhs::model::ModelDefinition 的所有 NamedFunction 注册为 SymbolTable 的 natives。
    // 不写任何全局状态；调用方持有 SymbolTable，函数闭包按值存于其中。
    void register_all_functions(
        mhs::core::SymbolTable& symbols, const std::vector<mhs::model::NamedFunction>& functions);

} // namespace mhs::sim
