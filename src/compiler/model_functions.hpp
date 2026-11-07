#pragma once

#include "core/vector.hpp"

#include "expression/expression.hpp"
#include "model/model_definition.hpp"

namespace mhs::compiler {

    // 把 mhs::model::ModelDefinition 的所有 NamedFunction 注册为 SymbolTable 的 natives。
    // 不写任何全局状态；调用方持有 SymbolTable，函数闭包按值存于其中。
    void register_all_functions(
        mhs::expression::SymbolTable& symbols, const mhs::core::Vector<mhs::model::NamedFunction>& functions);

} // namespace mhs::compiler
