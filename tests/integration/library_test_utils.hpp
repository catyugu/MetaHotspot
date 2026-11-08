#pragma once
#include "model/model_definition.hpp"
#include <utility>

namespace test_model {
    inline mhs::model::LibraryDefinition& library(mhs::model::ModelDefinition& model)
    { return *model.components.front().library; }
    inline const mhs::model::CompiledLibrary& library(const mhs::model::CompiledModel& model)
    { return *model.components.front().library; }

    template <class T, class U> inline mhs::core::EntityIndex append(mhs::core::Vector<T>& values, U&& value)
    {
        values.emplace_back(std::forward<U>(value));
        return static_cast<mhs::core::EntityIndex>(values.size() - 1);
    }
} // namespace test_model
