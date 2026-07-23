#include "solver/operator_assembly.hpp"

#include <gtest/gtest.h>

namespace {

    TEST(OperatorAssemblyTest, MergesIndependentContributions)
    {
        mhs::sim::OperatorContribution first;
        first.stiffness.emplace_back(0, 0, 2.0);
        first.capacity.emplace_back(0, 0, 3.0);
        first.source.push_back({0, 5.0});

        mhs::sim::OperatorContribution second;
        second.stiffness.emplace_back(0, 0, 7.0);
        second.stiffness.emplace_back(1, 1, 11.0);
        second.capacity.emplace_back(1, 1, 13.0);
        second.source.push_back({0, 17.0});
        second.source.push_back({1, 19.0});

        mhs::sim::OperatorAccumulator accumulator(2);
        accumulator.add(std::move(first));
        accumulator.add(std::move(second));
        auto result = std::move(accumulator).finish();

        EXPECT_DOUBLE_EQ(result.K.coeff(0, 0), 9.0);
        EXPECT_DOUBLE_EQ(result.K.coeff(1, 1), 11.0);
        EXPECT_DOUBLE_EQ(result.C.coeff(0, 0), 3.0);
        EXPECT_DOUBLE_EQ(result.C.coeff(1, 1), 13.0);
        EXPECT_DOUBLE_EQ(result.f(0), 22.0);
        EXPECT_DOUBLE_EQ(result.f(1), 19.0);
    }

} // namespace
