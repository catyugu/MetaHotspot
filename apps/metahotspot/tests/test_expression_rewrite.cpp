#include "pta_definition.hpp"
#include <gtest/gtest.h>
namespace {
    // ---- 字面替换 --------------------------------------------------------

    TEST(Substitute, BasicCallReplacesXForT)
    {
        auto out = mhs::app::pta::rewrite_formal_argument("test_gaussian(x)", "T");
        EXPECT_EQ(out, "test_gaussian(T)");
    }

    TEST(Substitute, MultipleXReplaced)
    {
        auto out = mhs::app::pta::rewrite_formal_argument("test_gaussian(x)/(x*0.01+1)", "t");
        EXPECT_EQ(out, "test_gaussian(t)/(t*0.01+1)");
    }

    TEST(Substitute, XFollowedByUnderscoreNotReplaced)
    {
        auto out = mhs::app::pta::rewrite_formal_argument("2*x + x_next", "t");
        EXPECT_EQ(out, "2*t + x_next");
    }

    TEST(Substitute, XBetweenLettersNotReplaced)
    {
        auto out = mhs::app::pta::rewrite_formal_argument("xx + axb", "t");
        EXPECT_EQ(out, "xx + axb");
    }

    TEST(Substitute, NoFunctionsNoChange)
    {
        auto out = mhs::app::pta::rewrite_formal_argument("x+1", "T");
        EXPECT_EQ(out, "T+1");
    }

    TEST(Substitute, BareFunctionCallNotRewritten)
    {
        auto out = mhs::app::pta::rewrite_formal_argument("foo(x)", "T");
        EXPECT_EQ(out, "foo(T)");
    }

    TEST(Substitute, NestedWithOtherVariables)
    {
        auto out = mhs::app::pta::rewrite_formal_argument("y + z + T + t + pi", "T");
        EXPECT_EQ(out, "y + z + T + t + pi");
    }

}
