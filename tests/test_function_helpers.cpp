#include "compiler/model_functions.hpp"
#include "mhs/expression.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using namespace mhs::core;
using namespace mhs::sim;

namespace {

    // ---- 通过 register_all_functions 端到端测试函数注册和求值 ----

    TEST(EndToEnd, ParseTakesRegisteredNative)
    {
        mhs::core::SymbolTable sym;
        std::vector<mhs::model::NamedFunction> fns;
        fns.push_back({"test_gaussian", mhs::model::GaussFunctionSpec {5.0, 10.0, 20.0}});
        register_all_functions(sym, fns);

        // 字面替换：用户写 test_gaussian(x)，preprocessor 在材料槽里替换为 test_gaussian(T)
        auto out = substitute_function_args("test_gaussian(x)", "T");
        EXPECT_EQ(out, "test_gaussian(T)");

        auto compiled = mhs::core::parse(out, sym);
        // Native 接收 muparser 绑定的参数向量 args 与当前 TLS 物理 ctx；
        // 现有 natives 读 ctx.t，所以测试时把值放在 ctx.T 上。
        FieldContext ctx {0, 0, 0, 20.0, 0.0};
        EXPECT_NEAR(compiled.eval(ctx), 5.0, 1e-9);
    }

    TEST(EndToEnd, NativeReadsTheBoundSymbol)
    {
        mhs::core::SymbolTable sym;
        std::vector<mhs::model::NamedFunction> fns;
        fns.push_back({"test_gaussian", mhs::model::GaussFunctionSpec {5.0, 10.0, 20.0}});
        register_all_functions(sym, fns);

        // 不做字面替换（"test_gaussian(x)" 直接编译），muparser 把 x 槽绑定。
        auto compiled = mhs::core::parse("test_gaussian(x)", sym);
        FieldContext ctx {20.0, 0, 0, 0, 0};
        EXPECT_NEAR(compiled.eval(ctx), 5.0, 1e-9);
    }

    TEST(EndToEnd, GaussEvaluator)
    {
        mhs::core::SymbolTable sym;
        std::vector<mhs::model::NamedFunction> fns;
        fns.push_back({"mygauss", mhs::model::GaussFunctionSpec {1.0, 1.0, 0.0}});
        register_all_functions(sym, fns);

        auto compiled = mhs::core::parse("mygauss(t)", sym);
        FieldContext ctx {0, 0, 0, 0, 0.0};
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 1.0);
        ctx.t = 1.0;
        EXPECT_NEAR(compiled.eval(ctx), std::exp(-1.0), 1e-12);
    }

    TEST(EndToEnd, SineEvaluator)
    {
        mhs::core::SymbolTable sym;
        std::vector<mhs::model::NamedFunction> fns;
        fns.push_back({"mysine", mhs::model::SineFunctionSpec {5.0, 200.0, 1.57}});
        register_all_functions(sym, fns);

        auto compiled = mhs::core::parse("mysine(t)", sym);
        FieldContext ctx {0, 0, 0, 0, 0.0};
        EXPECT_NEAR(compiled.eval(ctx), 5.0 * std::sin(1.57), 1e-9);
    }

    TEST(EndToEnd, DoubleExpEvaluator)
    {
        mhs::core::SymbolTable sym;
        std::vector<mhs::model::NamedFunction> fns;
        fns.push_back({"mydexp", mhs::model::DoubleExponentialFunctionSpec {1.0, 0.5, 0.1}});
        register_all_functions(sym, fns);

        auto compiled = mhs::core::parse("mydexp(t)", sym);
        FieldContext ctx {0, 0, 0, 0, 0.0};
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 0.0);
    }

    TEST(EndToEnd, PiecewiseEvaluator)
    {
        mhs::core::SymbolTable sym;
        std::vector<mhs::model::NamedFunction> fns;
        mhs::model::PiecewiseFunctionSpec pw;
        pw.points = {{0, -1}, {1, 2}, {5, 3}};
        fns.push_back({"mypw", std::move(pw)});
        register_all_functions(sym, fns);

        auto compiled = mhs::core::parse("mypw(t)", sym);
        FieldContext ctx {0, 0, 0, 0, -1.0};
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), -1.0);
        ctx.t = 10.0;
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 3.0);
        ctx.t = 3.0;
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 2.5);
    }

    TEST(EndToEnd, PeriodicPiecewiseConstant)
    {
        mhs::core::SymbolTable sym;
        std::vector<mhs::model::NamedFunction> fns;
        mhs::model::PeriodicPiecewiseConstantFunctionSpec ppc;
        ppc.period = 10.0;
        ppc.values = {1.0, 2.0, 3.0};
        fns.push_back({"myppc", std::move(ppc)});
        register_all_functions(sym, fns);

        auto compiled = mhs::core::parse("myppc(t)", sym);
        FieldContext ctx {};
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 1.0);
        ctx.t = 5.0;
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 1.0);
        ctx.t = 10.0;
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 2.0);
        ctx.t = 20.0;
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 3.0);
        ctx.t = 30.0;
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 1.0);
        ctx.t = -5.0;
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 3.0);
    }

    TEST(PeriodicPiecewiseConstant, Empty)
    {
        mhs::core::SymbolTable sym;
        std::vector<mhs::model::NamedFunction> fns;
        mhs::model::PeriodicPiecewiseConstantFunctionSpec ppc;
        ppc.period = 10.0;
        fns.push_back({"emptyppc", std::move(ppc)});
        register_all_functions(sym, fns);

        auto compiled = mhs::core::parse("emptyppc(t)", sym);
        FieldContext ctx {};
        EXPECT_DOUBLE_EQ(compiled.eval(ctx), 0.0);
    }

    TEST(PeriodicPiecewiseConstant, SingleValue)
    {
        mhs::core::SymbolTable sym;
        std::vector<mhs::model::NamedFunction> fns;
        mhs::model::PeriodicPiecewiseConstantFunctionSpec ppc;
        ppc.period = 10.0;
        ppc.values = {42.0};
        fns.push_back({"singleppc", std::move(ppc)});
        register_all_functions(sym, fns);

        auto compiled = mhs::core::parse("singleppc(t)", sym);
        FieldContext ctx {};
        for (double t = -20.0; t <= 20.0; t += 5.0) {
            ctx.t = t;
            EXPECT_DOUBLE_EQ(compiled.eval(ctx), 42.0);
        }
    }

    // ---- 字面替换 --------------------------------------------------------

    TEST(Substitute, BasicCallReplacesXForT)
    {
        auto out = substitute_function_args("test_gaussian(x)", "T");
        EXPECT_EQ(out, "test_gaussian(T)");
    }

    TEST(Substitute, MultipleXReplaced)
    {
        auto out = substitute_function_args("test_gaussian(x)/(x*0.01+1)", "t");
        EXPECT_EQ(out, "test_gaussian(t)/(t*0.01+1)");
    }

    TEST(Substitute, XFollowedByUnderscoreNotReplaced)
    {
        auto out = substitute_function_args("2*x + x_next", "t");
        EXPECT_EQ(out, "2*t + x_next");
    }

    TEST(Substitute, XBetweenLettersNotReplaced)
    {
        auto out = substitute_function_args("xx + axb", "t");
        EXPECT_EQ(out, "xx + axb");
    }

    TEST(Substitute, NoFunctionsNoChange)
    {
        auto out = substitute_function_args("x+1", "T");
        EXPECT_EQ(out, "T+1");
    }

    TEST(Substitute, BareFunctionCallNotRewritten)
    {
        auto out = substitute_function_args("foo(x)", "T");
        EXPECT_EQ(out, "foo(T)");
    }

    TEST(Substitute, NestedWithOtherVariables)
    {
        auto out = substitute_function_args("y + z + T + t + pi", "T");
        EXPECT_EQ(out, "y + z + T + t + pi");
    }

} // namespace
