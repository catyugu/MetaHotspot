#include "expr/expr.hpp"
#include "preprocessor/function_helpers.hpp"

#include "gtest/gtest.h"
#include <gtest/gtest.h>

#include <cmath>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace mhs;
using namespace mhs::expr;
using namespace mhs::preprocessor;

namespace {

    // ---- 5 类闭包的数学正确性 --------------------------------------------

    TEST(FunctionHelpers, ExpressionEvaluator)
    {
        mhs::expr::clear_registry();
        auto ev = make_expression_evaluator("2*x+1");
        // 内层表达式读 ctx.x；调用方负责把自变量放进 x 槽。
        FieldContext ctx {3, 0, 0, 0, 0};
        EXPECT_DOUBLE_EQ(ev(ctx), 7.0);
    }

    TEST(FunctionHelpers, ExpressionEvaluatorBindsXToT)
    {
        mhs::expr::clear_registry();
        auto ev = make_expression_evaluator("x*x");
        FieldContext ctx {5, 0, 0, 0, 0};
        EXPECT_DOUBLE_EQ(ev(ctx), 25.0);
    }

    TEST(FunctionHelpers, GaussEvaluatorAtCenter)
    {
        auto ev = make_gauss_evaluator(1.0, 1.0, 0.0);
        FieldContext ctx {0, 0, 0, 0, 0};
        EXPECT_DOUBLE_EQ(ev(ctx), 1.0);
    }

    TEST(FunctionHelpers, GaussEvaluatorAtOneTau)
    {
        auto ev = make_gauss_evaluator(1.0, 1.0, 0.0);
        FieldContext ctx {0, 0, 0, 0, 1.0};
        EXPECT_NEAR(ev(ctx), std::exp(-1.0), 1e-12);
    }

    TEST(FunctionHelpers, GaussEvaluatorOffset)
    {
        // A=5, tau=10, x0=20, t=20 → A*exp(0) = 5
        auto ev = make_gauss_evaluator(5.0, 10.0, 20.0);
        FieldContext ctx {0, 0, 0, 0, 20.0};
        EXPECT_DOUBLE_EQ(ev(ctx), 5.0);
    }

    TEST(FunctionHelpers, SineEvaluatorAtHalfPi)
    {
        auto ev = make_sine_evaluator(1.0, 1.0, 0.0);
        FieldContext ctx {0, 0, 0, 0, M_PI / 2.0};
        EXPECT_NEAR(ev(ctx), 1.0, 1e-12);
    }

    TEST(FunctionHelpers, SineEvaluatorWithPhase)
    {
        // A=5, omega=200, phi=1.57, t=0 → 5*sin(1.57) ≈ 5
        auto ev = make_sine_evaluator(5.0, 200.0, 1.57);
        FieldContext ctx {0, 0, 0, 0, 0};
        EXPECT_NEAR(ev(ctx), 5.0 * std::sin(1.57), 1e-9);
    }

    TEST(FunctionHelpers, DoubleExpEvaluatorAtZero)
    {
        // A*(exp(alpha*0) - exp(beta*0)) = A*(1-1) = 0
        auto ev = make_double_exp_evaluator(1.0, 0.5, 0.1);
        FieldContext ctx {0, 0, 0, 0, 0};
        EXPECT_DOUBLE_EQ(ev(ctx), 0.0);
    }

    TEST(FunctionHelpers, PiecewiseEvaluatorBelowFirst)
    {
        std::vector<PieceWiseFunction::Point> pts = {{0, -1}, {1, 2}, {5, 3}};
        auto ev = make_piecewise_evaluator(pts);
        FieldContext ctx {0, 0, 0, 0, -1.0};
        EXPECT_DOUBLE_EQ(ev(ctx), -1.0);
    }

    TEST(FunctionHelpers, PiecewiseEvaluatorAboveLast)
    {
        std::vector<PieceWiseFunction::Point> pts = {{0, -1}, {1, 2}, {5, 3}};
        auto ev = make_piecewise_evaluator(pts);
        FieldContext ctx {0, 0, 0, 0, 10.0};
        EXPECT_DOUBLE_EQ(ev(ctx), 3.0);
    }

    TEST(FunctionHelpers, PiecewiseEvaluatorLinearSegment)
    {
        std::vector<PieceWiseFunction::Point> pts = {{0, -1}, {1, 2}, {5, 3}};
        auto ev = make_piecewise_evaluator(pts);
        // 段 [1,2]→[5,3]：x=3 时 t = (3-1)/(5-1) = 0.5，y = 2 + 0.5*(3-2) = 2.5
        FieldContext ctx {0, 0, 0, 0, 3.0};
        EXPECT_DOUBLE_EQ(ev(ctx), 2.5);
    }

    // ---- 字面替换 --------------------------------------------------------

    std::unordered_map<std::string, Function> fns_with_gauss()
    {
        std::unordered_map<std::string, Function> fns;
        Function g;
        g.type = FunctionType::Gauss;
        g.gauss = {5.0, 10.0, 20.0, 0.0, 100.0};
        fns["test_gaussian"] = g;
        return fns;
    }

    TEST(Substitute, BasicCallReplacesX)
    {
        auto fns = fns_with_gauss();
        auto out = substitute_function_args("test_gaussian(x)", "t", fns);
        EXPECT_EQ(out, "test_gaussian(t)");
    }

    TEST(Substitute, BasicCallReplacesXForT)
    {
        auto fns = fns_with_gauss();
        auto out = substitute_function_args("test_gaussian(x)", "T", fns);
        EXPECT_EQ(out, "test_gaussian(T)");
    }

    TEST(Substitute, MultipleXReplaced)
    {
        auto fns = fns_with_gauss();
        // test_gaussian(x)/(x*0.01+1) → test_gaussian(t)/(t*0.01+1)
        auto out = substitute_function_args("test_gaussian(x)/(x*0.01+1)", "t", fns);
        EXPECT_EQ(out, "test_gaussian(t)/(t*0.01+1)");
    }

    TEST(Substitute, BareXAtStringStart)
    {
        auto fns = fns_with_gauss();
        // "x*0.01+1" → "t*0.01+1" (字符串首的 x 算孤立)
        auto out = substitute_function_args("x*0.01+1", "t", fns);
        EXPECT_EQ(out, "t*0.01+1");
    }

    TEST(Substitute, BareXAfterDigit)
    {
        auto fns = fns_with_gauss();
        // "2*x" → "2*t"
        auto out = substitute_function_args("2*x", "t", fns);
        EXPECT_EQ(out, "2*t");
    }

    TEST(Substitute, XFollowedByUnderscoreNotReplaced)
    {
        auto fns = fns_with_gauss();
        // "2*x + x_next" → "2*t + x_next"（第二个 x 后面是 _，不替换）
        auto out = substitute_function_args("2*x + x_next", "t", fns);
        EXPECT_EQ(out, "2*t + x_next");
    }

    TEST(Substitute, XBetweenLettersNotReplaced)
    {
        auto fns = fns_with_gauss();
        // "xx + axb" 中所有 x 都不替换
        auto out = substitute_function_args("xx + axb", "t", fns);
        EXPECT_EQ(out, "xx + axb");
    }

    TEST(Substitute, FunctionNameUnchanged)
    {
        auto fns = fns_with_gauss();
        // 函数名 test_gaussian 不能被改成 test_gaussian_t
        auto out = substitute_function_args("test_gaussian(x)", "t", fns);
        EXPECT_EQ(out, "test_gaussian(t)");
        EXPECT_EQ(out.find("test_gaussian_"), std::string::npos);
    }

    TEST(Substitute, NoFunctionsNoChange)
    {
        // 没有引用任何函数时，孤立 x 仍然替换
        std::unordered_map<std::string, Function> fns;
        auto out = substitute_function_args("x+1", "T", fns);
        EXPECT_EQ(out, "T+1");
    }

    TEST(Substitute, UnknownFunctionPanics)
    {
        std::unordered_map<std::string, Function> fns;
        EXPECT_DEATH(substitute_function_args("foo(x)", "T", fns), "");
    }

    // ---- 注册 native + 端到端 eval --------------------------------------

    TEST(RegisterAll, GaussNativeCompiles)
    {
        mhs::expr::clear_registry();
        auto fns = fns_with_gauss();
        register_all_functions(fns);
        EXPECT_TRUE(mhs::expr::get_native("test_gaussian") != nullptr);
    }

    TEST(EndToEnd, ParseTakesRegisteredNative)
    {
        mhs::expr::clear_registry();
        auto fns = fns_with_gauss();
        register_all_functions(fns);

        // 字面替换：用户写 test_gaussian(x)，preprocessor 在材料槽里替换为 test_gaussian(T)
        auto out = substitute_function_args("test_gaussian(x)", "T", fns);
        EXPECT_EQ(out, "test_gaussian(T)");

        auto compiled = mhs::expr::parse(out);
        // Native 接收 exprtk 绑定的 arg0（这里是 T 槽），广播到所有 ctx 槽；
        // 现有 natives 读 ctx.t，所以测试时把值放在 ctx.T 上。
        FieldContext ctx {0, 0, 0, 20.0, 0.0};
        EXPECT_NEAR(compiled.eval(ctx), 5.0, 1e-9);
    }

    TEST(EndToEnd, NativeReadsTheBoundSymbol)
    {
        // 通用性：future natives 可以读任何 ctx 槽。当前 natives 读 ctx.t，
        // 所以"绑定到 t"也能工作——验证 NativeFn 把 arg0 广播到了 ctx.t。
        mhs::expr::clear_registry();
        auto fns = fns_with_gauss();
        register_all_functions(fns);

        // 不做字面替换（"test_gaussian(x)" 直接编译），exprtk 把 x 槽绑定。
        auto compiled = mhs::expr::parse("test_gaussian(x)");
        FieldContext ctx {20.0, 0, 0, 0, 0};
        EXPECT_NEAR(compiled.eval(ctx), 5.0, 1e-9);
    }

} // namespace
