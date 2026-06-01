#include "expr/expr.hpp"
#include <gtest/gtest.h>

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double E = 2.71828182845904523536;

    TEST(CompiledExpression, DefaultConstruct)
    {
        mhs::expr::CompiledExpression expr;
        EXPECT_TRUE(expr.is_constant());
        EXPECT_EQ(expr.constant_value(), 0.0);
        EXPECT_EQ(expr.eval({}), 0.0);
    }

    TEST(CompiledExpression, MakeConstant)
    {
        auto expr = mhs::expr::CompiledExpression::make_constant(42.0);
        EXPECT_TRUE(expr.is_constant());
        EXPECT_EQ(expr.constant_value(), 42.0);
        EXPECT_EQ(expr.eval({}), 42.0);
    }

    TEST(CompiledExpression, MakeConstant_Negative)
    {
        auto expr = mhs::expr::CompiledExpression::make_constant(-3.14);
        EXPECT_TRUE(expr.is_constant());
        EXPECT_EQ(expr.constant_value(), -3.14);
        EXPECT_EQ(expr.eval({}), -3.14);
    }

    TEST(CompiledExpression, MakeEvaluator)
    {
        mhs::expr::clear_registry();
        auto expr = mhs::expr::parse("x + y");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {1.0, 2.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 3.0);
    }

    TEST(CompiledExpression, EvaluatorIgnoresContextForConstant)
    {
        auto expr = mhs::expr::CompiledExpression::make_constant(99.0);
        mhs::FieldContext ctx {1.0, 2.0, 3.0, 300.0, 10.0};
        EXPECT_EQ(expr.eval(ctx), 99.0);
    }

    TEST(CompiledExpression, Movable)
    {
        auto orig = mhs::expr::CompiledExpression::make_constant(7.0);
        auto dest = std::move(orig);
        EXPECT_EQ(dest.constant_value(), 7.0);
    }

} // namespace

namespace {

    TEST(EvalGeometry, SingleVariable)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("w", 10.0);

        EXPECT_EQ(mhs::expr::eval_geometry("w"), 10.0);
    }

    TEST(EvalGeometry, MultipleVariables)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("w", 5.0);
        mhs::expr::set_variable("h", 2.0);

        EXPECT_EQ(mhs::expr::eval_geometry("w"), 5.0);
        EXPECT_EQ(mhs::expr::eval_geometry("h"), 2.0);
    }

    TEST(EvalGeometry, ArithmeticAddition)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("w", 5.0);
        mhs::expr::set_variable("h", 3.0);

        EXPECT_EQ(mhs::expr::eval_geometry("w+h"), 8.0);
    }

    TEST(EvalGeometry, ArithmeticSubtraction)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("w", 10.0);
        mhs::expr::set_variable("h", 3.0);

        EXPECT_EQ(mhs::expr::eval_geometry("w-h"), 7.0);
    }

    TEST(EvalGeometry, ArithmeticMultiplication)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("w", 5.0);
        mhs::expr::set_variable("h", 3.0);

        EXPECT_EQ(mhs::expr::eval_geometry("w*h"), 15.0);
    }

    TEST(EvalGeometry, ArithmeticDivision)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("w", 10.0);
        mhs::expr::set_variable("h", 2.0);

        EXPECT_EQ(mhs::expr::eval_geometry("w/h"), 5.0);
    }

    TEST(EvalGeometry, ComplexExpression)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("w", 10.0);
        mhs::expr::set_variable("h", 2.0);

        // (w + h) * 2
        EXPECT_EQ(mhs::expr::eval_geometry("(w+h)*2"), 24.0);
    }

    TEST(EvalGeometry, NestedParentheses)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("a", 2.0);
        mhs::expr::set_variable("b", 3.0);
        mhs::expr::set_variable("c", 4.0);

        // (a + b) * c
        EXPECT_EQ(mhs::expr::eval_geometry("(a+b)*c"), 20.0);
    }

    TEST(EvalGeometry, OperatorPrecedence)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("a", 10.0);
        mhs::expr::set_variable("b", 3.0);
        mhs::expr::set_variable("c", 2.0);

        // a + b * c = 10 + 6 = 16
        EXPECT_EQ(mhs::expr::eval_geometry("a+b*c"), 16.0);
    }

    TEST(EvalGeometry, UndefinedVariable)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("defined", 1.0);

        // undefined variable should return 0.0
        EXPECT_EQ(mhs::expr::eval_geometry("undefined"), 0.0);
    }

} // namespace

namespace {

    TEST(Parse, SimpleConstant)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("42");
        EXPECT_TRUE(expr.is_constant());
        EXPECT_EQ(expr.constant_value(), 42.0);
    }

    TEST(Parse, NegativeConstant)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("-17.5");
        EXPECT_TRUE(expr.is_constant());
        EXPECT_EQ(expr.constant_value(), -17.5);
    }

    TEST(Parse, FloatingPointConstant)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("3.14159");
        EXPECT_TRUE(expr.is_constant());
        EXPECT_NEAR(expr.constant_value(), 3.14159, 1e-10);
    }

    TEST(Parse, SimpleArithmetic)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("x + 1");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {5.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 6.0);
    }

    TEST(Parse, ArithmeticWithY)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("x + y");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {3.0, 4.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 7.0);
    }

    TEST(Parse, ArithmeticWithZ)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("x + y + z");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {1.0, 2.0, 3.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 6.0);
    }

    TEST(Parse, Multiplication)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("x * y");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {3.0, 4.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 12.0);
    }

    TEST(Parse, Division)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("x / y");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {10.0, 2.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 5.0);
    }

    TEST(Parse, ComplexExpression)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("(x + y) * z");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {1.0, 2.0, 3.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 9.0);
    }

    TEST(Parse, OperatorPrecedence)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("2 + 3 * 4");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 14.0); // 2 + (3*4) = 14
    }

} // namespace

namespace {

    TEST(Parse, SinFunction)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("sin(x)");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_NEAR(expr.eval(ctx), 0.0, 1e-10);

        ctx.x = PI / 2;
        EXPECT_NEAR(expr.eval(ctx), 1.0, 1e-10);
    }

    TEST(Parse, CosFunction)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("cos(x)");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_NEAR(expr.eval(ctx), 1.0, 1e-10);

        ctx.x = PI;
        EXPECT_NEAR(expr.eval(ctx), -1.0, 1e-10);
    }

    TEST(Parse, ExpFunction)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("exp(x)");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_NEAR(expr.eval(ctx), 1.0, 1e-10);

        ctx.x = 1.0;
        EXPECT_NEAR(expr.eval(ctx), E, 1e-10);
    }

    TEST(Parse, LogFunction)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("log(x)");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {1.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_NEAR(expr.eval(ctx), 0.0, 1e-10);

        ctx.x = E;
        EXPECT_NEAR(expr.eval(ctx), 1.0, 1e-10);
    }

    TEST(Parse, SqrtFunction)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("sqrt(x)");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {4.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_NEAR(expr.eval(ctx), 2.0, 1e-10);

        ctx.x = 9.0;
        EXPECT_NEAR(expr.eval(ctx), 3.0, 1e-10);
    }

    TEST(Parse, AbsFunction)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("abs(x)");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {-5.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 5.0);

        ctx.x = 3.0;
        EXPECT_EQ(expr.eval(ctx), 3.0);
    }

    TEST(Parse, FloorCeil)
    {
        mhs::expr::clear_registry();

        auto floor_expr = mhs::expr::parse("floor(x)");
        auto ceil_expr = mhs::expr::parse("ceil(x)");

        mhs::FieldContext ctx {3.7, 0.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(floor_expr.eval(ctx), 3.0);
        EXPECT_EQ(ceil_expr.eval(ctx), 4.0);
    }

} // namespace

namespace {

    TEST(Parse, WithTemperatureT)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("T + 100");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {0.0, 0.0, 0.0, 300.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 400.0);
    }

    TEST(Parse, WithTimeT)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("t * 2");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 5.0};
        auto result = expr.eval(ctx);
        EXPECT_EQ(result, 10.0);
    }

    // Test using our wrapper (clears cache first)
    TEST(Parse, WrapperWithCache)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("x + 1");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {5.0, 0.0, 0.0, 0.0, 0.0};
        auto result = expr.eval(ctx);
        EXPECT_EQ(result, 6.0);
    }

    // Test that t variable works correctly
    TEST(Parse, TVariable)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("T * 2");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {0.0, 0.0, 0.0, 10.0, 0.0};
        auto result = expr.eval(ctx);
        EXPECT_EQ(result, 20.0);
    }

    // Test lowercase 't' specifically
    TEST(Parse, LowerCaseTWrapped)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("t");

        mhs::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 7.0};
        auto result = expr.eval(ctx);
        EXPECT_EQ(result, 7.0);
    }

    TEST(Parse, CombinedContext)
    {
        mhs::expr::clear_registry();

        auto expr = mhs::expr::parse("x + y + z + T + t");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {1.0, 2.0, 3.0, 10.0, 100.0};
        EXPECT_EQ(expr.eval(ctx), 116.0);
    }

    TEST(Parse, NonlinearTemperatureDependence)
    {
        mhs::expr::clear_registry();

        // k(T) = k0 * (1 + alpha * T)
        auto expr = mhs::expr::parse("10 * (1 + 0.001 * T)");
        EXPECT_FALSE(expr.is_constant());

        mhs::FieldContext ctx {0.0, 0.0, 0.0, 300.0, 0.0};
        EXPECT_NEAR(expr.eval(ctx), 13.0, 1e-10);
    }

} // namespace

namespace {

    TEST(NativeFunction, RegisterAndUse)
    {
        mhs::expr::clear_registry();

        // Register a piecewise function
        mhs::expr::register_native("piecewise", [](const mhs::FieldContext& ctx) {
            if (ctx.x < 1.0)
                return 0.0;
            if (ctx.x < 2.0)
                return 1.0;
            return 2.0;
        });

        auto native = mhs::expr::get_native("piecewise");
        EXPECT_TRUE(native != nullptr);

        mhs::FieldContext ctx {0.5, 0.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(native(ctx), 0.0);

        ctx.x = 1.5;
        EXPECT_EQ(native(ctx), 1.0);
    }

    TEST(NativeFunction, GetUnregistered)
    {
        mhs::expr::clear_registry();

        auto native = mhs::expr::get_native("nonexistent");
        EXPECT_TRUE(native == nullptr);
    }

} // namespace

namespace {

    TEST(ClearRegistry, ClearsVariables)
    {
        mhs::expr::clear_registry();
        mhs::expr::set_variable("x", 1.0);
        EXPECT_EQ(mhs::expr::eval_geometry("x"), 1.0);

        mhs::expr::clear_registry();
        EXPECT_EQ(mhs::expr::eval_geometry("x"), 0.0);
    }

    TEST(ClearRegistry, ClearsFunctions)
    {
        mhs::expr::clear_registry();
        mhs::expr::register_native("f", [](const mhs::FieldContext&) { return 42.0; });
        EXPECT_TRUE(mhs::expr::get_native("f") != nullptr);

        mhs::expr::clear_registry();
        EXPECT_TRUE(mhs::expr::get_native("f") == nullptr);
    }

} // namespace

namespace {

    TEST(ParserCaching, SameExpressionReturnsCached)
    {
        mhs::expr::clear_registry();

        auto expr1 = mhs::expr::parse("x + 1");
        auto expr2 = mhs::expr::parse("x + 1");

        // Both should evaluate the same
        mhs::FieldContext ctx {5.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr1.eval(ctx), 6.0);
        EXPECT_EQ(expr2.eval(ctx), 6.0);
    }

    TEST(ParserCaching, DifferentExpressionsAreSeparate)
    {
        mhs::expr::clear_registry();

        auto expr1 = mhs::expr::parse("x + 1");
        auto expr2 = mhs::expr::parse("x + 2");

        mhs::FieldContext ctx {5.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr1.eval(ctx), 6.0);
        EXPECT_EQ(expr2.eval(ctx), 7.0);
    }

} // namespace