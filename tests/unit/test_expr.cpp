#include "expression/expression.hpp"

#include "core/vector.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <thread>
using namespace mhs::expression;
namespace {
    constexpr double PI = 3.14159265358979323846;

    TEST(CompiledExpression, DefaultConstruct)
    {
        mhs::expression::CompiledExpression expr;
        EXPECT_TRUE(expr.is_constant());
        EXPECT_FALSE(expr.depends_on_temperature());
        EXPECT_EQ(expr.constant_value(), 0.0);
        EXPECT_EQ(expr.eval({}), 0.0);
    }

    TEST(CompiledExpression, MakeConstant)
    {
        auto expr = mhs::expression::CompiledExpression::make_constant(42.0);
        EXPECT_TRUE(expr.is_constant());
        EXPECT_FALSE(expr.depends_on_temperature());
        EXPECT_EQ(expr.constant_value(), 42.0);
        EXPECT_EQ(expr.eval({}), 42.0);
    }

    TEST(CompiledExpression, MakeEvaluator)
    {
        SymbolTable sym;
        auto expr = mhs::expression::parse("x + y", sym);
        EXPECT_FALSE(expr.is_constant());
        EXPECT_FALSE(expr.depends_on_temperature());
        EXPECT_FALSE(parse("280+10*t+y", sym).depends_on_temperature());

        mhs::expression::FieldContext ctx {1.0, 2.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 3.0);
        EXPECT_THROW(CompiledExpression::make_evaluator("x +", sym), std::runtime_error);
    }

} // namespace

namespace {

    TEST(EvalGeometry, ComplexExpression)
    {
        SymbolTable sym;
        sym.variables["w"] = 10.0;
        sym.variables["h"] = 2.0;

        EXPECT_EQ(mhs::expression::eval_geometry("(w+h)*2", sym), 24.0);
    }

    TEST(EvalGeometry, UndefinedVariable)
    {
        SymbolTable sym;
        sym.variables["defined"] = 1.0;

        // undefined variable should throw
        EXPECT_THROW(mhs::expression::eval_geometry("undefined", sym), std::runtime_error);
    }

} // namespace

namespace {

    TEST(Parse, SimpleConstant)
    {
        SymbolTable sym;
        auto expr = mhs::expression::parse("42", sym);
        EXPECT_TRUE(expr.is_constant());
        EXPECT_FALSE(expr.depends_on_temperature());
        EXPECT_EQ(expr.constant_value(), 42.0);
    }

    TEST(Parse, ComplexExpression)
    {
        SymbolTable sym;
        auto expr = mhs::expression::parse("(x + y) * z", sym);
        EXPECT_FALSE(expr.is_constant());

        mhs::expression::FieldContext ctx {1.0, 2.0, 3.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 9.0);
    }

} // namespace

namespace {

    TEST(Parse, SinFunction)
    {
        SymbolTable sym;
        auto expr = mhs::expression::parse("sin(x)", sym);
        EXPECT_FALSE(expr.is_constant());

        mhs::expression::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_NEAR(expr.eval(ctx), 0.0, 1e-10);

        ctx.x = PI / 2;
        EXPECT_NEAR(expr.eval(ctx), 1.0, 1e-10);
    }

} // namespace

namespace {

    TEST(Parse, CombinedContext)
    {
        SymbolTable sym;
        auto expr = mhs::expression::parse("x + y + z + T + t", sym);
        EXPECT_TRUE(expr.depends_on_temperature());
        EXPECT_FALSE(expr.is_constant());

        mhs::expression::FieldContext ctx {1.0, 2.0, 3.0, 10.0, 100.0};
        EXPECT_EQ(expr.eval(ctx), 116.0);
    }

} // namespace

namespace {

    TEST(ExprTest, ConcurrentEvaluationSingleExpression)
    {
        SymbolTable sym;
        auto expr = parse("x*x + 2*y - z + T*0.5 + t", sym);
        ASSERT_FALSE(expr.is_constant());

        const int num_threads = 16;
        const int num_iterations = 10000;

        mhs::core::Vector<double> results(num_threads * num_iterations, 0.0);
        mhs::core::Vector<std::thread> threads;

        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            threads.emplace_back([&, thread_id]() {
                for (int i = 0; i < num_iterations; ++i) {
                    double x = 1.0 + thread_id;
                    double y = 2.0 + i;
                    double z = 3.0 + thread_id * 0.1;
                    double T = 300.0 + i * 0.01;
                    double t_time = 0.1 * i;

                    FieldContext ctx {x, y, z, T, t_time};

                    double val = expr.eval(ctx);
                    results[thread_id * num_iterations + i] = val;
                }
            });
        }

        for (auto& th : threads) {
            th.join();
        }

        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (int i = 0; i < num_iterations; ++i) {
                double x = 1.0 + thread_id;
                double y = 2.0 + i;
                double z = 3.0 + thread_id * 0.1;
                double T = 300.0 + i * 0.01;
                double t_time = 0.1 * i;

                double expected = x * x + 2.0 * y - z + T * 0.5 + t_time;
                EXPECT_NEAR(results[thread_id * num_iterations + i], expected, 1e-9);
            }
        }
    }

    TEST(ExprTest, ConcurrentEvaluationDictionary)
    {
        SymbolTable sym;
        mhs::core::Vector<CompiledExpression> dict;
        dict.push_back(parse("0.0", sym));
        dict.push_back(parse("x + y", sym));
        dict.push_back(parse("x * y", sym));
        dict.push_back(parse("T^2 + t*10", sym));

        const int num_threads = 16;
        const int num_iterations = 5000;

        mhs::core::Vector<double> results(num_threads * num_iterations, 0.0);
        mhs::core::Vector<std::thread> threads;

        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            threads.emplace_back([&, thread_id]() {
                for (int i = 0; i < num_iterations; ++i) {
                    double x = (double)thread_id;
                    double y = (double)(i % 100);
                    double z = 0.0;
                    double T = 300.0 + thread_id;
                    double t_time = 1.0 + i * 0.1;

                    int expr_idx = (thread_id + i) % dict.size();
                    FieldContext ctx {x, y, z, T, t_time};

                    results[thread_id * num_iterations + i] = dict[expr_idx].eval(ctx);
                }
            });
        }

        for (auto& th : threads) {
            th.join();
        }

        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (int i = 0; i < num_iterations; ++i) {
                double x = (double)thread_id;
                double y = (double)(i % 100);
                [[maybe_unused]] double z = 0.0;
                double T = 300.0 + thread_id;
                double t_time = 1.0 + i * 0.1;

                int expr_idx = (thread_id + i) % dict.size();
                double expected = 0.0;

                if (expr_idx == 0)
                    expected = 0.0;
                else if (expr_idx == 1)
                    expected = x + y;
                else if (expr_idx == 2)
                    expected = x * y;
                else if (expr_idx == 3)
                    expected = std::pow(T, 2.0) + t_time * 10.0;

                EXPECT_NEAR(results[thread_id * num_iterations + i], expected, 1e-9)
                    << "Mismatch at thread " << thread_id << ", iteration " << i << " with expr_idx " << expr_idx;
            }
        }
    }

} // namespace

namespace {

    TEST(SymbolTable, EvalGeometryIsolatesPerCall)
    {
        mhs::expression::SymbolTable a;
        a.variables["x"] = 1.0;
        mhs::expression::SymbolTable b;
        b.variables["x"] = 100.0;

        EXPECT_EQ(mhs::expression::eval_geometry("x+1", a), 2.0);
        EXPECT_EQ(mhs::expression::eval_geometry("x+1", b), 101.0);
    }

    TEST(SymbolTable, ParseCapturesNatives)
    {
        mhs::expression::SymbolTable sym;
        // An unannotated callback is treated as reading the temperature field.
        sym.natives["twice"] = {[](const double* args, int /*nargs*/, const mhs::expression::FieldContext& /*ctx*/) {
                                    return 2.0 * args[0];
                                },
            true};
        auto expr = mhs::expression::parse("twice(3) + x", sym);
        EXPECT_TRUE(expr.depends_on_temperature());
        sym.natives["twice"].reads_temperature = false;
        EXPECT_FALSE(parse("twice(3)+x", sym).depends_on_temperature());
        EXPECT_TRUE(parse("twice(T)", sym).depends_on_temperature());
        EXPECT_FALSE(parse("t+1", sym).depends_on_temperature());
        EXPECT_FALSE(expr.is_constant());
        mhs::expression::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 0.0};
        ctx.x = 4.0;
        EXPECT_EQ(expr.eval(ctx), 10.0);
    }

    TEST(SymbolTable, ParallelParseDoesNotInterfere)
    {
        // Variables live in SymbolTable; they are exposed via eval_geometry, not parse.
        // Two threads construct their own SymbolTable and eval_geometry in parallel —
        // the result must reflect each thread's own table.
        const int N_THREADS = 8;
        mhs::core::Vector<std::thread> threads;
        mhs::core::Vector<double> results(N_THREADS, 0.0);

        for (int tid = 0; tid < N_THREADS; ++tid) {
            threads.emplace_back([tid, &results]() {
                mhs::expression::SymbolTable sym;
                sym.variables["k"] = static_cast<double>(tid + 1);
                results[tid] = mhs::expression::eval_geometry("k * 10", sym);
            });
        }
        for (auto& th : threads)
            th.join();

        for (int tid = 0; tid < N_THREADS; ++tid) {
            EXPECT_EQ(results[tid], static_cast<double>(tid + 1) * 10.0);
        }
    }

} // namespace
