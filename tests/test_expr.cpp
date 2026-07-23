#include "numerics/expression/expr.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
using namespace mhs::core;
namespace {
    constexpr double PI = 3.14159265358979323846;

    TEST(CompiledExpression, DefaultConstruct)
    {
        mhs::core::CompiledExpression expr;
        EXPECT_TRUE(expr.is_constant());
        EXPECT_EQ(expr.constant_value(), 0.0);
        EXPECT_EQ(expr.eval({}), 0.0);
    }

    TEST(CompiledExpression, MakeConstant)
    {
        auto expr = mhs::core::CompiledExpression::make_constant(42.0);
        EXPECT_TRUE(expr.is_constant());
        EXPECT_EQ(expr.constant_value(), 42.0);
        EXPECT_EQ(expr.eval({}), 42.0);
    }

    TEST(CompiledExpression, MakeEvaluator)
    {
        SymbolTable sym;
        auto expr = mhs::core::parse("x + y", sym);
        EXPECT_FALSE(expr.is_constant());

        mhs::core::FieldContext ctx {1.0, 2.0, 0.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 3.0);
    }

} // namespace

namespace {

    TEST(EvalGeometry, ComplexExpression)
    {
        SymbolTable sym;
        sym.variables["w"] = 10.0;
        sym.variables["h"] = 2.0;

        EXPECT_EQ(mhs::core::eval_geometry("(w+h)*2", sym), 24.0);
    }

    TEST(EvalGeometry, UndefinedVariable)
    {
        SymbolTable sym;
        sym.variables["defined"] = 1.0;

        // undefined variable should return 0.0
        EXPECT_EQ(mhs::core::eval_geometry("undefined", sym), 0.0);
    }

} // namespace

namespace {

    TEST(Parse, SimpleConstant)
    {
        SymbolTable sym;
        auto expr = mhs::core::parse("42", sym);
        EXPECT_TRUE(expr.is_constant());
        EXPECT_EQ(expr.constant_value(), 42.0);
    }

    TEST(Parse, ComplexExpression)
    {
        SymbolTable sym;
        auto expr = mhs::core::parse("(x + y) * z", sym);
        EXPECT_FALSE(expr.is_constant());

        mhs::core::FieldContext ctx {1.0, 2.0, 3.0, 0.0, 0.0};
        EXPECT_EQ(expr.eval(ctx), 9.0);
    }

} // namespace

namespace {

    TEST(Parse, SinFunction)
    {
        SymbolTable sym;
        auto expr = mhs::core::parse("sin(x)", sym);
        EXPECT_FALSE(expr.is_constant());

        mhs::core::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 0.0};
        EXPECT_NEAR(expr.eval(ctx), 0.0, 1e-10);

        ctx.x = PI / 2;
        EXPECT_NEAR(expr.eval(ctx), 1.0, 1e-10);
    }

} // namespace

namespace {

    TEST(Parse, CombinedContext)
    {
        SymbolTable sym;
        auto expr = mhs::core::parse("x + y + z + T + t", sym);
        EXPECT_FALSE(expr.is_constant());

        mhs::core::FieldContext ctx {1.0, 2.0, 3.0, 10.0, 100.0};
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

        std::vector<double> results(num_threads * num_iterations, 0.0);
        std::vector<std::thread> threads;

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
        std::vector<CompiledExpression> dict;
        dict.push_back(parse("0.0", sym));
        dict.push_back(parse("x + y", sym));
        dict.push_back(parse("x * y", sym));
        dict.push_back(parse("T^2 + t*10", sym));

        const int num_threads = 16;
        const int num_iterations = 5000;

        std::vector<double> results(num_threads * num_iterations, 0.0);
        std::vector<std::thread> threads;

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
        mhs::core::SymbolTable a;
        a.variables["x"] = 1.0;
        mhs::core::SymbolTable b;
        b.variables["x"] = 100.0;

        EXPECT_EQ(mhs::core::eval_geometry("x+1", a), 2.0);
        EXPECT_EQ(mhs::core::eval_geometry("x+1", b), 101.0);
    }

    TEST(SymbolTable, ParseCapturesNatives)
    {
        mhs::core::SymbolTable sym;
        sym.natives["twice"]
            = [](const double* args, int /*nargs*/, const mhs::core::FieldContext& /*ctx*/) { return 2.0 * args[0]; };
        auto expr = mhs::core::parse("twice(3) + x", sym);
        EXPECT_FALSE(expr.is_constant());
        mhs::core::FieldContext ctx {0.0, 0.0, 0.0, 0.0, 0.0};
        ctx.x = 4.0;
        EXPECT_EQ(expr.eval(ctx), 10.0);
    }

    TEST(SymbolTable, ParallelParseDoesNotInterfere)
    {
        // Variables live in SymbolTable; they are exposed via eval_geometry, not parse.
        // Two threads construct their own SymbolTable and eval_geometry in parallel —
        // the result must reflect each thread's own table.
        const int N_THREADS = 8;
        std::vector<std::thread> threads;
        std::vector<double> results(N_THREADS, 0.0);

        for (int tid = 0; tid < N_THREADS; ++tid) {
            threads.emplace_back([tid, &results]() {
                mhs::core::SymbolTable sym;
                sym.variables["k"] = static_cast<double>(tid + 1);
                results[tid] = mhs::core::eval_geometry("k * 10", sym);
            });
        }
        for (auto& th : threads)
            th.join();

        for (int tid = 0; tid < N_THREADS; ++tid) {
            EXPECT_EQ(results[tid], static_cast<double>(tid + 1) * 10.0);
        }
    }

} // namespace
