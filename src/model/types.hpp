#pragma once

#include <cstdint>
#include <functional>

namespace mhs {

    enum class StudyType { Steady,
        Transient };

    enum class BcType : uint8_t { None = 0,
        FirstType = 1,
        SecondType = 2,
        ThirdType = 3 };

    struct FieldContext {
        double x = 0.0, y = 0.0, z = 0.0;
        double T = 0.0;
        double t = 0.0;
    };

    using FieldEvaluator = std::function<double(const FieldContext&)>;

    class FieldExpression {
    public:
        FieldExpression() : is_const_(false), const_val_(0.0) { }
        explicit FieldExpression(FieldEvaluator eval) : is_const_(false), const_val_(0.0), eval_(std::move(eval)) { }
        explicit FieldExpression(double constant_value) : is_const_(true), const_val_(constant_value) { }

        double eval(const FieldContext& ctx) const
        {
            return is_const_ ? const_val_ : (eval_ ? eval_(ctx) : 0.0);
        }

        bool is_constant() const { return is_const_; }
        double constant_value() const { return const_val_; }

        void set_evaluator(FieldEvaluator eval)
        {
            eval_ = std::move(eval);
            is_const_ = false;
        }

        void set_constant(double value)
        {
            is_const_ = true;
            const_val_ = value;
            eval_ = nullptr;
        }

    private:
        bool is_const_;
        double const_val_;
        FieldEvaluator eval_;
    };

} // namespace mhs
