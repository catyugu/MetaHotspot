#include "geometry_expression.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mhs::app::pta {
    namespace {
        bool identifier_start(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }

        bool identifier_part(char c) { return identifier_start(c) || (c >= '0' && c <= '9'); }

        class Parser {
        public:
            Parser(std::string_view input, const std::unordered_map<std::string, double>& values)
                : input_(input), values_(values)
            {
            }

            double evaluate()
            {
                skip_space();
                if (pos_ == input_.size())
                    return 0.0; // Optional PTA geometry fields default to zero.
                const double result = expression();
                skip_space();
                if (pos_ != input_.size())
                    fail("unexpected token");
                if (!std::isfinite(result))
                    fail("non-finite result");
                return result;
            }

        private:
            void skip_space()
            {
                while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_])))
                    ++pos_;
            }

            bool consume(char c)
            {
                skip_space();
                if (pos_ < input_.size() && input_[pos_] == c) {
                    ++pos_;
                    return true;
                }
                return false;
            }

            [[noreturn]] void fail(const std::string& message) const
            {
                throw std::invalid_argument("invalid PTA geometry expression '" + std::string(input_) + "' at "
                    + std::to_string(pos_) + ": " + message);
            }

            double expression()
            {
                double value = term();
                while (true) {
                    if (consume('+'))
                        value += term();
                    else if (consume('-'))
                        value -= term();
                    else
                        return value;
                }
            }

            double term()
            {
                double value = unary();
                while (true) {
                    if (consume('*'))
                        value *= unary();
                    else if (consume('/'))
                        value /= unary();
                    else
                        return value;
                }
            }

            double unary()
            {
                if (consume('+'))
                    return unary();
                if (consume('-'))
                    return -unary();
                return power();
            }

            double power()
            {
                double value = primary();
                if (consume('^'))
                    value = std::pow(value, unary()); // Right associative, tighter than unary sign.
                return value;
            }

            double primary()
            {
                if (consume('(')) {
                    const double value = expression();
                    if (!consume(')'))
                        fail("expected ')'");
                    return value;
                }
                skip_space();
                if (pos_ == input_.size())
                    fail("expected number or variable");
                if (identifier_start(input_[pos_])) {
                    const auto start = pos_++;
                    while (pos_ < input_.size() && identifier_part(input_[pos_]))
                        ++pos_;
                    const std::string name(input_.substr(start, pos_ - start));
                    auto it = values_.find(name);
                    if (it == values_.end())
                        fail("unknown variable '" + name + "'");
                    return it->second;
                }
                if ((input_[pos_] >= '0' && input_[pos_] <= '9') || input_[pos_] == '.') {
                    const char* begin = input_.data() + pos_;
                    const char* end = input_.data() + input_.size();
                    double value = 0.0;
                    const auto parsed = std::from_chars(begin, end, value, std::chars_format::general);
                    if (parsed.ec != std::errc() || parsed.ptr == begin)
                        fail("invalid number");
                    pos_ = static_cast<std::size_t>(parsed.ptr - input_.data());
                    return value;
                }
                fail("expected number or variable");
            }

            std::string_view input_;
            const std::unordered_map<std::string, double>& values_;
            std::size_t pos_ = 0;
        };
    } // namespace

    GeometryEvaluator::GeometryEvaluator(const mhs::core::Vector<VariableSpec>& variables)
    {
        values_.reserve(variables.size());
        for (const auto& variable : variables) {
            const auto& name = variable.name;
            if (name.empty() || !identifier_start(name.front()))
                throw std::invalid_argument("invalid PTA geometry variable name: '" + name + "'");
            for (char c : name)
                if (!identifier_part(c))
                    throw std::invalid_argument("invalid PTA geometry variable name: '" + name + "'");
            if (values_.contains(name))
                throw std::invalid_argument("duplicate PTA geometry variable: '" + name + "'");
            if (variable.value.empty())
                throw std::invalid_argument("empty PTA geometry variable: '" + name + "'");
            try {
                values_.emplace(name, evaluate(variable.value));
            }
            catch (const std::exception& e) {
                throw std::invalid_argument("PTA geometry variable '" + name + "': " + e.what());
            }
        }
    }

    double GeometryEvaluator::evaluate(std::string_view expr) const { return Parser(expr, values_).evaluate(); }

} // namespace mhs::app::pta
