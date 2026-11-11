#include "thermal_physics.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace mhs::solver {
    std::pair<double, double> connection_resistances(const mhs::model::Model& model, mhs::model::Connection connection,
        const mhs::model::CellGeometry& a, const mhs::model::CellGeometry& b, mhs::model::Axis normal, double k_a,
        double k_b)
    {
        const auto axis = static_cast<std::size_t>(normal);
        double ra = side_thermal_resistance(a.size()[axis] / 2, k_a);
        double rb = side_thermal_resistance(b.size()[axis] / 2, k_b);
        const auto& fluid = model.fluid;
        if (!fluid.global_to_fluid.empty()) {
            const auto fi = fluid.global_to_fluid[connection.a], fj = fluid.global_to_fluid[connection.b];
            if (fi != mhs::core::invalid_id && fj == mhs::core::invalid_id)
                ra = fluid_film_resistance(fluid.interface_heat_transfer_factor[fi], k_a);
            else if (fj != mhs::core::invalid_id && fi == mhs::core::invalid_id)
                rb = fluid_film_resistance(fluid.interface_heat_transfer_factor[fj], k_b);
        }
        return {ra, rb};
    }
    namespace {
        // Bracketed bisection on the trace temperature. The condensed
        // coefficients only need the trace to a few digits beyond the global
        // nonlinear tolerance, so the loop stops on relative trace width rather
        // than on a residual whose unit changes between the boundary laws.
        template <class Residual> double solve_trace(Residual&& residual, double guess, double scale)
        {
            constexpr double trace_tolerance = 1e-9;
            double step = std::max(1.0, 0.05 * std::max(std::abs(guess), std::abs(scale)));
            double lo = guess - step, hi = guess + step;
            double flo = residual(lo), fhi = residual(hi);
            for (int i = 0; i < 24 && std::signbit(flo) == std::signbit(fhi); ++i) {
                step *= 2.0;
                lo = guess - step;
                hi = guess + step;
                flo = residual(lo);
                fhi = residual(hi);
            }
            if (!std::isfinite(flo) || !std::isfinite(fhi))
                throw std::runtime_error("non-finite boundary trace residual");
            if (std::signbit(flo) == std::signbit(fhi))
                throw std::runtime_error("boundary trace equation has no bracketed root");
            for (int i = 0; i < 64; ++i) {
                const double mid = std::midpoint(lo, hi);
                const double fmid = residual(mid);
                if (!std::isfinite(fmid))
                    throw std::runtime_error("non-finite boundary trace residual");
                if (fmid == 0.0 || std::abs(hi - lo) <= trace_tolerance * std::max(1.0, std::abs(mid)))
                    return mid;
                if (std::signbit(fmid) == std::signbit(flo)) {
                    lo = mid;
                    flo = fmid;
                }
                else {
                    hi = mid;
                    fhi = fmid;
                }
            }
            return std::midpoint(lo, hi);
        }

    } // namespace

    BoundaryTrace evaluate_boundary(const mhs::model::BoundaryCondition& condition,
        mhs::expression::FieldContext cell_context, double area, double conductive)
    {
        using namespace mhs::model;
        const double cell_temperature = cell_context.T;
        auto context = [&](double trace) {
            auto result = cell_context;
            result.T = trace;
            return result;
        };
        return std::visit(
            [&](const auto& law) -> BoundaryTrace {
                using C = std::decay_t<decltype(law)>;
                if constexpr (std::is_same_v<C, Dirichlet>) {
                    // The compiler rejects temperature-dependent Dirichlet data,
                    // so the prescribed trace is evaluated once.
                    const double trace = law.temperature.eval(context(cell_temperature));
                    return {trace, conductive, conductive * trace};
                }
                else if constexpr (std::is_same_v<C, Neumann>) {
                    auto flux = [&](double trace) { return law.heat_flux.eval(context(trace)); };
                    const double initial_flux = flux(cell_temperature);
                    double trace = cell_temperature + area * initial_flux / conductive;
                    if (!law.heat_flux.depends_on_temperature())
                        return {trace, 0.0, area * initial_flux};
                    trace = solve_trace(
                        [&](double value) { return conductive * (cell_temperature - value) + area * flux(value); },
                        trace, cell_temperature);
                    const double q = flux(trace);
                    return {trace, 0.0, area * q};
                }
                else {
                    static_assert(std::is_same_v<C, Convection>);
                    auto values = [&](double trace) {
                        const auto ctx = context(trace);
                        return std::pair {law.coefficient.eval(ctx), law.ambient_temperature.eval(ctx)};
                    };
                    auto [h, ambient] = values(cell_temperature);
                    double convective = h * area;
                    double trace = (conductive * cell_temperature + convective * ambient) / (conductive + convective);
                    if (!law.coefficient.depends_on_temperature()
                        && !law.ambient_temperature.depends_on_temperature()) {
                        const double condensed = conductive * convective / (conductive + convective);
                        return {trace, condensed, condensed * ambient};
                    }
                    trace = solve_trace(
                        [&](double value) {
                            const auto [coefficient, external_temperature] = values(value);
                            return conductive * (cell_temperature - value)
                                - area * coefficient * (value - external_temperature);
                        },
                        trace, cell_temperature);
                    std::tie(h, ambient) = values(trace);
                    convective = h * area;
                    const double condensed = conductive * convective / (conductive + convective);
                    return {trace, condensed, condensed * ambient};
                }
            },
            condition);
    }

} // namespace mhs::solver
