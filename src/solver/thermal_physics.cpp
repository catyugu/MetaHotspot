#include "thermal_physics.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace mhs::solver {
    std::pair<double, double> interior_resistances(
        const model::Fom& model, model::CellId i, model::CellId j, std::size_t axis, double ki, double kj)
    {
        double ri = side_thermal_resistance(model.mesh.cell_width(i, axis) / 2, ki);
        double rj = side_thermal_resistance(model.mesh.cell_width(j, axis) / 2, kj);
        const auto& map = model.fluid.global_to_fluid;
        const auto& film = model.fluid.interface_heat_transfer_factor;
        if (!map.empty()) {
            const auto fi = map[i], fj = map[j];
            if (fi != core::invalid_id && fj == core::invalid_id)
                ri = fluid_film_resistance(film[fi], ki);
            else if (fj != core::invalid_id && fi == core::invalid_id)
                rj = fluid_film_resistance(film[fj], kj);
        }
        return {ri, rj};
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

    BoundaryEvaluation evaluate_boundary(const model::BoundaryCondition& condition, const model::Facet& patch,
        double cell_temperature, double time, double resistance)
    {
        using namespace mhs::model;
        const auto p = patch.center();
        const double area = patch.area(), conductive = area / resistance;
        auto context = [&](double trace) { return expression::FieldContext {p[0], p[1], p[2], trace, time}; };
        return std::visit(
            [&](const auto& law) -> BoundaryEvaluation {
                using C = std::decay_t<decltype(law)>;
                if constexpr (std::is_same_v<C, Dirichlet>) {
                    // The compiler rejects temperature-dependent Dirichlet data,
                    // so the prescribed trace is evaluated once.
                    const double trace = law.temperature.eval(context(cell_temperature));
                    return {trace, conductive, conductive * trace, true};
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
                    const auto [h, ambient] = values(cell_temperature);
                    const double condensed = convection_conductance(area, h, resistance);
                    const double guess = std::lerp(cell_temperature, ambient, condensed / conductive);
                    if (!law.coefficient.depends_on_temperature() && !law.ambient_temperature.depends_on_temperature())
                        return {guess, condensed, condensed * ambient};
                    const double trace = solve_trace(
                        [&](double value) {
                            const auto [coefficient, external_temperature] = values(value);
                            return conductive * (cell_temperature - value)
                                - area * coefficient * (value - external_temperature);
                        },
                        guess, cell_temperature);
                    const auto [coefficient, external_temperature] = values(trace);
                    const double g = convection_conductance(area, coefficient, resistance);
                    return {trace, g, g * external_temperature};
                }
            },
            condition);
    }

} // namespace mhs::solver
