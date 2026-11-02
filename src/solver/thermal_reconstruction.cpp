#include "thermal_reconstruction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace mhs::sim {
    namespace {
        mhs::core::Vec3 conductivity(
            const mhs::core::Model& model, std::span<const double> temperature, mhs::core::CellId cell, double time)
        {
            const auto p = model.discrete.cells[cell.value].center();
            const mhs::core::FieldContext ctx {p[0], p[1], p[2], temperature[cell.value], time};
            const auto& m = model.material_table[model.cell_material[cell.value]];
            return {m.kx.eval(ctx), m.ky.eval(ctx), m.kz.eval(ctx)};
        }

        template <class Residual> double solve_trace(Residual&& residual, double guess, double scale)
        {
            auto converged = [&](double x, double value) {
                return std::abs(value) <= 1e-14 * (1.0 + std::abs(scale) + std::abs(x));
            };
            const double fguess = residual(guess);
            if (converged(guess, fguess))
                return guess;
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
            if (converged(lo, flo))
                return lo;
            if (converged(hi, fhi))
                return hi;
            if (std::signbit(flo) == std::signbit(fhi))
                throw std::runtime_error("boundary trace equation has no bracketed root");
            for (int i = 0; i < 64; ++i) {
                const double mid = std::midpoint(lo, hi);
                const double fmid = residual(mid);
                if (!std::isfinite(fmid))
                    throw std::runtime_error("non-finite boundary trace residual");
                if (converged(mid, fmid) || std::abs(hi - lo) <= 1e-14 * (1.0 + std::abs(mid)))
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

        // Green-Gauss gradient on an orthogonal cell. Internal face temperatures
        // use the same two-point thermal-resistance law as the diffusion assembly.
        mhs::core::Vec3 reconstruct_cell_gradient(
            const mhs::core::Model& model, std::span<const double> temperature, double time, mhs::core::CellId cell)
        {
            using namespace mhs::core;
            const auto k = conductivity(model, temperature, cell, time);
            Vec3 gradient {};
            for (auto id : model.discrete.incident_faces(cell)) {
                const auto& f = model.discrete.faces[id.value];
                if (!f.neighbor.valid()) {
                    for (const auto& term : model.boundary_faces_on(id)) {
                        const auto patch = model.discrete.faces[term.parent.value].project(term.bounds);
                        const auto trace = evaluate_boundary(model.boundary_conditions[term.condition], patch,
                            temperature[cell.value], k[patch.axis], time);
                        gradient[f.axis] += f.sign * patch.area * trace.temperature;
                    }
                    continue;
                }
                const auto neighbor = f.other(cell);
                const double a = (f.owner == cell ? f.owner_distance : f.neighbor_distance) / k[f.axis];
                const double b = (f.owner == cell ? f.neighbor_distance : f.owner_distance)
                    / conductivity(model, temperature, neighbor, time)[f.axis];
                const double trace = (b * temperature[cell.value] + a * temperature[neighbor.value]) / (a + b);
                gradient[f.axis] += (f.owner == cell ? f.sign : -f.sign) * f.area * trace;
            }
            const auto volume = model.discrete.cells[cell.value].volume();
            return {gradient[0] / volume, gradient[1] / volume, gradient[2] / volume};
        }
    } // namespace

    BoundaryTrace evaluate_boundary(const mhs::core::BoundaryCondition& condition, const mhs::core::FacePatch& face,
        double cell_temperature, double normal_conductivity, double time)
    {
        using namespace mhs::core;
        const auto p = face.center();
        const double conductive = normal_conductivity * face.area / face.owner_distance;
        auto context = [&](double trace) { return FieldContext {p[0], p[1], p[2], trace, time}; };
        return std::visit(
            [&](const auto& law) -> BoundaryTrace {
                using C = std::decay_t<decltype(law)>;
                if constexpr (std::is_same_v<C, Dirichlet>) {
                    double trace = law.temperature.eval(context(cell_temperature));
                    if (!law.temperature.depends_on_temperature())
                        return {trace, conductive, conductive * trace};
                    trace = solve_trace([&](double value) { return value - law.temperature.eval(context(value)); },
                        trace, cell_temperature);
                    return {trace, conductive, conductive * trace};
                }
                else if constexpr (std::is_same_v<C, Neumann>) {
                    auto flux = [&](double trace) { return law.heat_flux.eval(context(trace)); };
                    const double initial_flux = flux(cell_temperature);
                    double trace = cell_temperature + face.area * initial_flux / conductive;
                    if (!law.heat_flux.depends_on_temperature())
                        return {trace, 0.0, face.area * initial_flux};
                    trace = solve_trace(
                        [&](double value) { return conductive * (cell_temperature - value) + face.area * flux(value); },
                        trace, cell_temperature);
                    const double q = flux(trace);
                    return {trace, 0.0, face.area * q};
                }
                else {
                    static_assert(std::is_same_v<C, Convection>);
                    auto values = [&](double trace) {
                        const auto ctx = context(trace);
                        return std::pair {law.coefficient.eval(ctx), law.ambient_temperature.eval(ctx)};
                    };
                    auto [h, ambient] = values(cell_temperature);
                    double convective = h * face.area;
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
                                - face.area * coefficient * (value - external_temperature);
                        },
                        trace, cell_temperature);
                    std::tie(h, ambient) = values(trace);
                    convective = h * face.area;
                    const double condensed = conductive * convective / (conductive + convective);
                    return {trace, condensed, condensed * ambient};
                }
            },
            condition);
    }

    std::optional<double> boundary_value_at(const mhs::core::Model& model, std::span<const double> temperature,
        double time, mhs::core::CellId cell, const mhs::core::Vec3& p)
    {
        if (!cell.valid())
            return std::nullopt;
        const double cell_temperature = temperature[cell.value];
        std::optional<mhs::core::Vec3> k;
        double sum = 0, dirichlet_sum = 0;
        size_t count = 0, dirichlet_count = 0;
        for (auto id : model.discrete.incident_faces(cell)) {
            if (model.discrete.faces[id.value].neighbor.valid())
                continue;
            for (const auto& term : model.boundary_faces_on(id)) {
                const auto patch = model.discrete.faces[term.parent.value].project(term.bounds);
                if (!patch.contains(p))
                    continue;
                if (!k)
                    k = conductivity(model, temperature, cell, time);
                const auto& condition = model.boundary_conditions[term.condition];
                const double trace
                    = evaluate_boundary(condition, patch, cell_temperature, (*k)[patch.axis], time).temperature;
                sum += trace;
                ++count;
                if (std::holds_alternative<mhs::core::Dirichlet>(condition)) {
                    dirichlet_sum += trace;
                    ++dirichlet_count;
                }
            }
        }
        if (dirichlet_count)
            return dirichlet_sum / static_cast<double>(dirichlet_count);
        if (count)
            return sum / static_cast<double>(count);
        return std::nullopt;
    }

    double sample_point(const mhs::core::Model& model, std::span<const double> temperature, double time,
        mhs::core::CellId cell, const mhs::core::Vec3& p)
    {
        if (!cell.valid())
            return std::numeric_limits<double>::quiet_NaN();
        if (const auto trace = boundary_value_at(model, temperature, time, cell, p))
            return *trace;
        const auto gradient = reconstruct_cell_gradient(model, temperature, time, cell);
        const auto center = model.discrete.cells[cell.value].center();
        return temperature[cell.value] + gradient[0] * (p[0] - center[0]) + gradient[1] * (p[1] - center[1])
            + gradient[2] * (p[2] - center[2]);
    }

} // namespace mhs::sim
