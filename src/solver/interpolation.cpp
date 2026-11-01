#include "solver/interpolation.hpp"
#include "core/boundary_geometry.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace mhs::utils {
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
    } // namespace

    BoundaryTrace evaluate_boundary(const mhs::core::Model& model, std::span<const double> temperature, double time,
        const mhs::core::BoundaryFace& boundary, double normal_conductivity)
    {
        using namespace mhs::core;
        const auto face = projected_face(model.discrete, boundary);
        const auto& bc = model.boundary_conditions[boundary.condition];
        const double cell_temperature = temperature[face.owner.value];
        const auto p = face.center();
        const double conductive = normal_conductivity * face.area / face.owner_distance;
        auto context = [&](double trace) { return FieldContext {p[0], p[1], p[2], trace, time}; };
        return std::visit(
            [&](const auto& condition) -> BoundaryTrace {
                using C = std::decay_t<decltype(condition)>;
                if constexpr (std::is_same_v<C, Dirichlet>) {
                    double trace = condition.temperature.eval(context(cell_temperature));
                    if (condition.temperature.is_constant())
                        return {trace, conductive, conductive * trace};
                    trace
                        = solve_trace([&](double value) { return value - condition.temperature.eval(context(value)); },
                            trace, cell_temperature);
                    return {trace, conductive, conductive * trace};
                }
                else if constexpr (std::is_same_v<C, Neumann>) {
                    auto flux = [&](double trace) { return condition.heat_flux.eval(context(trace)); };
                    const double initial_flux = flux(cell_temperature);
                    double trace = cell_temperature + face.area * initial_flux / conductive;
                    if (condition.heat_flux.is_constant())
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
                        return std::pair {condition.coefficient.eval(ctx), condition.ambient_temperature.eval(ctx)};
                    };
                    auto [h, ambient] = values(cell_temperature);
                    double convective = h * face.area;
                    double trace = (conductive * cell_temperature + convective * ambient) / (conductive + convective);
                    if (condition.coefficient.is_constant() && condition.ambient_temperature.is_constant()) {
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
            bc);
    }

    double boundary_trace(const mhs::core::Model& model, std::span<const double> temperature, double time,
        const mhs::core::BoundaryFace& face)
    {
        const auto projected = mhs::core::projected_face(model.discrete, face);
        const auto k = conductivity(model, temperature, projected.owner, time)[projected.axis];
        return evaluate_boundary(model, temperature, time, face, k).temperature;
    }

    std::vector<double> recover_boundary_temperature(
        const mhs::core::Model& model, std::span<const double> temperature, double time)
    {
        std::vector<double> traces;
        traces.reserve(model.boundary_faces.size());
        for (const auto& face : model.boundary_faces)
            traces.push_back(boundary_trace(model, temperature, time, face));
        return traces;
    }

    TemperatureGradient reconstruct_cell_gradient(
        const mhs::core::Model& model, std::span<const double> temperature, double time, mhs::core::CellId cell)
    {
        using namespace mhs::core;
        const auto k = conductivity(model, temperature, cell, time);
        Vec3 gradient {};
        for (auto id : model.discrete.incident_faces(cell)) {
            const auto& f = model.discrete.faces[id.value];
            double trace;
            if (!f.neighbor.valid()) {
                for (const auto& term : boundary_faces_on(model, id))
                    gradient[f.axis] += f.sign * projected_face(model.discrete, term).area
                        * boundary_trace(model, temperature, time, term);
                continue;
            }
            else {
                auto neighbor = f.other(cell);
                const double a = (f.owner == cell ? f.owner_distance : f.neighbor_distance) / k[f.axis];
                const double b = (f.owner == cell ? f.neighbor_distance : f.owner_distance)
                    / conductivity(model, temperature, neighbor, time)[f.axis];
                trace = (b * temperature[cell.value] + a * temperature[neighbor.value]) / (a + b);
            }
            gradient[f.axis] += (f.owner == cell ? f.sign : -f.sign) * f.area * trace;
        }
        const auto volume = model.discrete.cells[cell.value].volume();
        return {gradient[0] / volume, gradient[1] / volume, gradient[2] / volume};
    }
    double extrapolate_cell_temperature(
        double T, const TemperatureGradient& g, double x, double y, double z, double px, double py, double pz)
    { return T + g.x * (px - x) + g.y * (py - y) + g.z * (pz - z); }
    double recover_point_temperature(
        const std::vector<PointTemperatureSample>& samples, double px, double py, double pz)
    {
        assert(!samples.empty());
        double numerator = 0, denominator = 0;
        for (const auto& s : samples) {
            const auto x = px - s.x, y = py - s.y, z = pz - s.z;
            const double w
                = 1.0 / (x * x / s.kx + y * y / s.ky + z * z / s.kz + std::numeric_limits<double>::epsilon());
            numerator += w * s.temperature;
            denominator += w;
        }
        return numerator / denominator;
    }
    double sample_point(const mhs::core::Model& model, std::span<const double> temperature, double time,
        mhs::core::CellId cell, const mhs::core::Vec3& p)
    {
        if (!cell.valid())
            return std::numeric_limits<double>::quiet_NaN();
        const auto T = temperature[cell.value];
        double sum = 0, dirichlet_sum = 0;
        size_t count = 0, dirichlet_count = 0;
        for (auto id : model.discrete.incident_faces(cell)) {
            const auto& f = model.discrete.faces[id.value];
            if (f.neighbor.valid())
                continue;
            for (const auto& boundary : mhs::core::boundary_faces_on(model, id)) {
                if (!mhs::core::projected_face(model.discrete, boundary).contains(p))
                    continue;
                const auto& bc = model.boundary_conditions[boundary.condition];
                const double trace = boundary_trace(model, temperature, time, boundary);
                sum += trace;
                ++count;
                if (std::holds_alternative<mhs::core::Dirichlet>(bc)) {
                    dirichlet_sum += trace;
                    ++dirichlet_count;
                }
            }
        }
        if (dirichlet_count)
            return dirichlet_sum / static_cast<double>(dirichlet_count);
        if (count)
            return sum / static_cast<double>(count);
        const auto g = reconstruct_cell_gradient(model, temperature, time, cell);
        const auto center = model.discrete.cells[cell.value].center();
        return extrapolate_cell_temperature(T, g, center[0], center[1], center[2], p[0], p[1], p[2]);
    }
} // namespace mhs::utils
