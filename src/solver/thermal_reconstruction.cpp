#include "thermal_reconstruction.hpp"
#include <numeric>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace mhs::solver {
    mhs::model::CellId locate_cell(const mhs::model::DiscreteMesh& mesh, const mhs::model::Vec3& p)
    {
        using namespace mhs::model;
        const auto& cells = mesh.cells;
        // Cache once per probe. Shared-face ties select the lexicographically
        // greatest lower corner, independent of cell storage/DOF ordering.
        CellId result = mhs::core::invalid_id;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            const auto& cell = cells[i];
            bool contains = true;
            for (std::size_t axis = 0; axis < 3; ++axis)
                contains = contains && p[axis] >= cell.min[axis] && p[axis] <= cell.max[axis];
            if (contains && (result == mhs::core::invalid_id || cell.min > cells[result].min))
                result = static_cast<CellId>(i);
        }
        return result;
    }

    namespace {
        double conductivity(const mhs::model::Model& model, std::span<const double> temperature,
            mhs::model::CellId cell, double time, int axis)
        {
            const auto p = model.discrete.cells[cell].center();
            const mhs::expression::FieldContext ctx {p[0], p[1], p[2], temperature[cell], time};
            const auto& m = model.material_table[model.cell_material[cell]];
            return (axis == 0 ? m.kx : axis == 1 ? m.ky : m.kz).eval(ctx);
        }

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

        // Green-Gauss gradient on an orthogonal cell. Internal face temperatures
        // use the same two-point thermal-resistance law as the diffusion assembly.
        mhs::model::Vec3 reconstruct_cell_gradient(
            const mhs::model::Model& model, std::span<const double> temperature, double time, mhs::model::CellId cell)
        {
            using namespace mhs::core;
            using namespace mhs::model;
            const Vec3 k {conductivity(model, temperature, cell, time, 0),
                conductivity(model, temperature, cell, time, 1), conductivity(model, temperature, cell, time, 2)};
            Vec3 gradient {};
            for (auto id : model.discrete.incident_faces(cell)) {
                const auto& f = model.discrete.faces[id];
                if (f.exterior()) {
                    for (const auto& term : boundary_faces_on(model, id)) {
                        const auto trace = evaluate_boundary(model.boundary_conditions[term.condition],
                            f.center(model.discrete.cells, term.bounds), f.area(term.bounds),
                            f.distance(model.discrete.cells[f.cell()]), temperature[cell],
                            k[static_cast<std::size_t>(f.axis)], time);
                        gradient[static_cast<std::size_t>(f.axis)]
                            += f.normal_sign(f.cell()) * f.area(term.bounds) * trace.temperature;
                    }
                    continue;
                }
                const auto neighbor = f.other(cell);
                const double a = f.distance(model.discrete.cells[cell]) / k[static_cast<std::size_t>(f.axis)];
                const double b = f.distance(model.discrete.cells[neighbor])
                    / conductivity(model, temperature, neighbor, time, static_cast<int>(f.axis));
                const double trace = (b * temperature[cell] + a * temperature[neighbor]) / (a + b);
                gradient[static_cast<std::size_t>(f.axis)] += f.normal_sign(cell) * f.area() * trace;
            }
            const auto volume = model.discrete.cells[cell].volume();
            return {gradient[0] / volume, gradient[1] / volume, gradient[2] / volume};
        }
    } // namespace

    std::span<const mhs::model::BoundaryFace> boundary_faces_on(
        const mhs::model::Model& model, mhs::model::FaceId parent)
    {
        const auto& patches = model.boundary_faces;
        auto first = std::lower_bound(patches.begin(), patches.end(), parent,
            [](const mhs::model::BoundaryFace& patch, mhs::model::FaceId id) { return patch.parent < id; });
        auto last = first;
        while (last != patches.end() && last->parent == parent)
            ++last;
        return {first, last};
    }

    BoundaryTrace evaluate_boundary(const mhs::model::BoundaryCondition& condition, const mhs::model::Vec3& p,
        double area, double normal_distance, double cell_temperature, double normal_conductivity, double time)
    {
        using namespace mhs::core;
        using namespace mhs::model;
        using namespace mhs::expression;
        const double conductive = normal_conductivity * area / normal_distance;
        auto context = [&](double trace) { return FieldContext {p[0], p[1], p[2], trace, time}; };
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

    std::optional<double> boundary_value_at(const mhs::model::Model& model, std::span<const double> temperature,
        double time, mhs::model::CellId cell, const mhs::model::Vec3& p)
    {
        if (cell == mhs::core::invalid_id)
            return std::nullopt;
        const double cell_temperature = temperature[cell];
        double sum = 0, dirichlet_sum = 0;
        size_t count = 0, dirichlet_count = 0;
        for (auto id : model.discrete.incident_faces(cell)) {
            const auto& f = model.discrete.faces[id];
            if (!f.exterior())
                continue;
            for (const auto& term : boundary_faces_on(model, id)) {
                if (!f.contains(model.discrete.cells, p, term.bounds))
                    continue;
                const auto& condition = model.boundary_conditions[term.condition];
                const double trace = evaluate_boundary(condition, f.center(model.discrete.cells, term.bounds),
                    f.area(term.bounds), f.distance(model.discrete.cells[f.cell()]), cell_temperature,
                    conductivity(model, temperature, cell, time, static_cast<int>(f.axis)), time)
                                         .temperature;
                sum += trace;
                ++count;
                if (std::holds_alternative<mhs::model::Dirichlet>(condition)) {
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

    double sample_point(const mhs::model::Model& model, std::span<const double> temperature, double time,
        mhs::model::CellId cell, const mhs::model::Vec3& p)
    {
        if (cell == mhs::core::invalid_id)
            return std::numeric_limits<double>::quiet_NaN();
        if (const auto trace = boundary_value_at(model, temperature, time, cell, p))
            return *trace;
        const auto gradient = reconstruct_cell_gradient(model, temperature, time, cell);
        const auto center = model.discrete.cells[cell].center();
        return temperature[cell] + gradient[0] * (p[0] - center[0]) + gradient[1] * (p[1] - center[1])
            + gradient[2] * (p[2] - center[2]);
    }

} // namespace mhs::solver
