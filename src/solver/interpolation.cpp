#include "solver/interpolation.hpp"
#include <cassert>
#include <cmath>
#include <limits>

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
    } // namespace
    double boundary_trace(const mhs::core::Model& model, std::span<const double> temperature, double time,
        const mhs::core::BoundaryTerm& term)
    {
        const auto face = mhs::core::projected_face(model.discrete, term);
        const auto& bc = term.bc;
        const auto T = temperature[face.owner.value];
        const auto p = face.center();
        const mhs::core::FieldContext ctx {p[0], p[1], p[2], T, time};
        const auto k = conductivity(model, temperature, face.owner, time)[face.axis];
        const auto& b = model.bc_params;
        switch (bc.type) {
        case mhs::core::BcType::FirstType:
            return b.dirichlet_T[bc.param_idx].eval(ctx);
        case mhs::core::BcType::SecondType:
            return T + b.neumann_q[bc.param_idx].eval(ctx) * face.owner_distance / k;
        case mhs::core::BcType::ThirdType: {
            auto h = b.cauchy_h[bc.param_idx].eval(ctx), g = k / face.owner_distance;
            return (h * b.cauchy_T_inf[bc.param_idx].eval(ctx) + g * T) / (h + g);
        }
        default:
            return T;
        }
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
                for (const auto& term : boundary_terms_on_face(model, id))
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
        double sum = 0;
        size_t count = 0;
        for (auto id : model.discrete.incident_faces(cell)) {
            const auto& f = model.discrete.faces[id.value];
            if (f.neighbor.valid())
                continue;
            for (const auto& term : mhs::core::boundary_terms_on_face(model, id)) {
                const auto& bc = term.bc;
                if (bc.type == mhs::core::BcType::FirstType
                    && mhs::core::projected_face(model.discrete, term).contains(p)) {
                    sum += model.bc_params.dirichlet_T[bc.param_idx].eval({p[0], p[1], p[2], T, time});
                    ++count;
                }
            }
        }
        if (count)
            return sum / static_cast<double>(count);
        const auto g = reconstruct_cell_gradient(model, temperature, time, cell);
        const auto center = model.discrete.cells[cell.value].center();
        return extrapolate_cell_temperature(T, g, center[0], center[1], center[2], p[0], p[1], p[2]);
    }
} // namespace mhs::utils
