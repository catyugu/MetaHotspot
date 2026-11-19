#include "compiler/model_compiler.hpp"
#include "model/project.hpp"
#include "pta/io.hpp"
#include "solver/solver.hpp"

#include <cmath>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string_view>

using namespace mhs;
using namespace mhs::model;

namespace {
    void check(bool passed, const char* message)
    {
        if (!passed)
            throw std::runtime_error(message);
    }

    void near(
        double a, double b, double tolerance = 1e-10, const std::source_location at = std::source_location::current())
    {
        if (std::abs(a - b) > tolerance || !std::isfinite(a) || !std::isfinite(b)) {
            std::cerr.precision(16);
            std::cerr << "line " << at.line() << ": " << a << " vs " << b << ", tolerance " << tolerance << '\n';
            throw std::runtime_error("numeric mismatch");
        }
    }

    void check_asset(const Erom& e)
    {
        check(e.K.rows() == 3 && e.C.rows() == 3 && e.Vb.rows() == 4 && e.B.cols() == 2 && e.L.rows() == 3,
            "wrong matrix dimensions");
        check(e.surfaces.size() == 3 && e.htc[0] && e.htc[1] && !e.htc[2], "wrong HTC groups");
        check(e.surfaces[0].id == 0 && e.surfaces[0].name == "upstream"
                && e.surfaces[0].facets == core::Vector<FacetId> {0, 1} && e.surfaces[1].id == 1
                && e.surfaces[1].facets == core::Vector<FacetId> {2, 3},
            "wrong facet ownership");
        check(e.sources[0].id == 0 && e.sources[0].name == "heater A" && e.sources[1].id == 2 && e.outputs[0].id == 0
                && e.outputs[0].name == "off center" && e.outputs[1].source == 0 && e.outputs[2].source == 2,
            "wrong source or output identity");
        near(e.sources[0].default_power, 5);
        near(e.sources[1].default_power, 8);
        near(e.htc[1]->maximum, 9);
        near(e.reference_temperature, 295);
        near(e.initial_temperature, 295);
        for (double r : e.R)
            near(r, 0.25);
        near(e.facets[3].area(), 0.5);

        // Reconstruct the three x-cell rows of the basis from the two end
        // surfaces and the Green-Gauss observation at x=1.2.
        core::Matrix<double> v(3, 3);
        for (std::size_t i = 0; i < 3; ++i) {
            near(e.Vb(0, i), e.Vb(1, i));
            near(e.Vb(2, i), e.Vb(3, i));
            v(0, i) = e.Vb(0, i);
            v(2, i) = e.Vb(2, i);
            v(1, i) = e.L(0, i) - 0.15 * (v(0, i) - v(2, i));
            near(e.L(1, i), v(0, i));
            near(e.L(2, i), v(2, i));
            near(e.B(i, 0), v(0, i));
            near(e.B(i, 1), v(2, i));
            near(e.initial_state[i], 0);
        }
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                double k = 0, c = 0;
                for (std::size_t x = 0; x < 3; ++x)
                    c += 4 * v(x, i) * v(x, j);
                for (std::size_t x = 1; x < 3; ++x)
                    k += 2 * (v(x, i) - v(x - 1, i)) * (v(x, j) - v(x - 1, j));
                near(e.K(i, j), k);
                near(e.C(i, j), c);
            }
    }

    void compare(const Project& mixed, const Project& full, Study study)
    {
        solver::SolveOptions opts;
        opts.time_tolerance = 1e-8;
        opts.max_time_step = 0.02;
        opts.linear_tolerance = 1e-11;
        opts.nonlinear_relative_tolerance = 1e-10;
        const auto detailed = solver::solve(full, study, {}, opts), reduced = solver::solve(mixed, study, {}, opts);
        check(detailed.converged && reduced.converged, "a reference or mixed solve did not converge");
        check(detailed.snapshot_times == reduced.snapshot_times, "probe timeline mismatch");
        check(reduced.probe_traces.size() == 3, "mixed output count mismatch");
        check(reduced.find_probe(10, 0) == &reduced.probe_traces[0], "mixed output identity mismatch");

        for (std::size_t i = 0; i < detailed.snapshot_times.size(); ++i) {
            const auto* t = detailed.snapshot_states.data() + i * full.dof_count();
            near(reduced.probe_traces[0].values[i], detailed.probe_traces[0].values[i], 1e-4);
            near(reduced.probe_traces[1].values[i], (t[0] + t[1]) / 2, 1e-4);
            near(reduced.probe_traces[2].values[i], (t[4] + t[5]) / 2, 1e-4);
            if (i == 0 && study.type == StudyType::Transient)
                near(reduced.probe_traces[0].values[i], study.initial_temperature, 1e-10);
        }
        const auto& t = detailed.state;
        check(std::abs(detailed.probe_traces[0].values.back() - (t[2] + t[3]) / 2) > 0.01,
            "fixture does not distinguish interpolation from nearest cell");
        near(reduced.state.back(), detailed.state.back(), 1e-4);
    }
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 4 && std::string_view(argv[1]) == "--reject") {
            try {
                (void)read_erom(argv[2]);
            }
            catch (const std::invalid_argument& error) {
                check(std::string_view(error.what()).find(argv[3]) != std::string_view::npos, error.what());
                return 0;
            }
            throw std::runtime_error("invalid EROM file was accepted");
        }
        if (argc == 3 && std::string_view(argv[1]) == "--no-channels") {
            const auto e = read_erom(argv[2]);
            check(e->sources.empty() && e->outputs.empty() && e->B.cols() == 0 && e->L.rows() == 0,
                "expected empty source and output channels");
            const auto project = compile_project({{{0, {}, e, {}, {}}}, {}});
            check(solver::solve(project).converged, "source-free EROM did not converge");
            return 0;
        }
        check(argc == 3, "expected a .mherom and a PTA reference");
        const auto e = read_erom(argv[1]);
        check_asset(*e);
        auto input = app::pta::read_xml(argv[2]);
        input.model.point_probes.resize(1);
        auto fom = std::make_shared<const Fom>(compiler::build_model(input.model, input.mesh));

        ModelDefinition sink;
        sink.materials.push_back(input.model.materials[0]);
        sink.volumes = {{0, core::invalid_id, {{GeometryOperation::Add, {}, {1, 1, 1}}}}};
        sink.thermal_boundaries = {{{{Axis::X, 1, {0, 1, 0, 1}}}, DirichletBoundary {"288"}}};
        auto body = std::make_shared<const Fom>(compiler::build_model(sink, {{0, 1}, {0, 1}, {0, 1}}));
        Component external {20, {3, 0, 0}, body, {}, {}};
        const auto mixed = compile_project({{{10, {}, e, {}, {}}, external}, {}});
        const auto full = compile_project({{{10, {}, fom, {}, {}}, external}, {}});
        check(mixed.dof_count() == 4 && full.dof_count() == 7 && mixed.connections().size() == 2,
            "embedding failed to reduce DOFs and match nonconforming facets");
        compare(mixed, full, {});
        compare(mixed, full, {StudyType::Transient, 304, 4, 0.5});
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
