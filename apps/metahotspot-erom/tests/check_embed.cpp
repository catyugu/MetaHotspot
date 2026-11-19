#include "compiler/fom_compiler.hpp"
#include "compiler/project_compiler.hpp"
#include "erom/extraction.hpp"
#include "solver/solver.hpp"

#include <cmath>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string_view>

using namespace mhs;
using namespace mhs::model;

namespace {
    const MeshSpec mesh {{0, 1, 2, 3}, {0, 0.5, 1}, {0, 1}};

    FomDefinition fixture()
    {
        FomDefinition definition;
        definition.reference_temperature = 295;
        definition.materials = {{"solid", {MaterialKind::Solid, "2", "2", "2", "1", "4", {}}}, {"unused", {}}};
        definition.sources = {"5", "0", "8"};
        definition.source_names = {"heater A", "", "heater B"};
        for (SourceId i = 0; i < 3; ++i) {
            const double x = i;
            definition.volumes.push_back({0, i, {{GeometryOperation::Add, {x, 0, 0}, {x + 1, 1, 1}}}});
        }
        definition.thermal_boundaries = {
            {{{Axis::X, 0, {0, 1, 0, 1}}}, ConvectionBoundary {"3", "299"}, "upstream"},
            {{{Axis::X, 3, {0, 1, 0, 1}}}, ConvectionBoundary {"4", "302"}},
        };
        definition.point_probes = {{"off center", 1.2, 0.5, 0.5}};
        return definition;
    }

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
        check(e.surfaces.size() == 3 && e.surface_training[0] && e.surface_training[1] && !e.surface_training[2],
            "wrong HTC groups");
        check(e.surfaces[0].id == 0 && e.surfaces[0].name == "upstream"
                && e.surfaces[0].facets == core::Vector<FacetId> {0, 1} && e.surfaces[1].id == 1
                && e.surfaces[1].facets == core::Vector<FacetId> {2, 3},
            "wrong facet ownership");
        check(e.sources[0].id == 0 && e.sources[0].name == "heater A" && e.sources[1].id == 2 && e.outputs[0].id == 0
                && e.outputs[0].name == "off center" && e.outputs[1].source == 0 && e.outputs[2].source == 2,
            "wrong source or output identity");
        near(e.sources[0].default_power, 5);
        near(e.sources[1].default_power, 8);
        near(e.surface_training[0]->maximum, 1e4);
        near(e.surface_training[1]->maximum, 9);
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

    void extract_fixture(const std::filesystem::path& path, std::string_view variant)
    {
        using namespace mhs::app::erom;
        auto definition = fixture();
        HtcRanges ranges {{0, {1, 1e4}}, {1, {2, 9}}};
        if (variant == "trace")
            definition.point_probes.push_back({"HTC trace", 3, 0.5, 0.5});
        if (variant == "passive" || variant == "flux") {
            definition.sources = {"0", "0", "0"};
            definition.point_probes.clear();
        }
        if (variant == "flux")
            definition.default_thermal_boundary = NeumannBoundary {"1"};
        if (variant == "empty") {
            definition.thermal_boundaries.push_back({{}, NeumannBoundary {}});
            ranges.emplace(2, HtcRange {1, 10});
        }
        if (variant == "partial") {
            definition.thermal_boundaries[0].regions[0].rectangle.a_max = 0.75;
            definition.sources = {"0", "0", "0"};
            definition.materials[1] = {"resistive", {MaterialKind::Solid, "0.02", "0.02", "0.02", "1", "4", {}}};
            definition.volumes.push_back({1, core::invalid_id, {{GeometryOperation::Add, {0, 0.5, 0}, {1, 1, 1}}}});
        }
        constexpr double tolerance = 1e-6;
        const auto result = extract(definition, mesh, ranges, tolerance);
        check(result.statistics.output_error <= tolerance, "sampled output error exceeds extraction tolerance");
        if (variant == "partial") {
            const auto& e = result.asset;
            const auto& facets = e.surfaces[0].facets;
            check(facets.size() == 2, "wrong facet partition for the partial HTC group");
            near(e.facets[facets[0]].area() + e.facets[facets[1]].area(), 0.75);
            near(e.R[facets[0]], 0.25);
            near(e.R[facets[1]], 25);

            const auto fom = std::make_shared<const Fom>(compiler::compile_fom(definition, mesh));
            const auto rom = std::make_shared<const Erom>(e);
            using expression::CompiledExpression;
            const BoundaryOverrides boundary {
                {0, Convection {CompiledExpression::make_constant(70), CompiledExpression::make_constant(310)}}};
            const auto full = compiler::compile_project({{{0, {}, fom, {}, boundary}}, {}});
            const auto reduced = compiler::compile_project({{{0, {}, rom, {}, boundary}}, {}});
            const auto full_solution = solver::solve(full), reduced_solution = solver::solve(reduced);
            check(full_solution.converged && reduced_solution.converged, "steady solve did not converge");
            near(full_solution.output_traces.at(0).values.back(), reduced_solution.output_traces.at(0).values.back(),
                1e-5);
        }
        write_erom(path, result.asset);
    }

    void compare(const Project& mixed, const Project& full, mhs::solver::Study study)
    {
        solver::SolveOptions opts;
        opts.time_tolerance = 1e-8;
        opts.max_time_step = 0.02;
        opts.linear_tolerance = 1e-11;
        opts.nonlinear_relative_tolerance = 1e-10;
        const auto detailed = solver::solve(full, study, {}, opts), reduced = solver::solve(mixed, study, {}, opts);
        check(detailed.converged && reduced.converged, "a reference or mixed solve did not converge");
        check(detailed.snapshot_times == reduced.snapshot_times, "probe timeline mismatch");
        check(reduced.output_traces.size() == 3, "mixed output count mismatch");
        check(reduced.find_output(10, 0) == &reduced.output_traces[0], "mixed output identity mismatch");

        for (std::size_t i = 0; i < detailed.snapshot_times.size(); ++i) {
            const auto* t = detailed.snapshot_states.data() + i * full.dof_count();
            near(reduced.output_traces[0].values[i], detailed.output_traces[0].values[i], 1e-4);
            near(reduced.output_traces[1].values[i], (t[0] + t[1]) / 2, 1e-4);
            near(reduced.output_traces[2].values[i], (t[4] + t[5]) / 2, 1e-4);
            if (i == 0 && study.type == mhs::solver::StudyType::Transient)
                near(reduced.output_traces[0].values[i], study.initial_temperature, 1e-10);
        }
        const auto& t = detailed.state;
        check(std::abs(detailed.output_traces[0].values.back() - (t[2] + t[3]) / 2) > 0.01,
            "fixture does not distinguish interpolation from nearest cell");
        near(reduced.state.back(), detailed.state.back(), 1e-4);
    }
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 4 && std::string_view(argv[1]) == "--extract") {
            extract_fixture(argv[2], argv[3]);
            return 0;
        }
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
            const auto project = mhs::compiler::compile_project({{{0, {}, e, {}, {}}}, {}});
            check(solver::solve(project).converged, "source-free EROM did not converge");
            return 0;
        }
        check(argc == 2, "expected a .mherom file");
        const auto e = read_erom(argv[1]);
        check_asset(*e);
        const auto definition = fixture();
        auto fom = std::make_shared<const Fom>(compiler::compile_fom(definition, mesh));

        FomDefinition sink;
        sink.materials.push_back(definition.materials[0]);
        sink.volumes = {{0, core::invalid_id, {{GeometryOperation::Add, {}, {1, 1, 1}}}}};
        sink.thermal_boundaries = {{{{Axis::X, 1, {0, 1, 0, 1}}}, DirichletBoundary {"288"}}};
        auto body = std::make_shared<const Fom>(compiler::compile_fom(sink, {{0, 1}, {0, 1}, {0, 1}}));
        ComponentInstance external {20, {3, 0, 0}, body, {}, {}};
        const auto mixed = mhs::compiler::compile_project({{{10, {}, e, {}, {}}, external}, {}});
        const auto full = mhs::compiler::compile_project({{{10, {}, fom, {}, {}}, external}, {}});
        check(mixed.dof_count() == 4 && full.dof_count() == 7 && mixed.contacts().size() == 2,
            "embedding failed to reduce DOFs and match nonconforming facets");
        compare(mixed, full, {});
        compare(mixed, full, {mhs::solver::StudyType::Transient, 304, 4, 0.5});
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
