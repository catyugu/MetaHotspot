#include "compiler/fom_compiler.hpp"
#include "compiler/project_compiler.hpp"
#include "erom/extraction.hpp"
#include "solver/solver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
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
        definition.sources = {{"heater A", "5"}, {"heater B", "8"}};
        for (SourceId i = 0; i < 3; ++i) {
            const double x = i;
            definition.volumes.push_back({0, i == 1 ? core::invalid_id : i / 2,
                {{GeometryOperation::Add, {x, 0, 0}, {x + 1, 1, 1}}}});
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
        check(e.boundary_groups.size() == 2 && e.outline.size() == 1 && e.outline[0].facets.size() == 4,
            "wrong HTC groups or outline");
        check(e.boundary_groups[0].name == "upstream" && e.interfaces.size() == 4 && e.interfaces[0].boundary_group == 0
                && e.interfaces[1].boundary_group == 0 && e.interfaces[2].boundary_group == 1
                && e.interfaces[3].boundary_group == 1,
            "wrong facet ownership");
        check(e.sources[0].name == "heater A" && e.sources[1].name == "heater B" && e.outputs[0].name == "off center"
                && !e.outputs[1].point && !e.outputs[2].point,
            "wrong source or output identity");
        near(e.interfaces[0].default_coefficient, 3);
        near(e.interfaces[2].default_ambient, 302);
        for (const auto value : e.f)
            near(value, 0);
        for (const auto value : e.output_offset)
            near(value, 0);
        near(e.boundary_groups[0].maximum, 1e4);
        near(e.boundary_groups[1].maximum, 9);
        near(e.reference_temperature, 295);
        for (const auto& facet : e.interfaces)
            near(facet.resistance, 0.25);
        near(e.interfaces[3].facet.area(), 0.5);

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

    void check_roundtrip(const Erom& before, const Erom& after)
    {
        const auto matrix = [](const auto& a, const auto& b) {
            check(a.rows() == b.rows() && a.cols() == b.cols(), "XML matrix dimensions changed");
            for (std::size_t i = 0; i < a.rows(); ++i)
                check(std::ranges::equal(a.row(i), b.row(i)), "XML matrix values changed");
        };
        matrix(before.K, after.K);
        matrix(before.C, after.C);
        matrix(before.B, after.B);
        matrix(before.L, after.L);
        matrix(before.Vb, after.Vb);
        check(before.f == after.f && before.output_offset == after.output_offset
                && before.uniform_temperature_mode == after.uniform_temperature_mode
                && before.reference_temperature == after.reference_temperature, "XML vector or reference changed");
        check(before.interfaces.size() == after.interfaces.size()
                && before.boundary_groups.size() == after.boundary_groups.size()
                && before.outline.size() == after.outline.size()
                && before.sources.size() == after.sources.size()
                && before.outputs.size() == after.outputs.size(), "XML record counts changed");
        const auto same_facet = [](const Facet& a, const Facet& b) {
            return a.axis == b.axis && a.normal == b.normal && a.coordinate == b.coordinate && a.bounds == b.bounds;
        };
        for (std::size_t i = 0; i < before.interfaces.size(); ++i) {
            const auto& a = before.interfaces[i];
            const auto& b = after.interfaces[i];
            check(same_facet(a.facet, b.facet) && a.resistance == b.resistance && a.boundary_group == b.boundary_group
                    && a.default_coefficient == b.default_coefficient && a.default_ambient == b.default_ambient,
                "XML interface changed");
        }
        for (std::size_t i = 0; i < before.boundary_groups.size(); ++i) {
            const auto& a = before.boundary_groups[i];
            const auto& b = after.boundary_groups[i];
            check(a.name == b.name && a.minimum == b.minimum && a.maximum == b.maximum, "XML group changed");
        }
        for (std::size_t i = 0; i < before.outline.size(); ++i) {
            const auto& a = before.outline[i];
            const auto& b = after.outline[i];
            check(a.name == b.name && std::ranges::equal(a.facets, b.facets, same_facet), "XML outline changed");
        }
        for (std::size_t i = 0; i < before.sources.size(); ++i)
            check(before.sources[i].name == after.sources[i].name, "XML source changed");
        for (std::size_t i = 0; i < before.outputs.size(); ++i)
            check(before.outputs[i].name == after.outputs[i].name && before.outputs[i].point == after.outputs[i].point,
                "XML output changed");
    }

    void extract_fixture(const std::filesystem::path& path, std::string_view variant)
    {
        using namespace mhs::app::erom;
        auto definition = fixture();
        HtcRanges ranges {{0, {1, 1e4}}, {1, {2, 9}}};
        if (variant == "trace")
            definition.point_probes.push_back({"HTC trace", 3, 0.5, 0.5});
        if (variant == "passive" || variant == "flux") {
            definition.sources.clear();
            for (auto& volume : definition.volumes)
                volume.source = core::invalid_id;
            definition.point_probes.clear();
        }
        if (variant == "flux")
            definition.default_thermal_boundary = NeumannBoundary {"1"};
        if (variant == "empty") {
            definition.thermal_boundaries.push_back({{}, NeumannBoundary {}});
            ranges.emplace(2, HtcRange {1, 10});
        }
        if (variant == "partial") {
            definition.thermal_boundaries[0].regions[0].bounds[1] = 0.75;
            for (auto& source : definition.sources)
                source.density = "0";
            definition.materials[1] = {"resistive", {MaterialKind::Solid, "0.02", "0.02", "0.02", "1", "4", {}}};
            definition.volumes.push_back({1, core::invalid_id, {{GeometryOperation::Add, {0, 0.5, 0}, {1, 1, 1}}}});
        }
        constexpr double tolerance = 1e-6;
        const auto result = extract(definition, mesh, ranges, tolerance);
        check(result.statistics.output_error <= tolerance, "sampled output error exceeds extraction tolerance");
        if (variant == "partial") {
            const auto& e = result.asset;
            check(e.interfaces[0].boundary_group == 0 && e.interfaces[1].boundary_group == 0
                    && e.interfaces[2].boundary_group == 1,
                "wrong facet partition for the partial HTC group");
            near(e.interfaces[0].facet.area() + e.interfaces[1].facet.area(), 0.75);
            near(e.interfaces[0].resistance, 0.25);
            near(e.interfaces[1].resistance, 25);

            const auto fom = std::make_shared<const Fom>(compiler::compile_fom(definition, mesh));
            const auto rom = std::make_shared<const Erom>(e);
            using expression::CompiledExpression;
            const BoundaryOverrides boundary {
                Convection {CompiledExpression::make_constant(70), CompiledExpression::make_constant(310)}};
            const auto full = compiler::compile_project({{{{}, fom, {}, boundary}}, {}});
            const auto reduced = compiler::compile_project({{{{}, rom, {}, boundary}}, {}});
            const auto full_solution = solver::solve(full), reduced_solution = solver::solve(reduced);
            check(full_solution.converged && reduced_solution.converged, "steady solve did not converge");
            near(full_solution.output_traces.at(0).values.back(), reduced_solution.output_traces.at(0).values.back(),
                1e-5);
        }
        write_erom(path, result.asset);
        check_roundtrip(result.asset, *read_erom(path));
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
        check(reduced.find_output(0, 0) == &reduced.output_traces[0], "mixed output identity mismatch");

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
    void check_embedding(const std::filesystem::path& path)
    {
        const auto e = read_erom(path);
        check_asset(*e);
        const auto definition = fixture();
        auto fom = std::make_shared<const Fom>(compiler::compile_fom(definition, mesh));

        FomDefinition sink;
        sink.materials.push_back(definition.materials[0]);
        sink.volumes = {{0, core::invalid_id, {{GeometryOperation::Add, {}, {1, 1, 1}}}}};
        sink.thermal_boundaries = {{{{Axis::X, 1, {0, 1, 0, 1}}}, DirichletBoundary {"288"}}};
        auto body = std::make_shared<const Fom>(compiler::compile_fom(sink, {{0, 1}, {0, 1}, {0, 1}}));
        ComponentInstance external {{3, 0, 0}, body, {}, {}};
        ComponentInstance package {{}, e, {[](double) { return 5.; }, [](double) { return 8.; }}, {}};
        const auto mixed = mhs::compiler::compile_project({{package, external}, {}});
        const auto full = mhs::compiler::compile_project({{{{}, fom, {}, {}}, external}, {}});
        check(mixed.dof_count() == 4 && full.dof_count() == 7 && mixed.contacts().size() == 2,
            "embedding failed to reduce DOFs and match nonconforming facets");
        compare(mixed, full, {});
        compare(mixed, full, {mhs::solver::StudyType::Transient, 304, 4, 0.5});
    }

    void check_frozen(const std::filesystem::path& path)
    {
        using namespace mhs::app::erom;
        FomDefinition definition;
        definition.reference_temperature = 295;
        definition.materials = {{"solid", {MaterialKind::Solid, "2+x+0.01*T+t", "3+0.005*T+t",
            "3+0.005*T+t", "1+0.001*T+t", "4+y+t", {}}}};
        // A zero-density explicit channel spans both blocks.
        definition.sources = {{"shared heater", "0"}};
        definition.volumes = {{0, 0, {{GeometryOperation::Add, {2, 1, 0}, {3, 2, 1}}}},
            {0, 0, {{GeometryOperation::Add, {3, 1, 0}, {4, 2, 1}}}}};
        definition.thermal_boundaries = {
            {{{Axis::X, 4, {1, 2, 0, 1}}}, ConvectionBoundary {"2+0.01*T+y+t", "T+5+z+t"}, "BCI"},
            {{{Axis::X, 2, {1, 2, 0, 1}}}, DirichletBoundary {"T+10+t"}, "fixed temperature"},
            {{{Axis::Y, 1, {2, 4, 0, 1}}}, NeumannBoundary {"1+0.01*T+z+t"}, "fixed flux"},
            {{{Axis::Z, 0, {2, 4, 1, 2}}}, ConvectionBoundary {"1+0.01*T+x+t", "T-3+t"}, "fixed HTC"}};
        definition.default_thermal_boundary = ConvectionBoundary {"1+y", "T+7+t"};
        definition.point_probes = {{"fixed trace", 2, 1.5, 0.5}, {"interior", 2.7, 1.5, 0.5},
            {"BCI cell center", 3.5, 1.5, 0.5}};
        const MeshSpec grid {{2, 3, 4}, {1, 2}, {0, 1}};
        const auto result = extract(definition, grid, {{0, {1, 20}}}, 1e-6);
        const auto& asset = result.asset;
        check(asset.sources.size() == 1 && asset.outputs.size() == 4, "explicit zero source was not preserved");
        near(asset.output_offset[0], 10);
        near(asset.interfaces.at(0).default_coefficient, 6.45);
        near(asset.interfaces.at(0).default_ambient, 300.5);
        near(asset.interfaces.at(0).facet.coordinate, 2);
        check(asset.outputs[0].point == Vec3 {0, 0.5, 0.5}, "observation was not translated");
        check(std::ranges::any_of(asset.f, [](double v) { return v != 0; }), "fixed load was lost");
        write_erom(path, asset);
        auto rom = read_erom(path);
        check_roundtrip(asset, *rom);

        // Independently specify the linear FOM frozen at (T=295, t=0).
        auto frozen = definition;
        frozen.materials[0].value = {MaterialKind::Solid, "2+x+2.95", "4.475", "4.475", "1.295", "4+y", {}};
        frozen.thermal_boundaries[0].condition = ConvectionBoundary {"4.95+y", "300+z"};
        frozen.thermal_boundaries[1].condition = DirichletBoundary {"305"};
        frozen.thermal_boundaries[2].condition = NeumannBoundary {"3.95+z"};
        frozen.thermal_boundaries[3].condition = ConvectionBoundary {"3.95+x", "292"};
        frozen.default_thermal_boundary = ConvectionBoundary {"1+y", "302"};

        for (double power : {0.0, 3.0}) {
            frozen.sources[0].density = power == 0 ? "0" : "1.5";
            auto fom = std::make_shared<const Fom>(compiler::compile_fom(frozen, grid));
            ComponentInstance package {{2, 1, 0}, rom, {}, {}};
            if (power != 0)
                package.powers = {[power](double) { return power; }};
            const auto full = compiler::compile_project({{{{}, fom, {}, {}}}, {}});
            const auto reduced = compiler::compile_project({{package}, {}});
            for (auto study : {solver::Study {}, solver::Study {solver::StudyType::Transient, 304, 1, 0.25}}) {
                solver::SolveOptions opts;
                opts.time_tolerance = 1e-8;
                opts.max_time_step = 0.02;
                opts.linear_tolerance = 1e-11;
                opts.nonlinear_relative_tolerance = 1e-10;
                const auto a = solver::solve(full, study, {}, opts), b = solver::solve(reduced, study, {}, opts);
                check(a.converged && b.converged && a.snapshot_times == b.snapshot_times, "frozen solve failed");
                for (std::size_t i = 0; i < a.snapshot_times.size(); ++i) {
                    for (std::size_t j = 0; j < 3; ++j)
                        near(a.output_traces[j].values[i], b.output_traces[j].values[i], 1e-4);
                    near((a.snapshot_states[i * 2] + a.snapshot_states[i * 2 + 1]) / 2,
                        b.output_traces[3].values[i], 1e-4);
                }
            }
        }
    }

    void self_test(const std::filesystem::path& path)
    {
        struct Cleanup {
            const std::filesystem::path& path;
            ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
        } cleanup {path};
        extract_fixture(path, "normal");
        check_embedding(path);
        extract_fixture(path, "passive");
        const auto passive = read_erom(path);
        check(passive->sources.empty() && passive->outputs.empty() && passive->B.cols() == 0 && passive->L.rows() == 0,
            "expected empty source and output channels");
        check(solver::solve(compiler::compile_project({{{{}, passive, {}, {}}}, {}})).converged,
            "passive solve failed");
        check_frozen(path);
        for (auto variant : {"trace", "empty"}) {
            bool rejected = false;
            try {
                extract_fixture(path, variant);
            }
            catch (const std::invalid_argument&) {
                rejected = true;
            }
            check(rejected, "invalid extraction was accepted");
        }
        {
            std::ofstream invalid(path);
            invalid << "<Erom version=\"0\"/>";
        }
        bool rejected = false;
        try {
            (void)read_erom(path);
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "unsupported XML version was accepted");
    }
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--self-test") {
            self_test(argv[2]);
            return 0;
        }
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
            const auto project = mhs::compiler::compile_project({{{{}, e, {}, {}}}, {}});
            check(solver::solve(project).converged, "source-free EROM did not converge");
            return 0;
        }
        check(argc == 2, "expected an EROM XML file");
        check_embedding(argv[1]);
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
