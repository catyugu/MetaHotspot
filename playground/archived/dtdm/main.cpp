// DTDM reproduction driver: builds the four mesh levels of the paper and
// compares Thermal-WLP, DTDM and the implicit time-stepping reference.
#include "dtdm.hpp"
#include "mesh.hpp"
#include "stack.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    struct Case {
        int level = 0;
        int patches = 0;
        int orders = 300;
        /// The paper's time-marching configuration: 0.01 s steps over 50 s.
        int steps = 5000;
        double duration = 50.0;
        double scale_over_duration = 10.0;
        double linear_tolerance = 1e-8;
        std::string system = "multi_gpu";
        std::string solver = "amg";
        bool marching = false;
        bool dof_only = false;
        bool verbose = false;
    };

    Case parse(int argc, char** argv)
    {
        Case options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const bool has_value = i + 1 < argc;
            auto value = [&]() -> std::string {
                if (!has_value)
                    throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--level")
                options.level = std::stoi(value());
            else if (arg == "--patch")
                options.patches = std::stoi(value());
            else if (arg == "--orders")
                options.orders = std::stoi(value());
            else if (arg == "--steps")
                options.steps = std::stoi(value());
            else if (arg == "--duration")
                options.duration = std::stod(value());
            else if (arg == "--scale")
                options.scale_over_duration = std::stod(value());
            else if (arg == "--system")
                options.system = value();
            else if (arg == "--solver")
                options.solver = value();
            else if (arg == "--tolerance")
                options.linear_tolerance = std::stod(value());
            else if (arg == "--marching")
                options.marching = true;
            else if (arg == "--dof-only")
                options.dof_only = true;
            else if (arg == "--verbose")
                options.verbose = true;
            else
                throw std::invalid_argument("unknown option " + arg);
        }
        return options;
    }

    /// Probe times that resolve the early transient as well as the end of the
    /// window; the last probe is the window itself, so the reported field is the
    /// field at t_d. The first probe is the initial condition: the reference holds
    /// it exactly, so whatever the transform reports at t = 0 is representation
    /// error and nothing else.
    std::vector<double> probe_times(double duration)
    {
        std::vector<double> times;
        for (const double t : {0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0})
            if (t < duration)
                times.push_back(t);
        times.push_back(duration);
        return times;
    }

    double max_difference(const std::vector<double>& a, const std::vector<double>& b)
    {
        const std::size_t n = std::min(a.size(), b.size());
        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            worst = std::max(worst, std::abs(a[i] - b[i]));
        return worst;
    }

    /// The hottest cell of a field: the waveform the paper plots.
    double peak(const std::vector<double>& field) { return *std::max_element(field.begin(), field.end()); }

    double finest_cell(const std::array<std::vector<double>, 3>& axes)
    {
        double finest = 1e9;
        for (const auto& axis : axes)
            for (std::size_t i = 0; i + 1 < axis.size(); ++i)
                finest = std::min(finest, axis[i + 1] - axis[i]);
        return finest;
    }

    mhs::sim::SolverType parse_solver(const std::string& name)
    {
        if (name == "amg")
            return mhs::sim::SolverType::AmgCg;
        if (name == "pardiso")
            return mhs::sim::SolverType::Pardiso;
        throw std::invalid_argument("unknown solver " + name);
    }

    /// One method's cost, its endpoint field and, when a reference exists, how
    /// far its endpoint is from it.
    void report(const char* name, const mhs::dtdm::MarchResult& result, const mhs::dtdm::MarchResult* reference)
    {
        const double total = result.setup_seconds + result.march_seconds;
        std::printf("  %-14s %6d orders %7.2f s (setup %6.2f + march %6.2f)  peak %8.3f K  last coefficient %.2e", name,
            result.count, total, result.setup_seconds, result.march_seconds, peak(result.samples.back()),
            result.last_coefficient);
        if (reference)
            std::printf("  err %8.4f K", max_difference(result.samples.back(), reference->samples.back()));
        std::printf("\n");
        std::printf("  %-14s %lld setup solves, %lld march solves, %lld CG iterations\n", "", result.setup_solves,
            result.march_solves, result.march_iterations);
    }

    /// The transient at every probe time, and the hotspot waveform the paper plots.
    void report_transient(
        const mhs::dtdm::MarchResult& wlp, const mhs::dtdm::MarchResult& dtdm, const mhs::dtdm::MarchResult* reference)
    {
        const std::vector<double>& times = wlp.sample_times;
        std::printf("  max abs difference over the field, K");
        if (reference)
            std::printf(" (reference: %d steps, %.1e K left in it)\n", reference->count, reference->estimated_error);
        else
            std::printf(" (no reference)\n");
        std::printf("    %8s %12s %12s %12s\n", "time", "Thermal-WLP", "DTDM", "DTDM-WLP");
        for (std::size_t k = 0; k < times.size(); ++k) {
            if (reference)
                std::printf("    %6.2f s %12.4f %12.4f %12.4f\n", times[k],
                    max_difference(wlp.samples[k], reference->samples[k]),
                    max_difference(dtdm.samples[k], reference->samples[k]),
                    max_difference(dtdm.samples[k], wlp.samples[k]));
            else
                std::printf("    %6.2f s %12s %12s %12.4f\n", times[k], "-", "-",
                    max_difference(dtdm.samples[k], wlp.samples[k]));
        }

        std::printf("  hotspot temperature, K (max over the field)\n");
        std::printf("    %8s %12s %12s %12s\n", "time", "Thermal-WLP", "DTDM", "reference");
        for (std::size_t k = 0; k < times.size(); ++k) {
            if (reference)
                std::printf("    %6.2f s %12.4f %12.4f %12.4f\n", times[k], peak(wlp.samples[k]), peak(dtdm.samples[k]),
                    peak(reference->samples[k]));
            else
                std::printf(
                    "    %6.2f s %12.4f %12.4f %12s\n", times[k], peak(wlp.samples[k]), peak(dtdm.samples[k]), "-");
        }
    }

} // namespace

int run(int argc, char** argv)
{
    const Case options = parse(argc, argv);
    if (options.level < 0 || options.level >= static_cast<int>(mhs::dtdm::kLevels.size()))
        throw std::invalid_argument("level out of range");

    using namespace mhs::dtdm;
    const Layout layout = kLevels[static_cast<std::size_t>(options.level)];
    const SystemSpec spec = options.system == "cpu_dram" ? cpu_dram()
        : options.system == "thermal_aware"              ? multi_gpu_thermal_aware()
                                                         : multi_gpu_compact();

    const auto start = std::chrono::steady_clock::now();
    StackMesh mesh = build_stack_mesh(spec, layout);
    const double build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const double miss = 100.0 * static_cast<double>(mesh.dof - layout.target) / static_cast<double>(layout.target);
    std::printf("level %s  cells %ld (paper %ld, %+.2f%%)  power %.3f W  build %.2f s\n", layout.name, mesh.dof,
        layout.target, miss, mesh.total_power, build_seconds);
    std::printf("  in-plane %zu x %zu cells, finest %.4f mm\n", mesh.axes[0].size() - 1, mesh.axes[1].size() - 1,
        1e3 * finest_cell(mesh.axes));
    std::printf("  group z counts: sink %d spreader %d chip-side %d interposer-side %d\n", mesh.group_nz[0],
        mesh.group_nz[1], mesh.group_nz[2], mesh.group_nz[3]);
    if (options.verbose)
        for (std::size_t l = 0; l < mesh.cells_of_layer.size(); ++l)
            std::printf("    layer %zu cells %zu\n", l, mesh.cells_of_layer[l].size());

    if (options.dof_only) {
        for (int m = 0; m < kModuleCount; ++m)
            std::printf("    module %s cells %zu (%.1f%%)\n", kModuleNames[static_cast<std::size_t>(m)],
                mesh.cells_of_module[m].size(),
                100.0 * static_cast<double>(mesh.cells_of_module[m].size()) / static_cast<double>(mesh.dof));
        return 0;
    }

    const std::vector<double> sample_times = probe_times(options.duration);
    const double scale = options.scale_over_duration / options.duration;
    System system(
        std::move(mesh), options.patches, options.patches, parse_solver(options.solver), options.linear_tolerance);

    MarchResult reference;
    if (options.marching) {
        reference = system.reference(options.steps, options.duration, sample_times);
        std::printf("  reference        %6d steps %7.2f s  peak %8.3f K  %.1e K left in it\n", reference.count,
            reference.march_seconds, peak(reference.samples.back()), reference.estimated_error);
    }
    const MarchResult* against = options.marching ? &reference : nullptr;

    const MarchResult wlp = system.monolithic_wlp(scale, options.orders, sample_times);
    report("Thermal-WLP", wlp, against);
    const MarchResult dtdm = system.dtdm(scale, options.orders, sample_times);
    report("DTDM", dtdm, against);
    std::printf("  DTDM vs Thermal-WLP  %.3e K at t_d\n", max_difference(dtdm.samples.back(), wlp.samples.back()));

    report_transient(wlp, dtdm, against);

    if (options.marching) {
        const double base = reference.single_march_seconds;
        std::printf("  against one %d-step march (%.3f s): Thermal-WLP %.1fx, DTDM %.1fx\n", options.steps, base,
            base / (wlp.setup_seconds + wlp.march_seconds), base / (dtdm.setup_seconds + dtdm.march_seconds));
    }
    return 0;
}

int main(int argc, char** argv)
{
    try {
        return run(argc, argv);
    }
    catch (const std::exception& error) {
        std::fprintf(stderr, "fatal: %s\n", error.what());
        return 1;
    }
}
