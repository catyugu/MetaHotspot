// Driver for the HDI reproduction: geometry, the steady state of Sec. III-A, the
// two alternating transient cases of Sec. III-B, and the comparison against the
// standard FVM (accuracy and CPU time).
#include "hdi.hpp"
#include "model.hpp"
#include "reference.hpp"
#include "stack.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    using namespace mhs::hdim;

    struct Settings {
        int level = 0;
        double interface_step = 1.6; // mm, 0 = one block per mesh face
        double dt = 1.0;
        int steps = 200;
        double tolerance = 1e-4; // K, between two nonlinear sweeps
        double linear_tolerance = 1e-10;
        int max_sweeps = 50;
        bool steady = true;
        bool transient = true;
        bool hdi = false;
        bool verbose = false;
    };

    /// The observation points of Fig. 8. The paper gives no coordinates, so these
    /// are the centres of the chiplets that Fig. 4 and Fig. 5 single out: the CPU
    /// of chip1 under the AI core (P1), a GPU of chip1 (P2) and the AI core of
    /// chip2 (P3), in mm from the centre of the stack.
    struct Observation {
        const char* name;
        int layer;
        double x;
        double y;
    };
    constexpr std::array<Observation, 3> kObservation { {
        { "P1", 5, -3.9, -6.15 }, // chip1, CPU under the AI core
        { "P2", 5, -12.8, 7.8 }, // chip1, GPU0
        { "P3", 3, -6.2, -0.15 }, // chip2, AI core
    } };

    int nearest_cell(const Mesh& mesh, int layer, double x, double y)
    {
        const int k = mesh.layers[static_cast<std::size_t>(layer)].nz / 2;
        int best = -1;
        double best_distance = 0.0;
        for (int j = 0; j < mesh.ny; ++j)
            for (int i = 0; i < mesh.nx; ++i) {
                const int id = mesh.id(layer, i, j, k);
                if (id < 0)
                    continue;
                const double cx = 0.5 * (mesh.x[static_cast<std::size_t>(i)] + mesh.x[static_cast<std::size_t>(i + 1)]);
                const double cy = 0.5 * (mesh.y[static_cast<std::size_t>(j)] + mesh.y[static_cast<std::size_t>(j + 1)]);
                const double distance = (cx - x) * (cx - x) + (cy - y) * (cy - y);
                if (best < 0 || distance < best_distance) {
                    best = id;
                    best_distance = distance;
                }
            }
        return best;
    }

    struct Hotspot {
        double temperature = 0.0;
        int cell = -1;
    };

    Hotspot hotspot(const Model& model, const std::vector<double>& field)
    {
        Hotspot out;
        for (int c = 0; c < model.mesh.n; ++c)
            if (out.cell < 0 || field[static_cast<std::size_t>(c)] > out.temperature) {
                out.temperature = field[static_cast<std::size_t>(c)];
                out.cell = c;
            }
        return out;
    }

    void print_hotspot(const Model& model, const std::vector<double>& field)
    {
        const Hotspot hot = hotspot(model, field);
        const int layer = model.mesh.cell_layer[static_cast<std::size_t>(hot.cell)];
        const auto& cells = model.mesh.layers[static_cast<std::size_t>(layer)];
        double x = 0.0;
        double y = 0.0;
        for (int k = 0; k < cells.nz; ++k)
            for (int j = 0; j < model.mesh.ny; ++j)
                for (int i = 0; i < model.mesh.nx; ++i)
                    if (model.mesh.id(layer, i, j, k) == hot.cell) {
                        x = 0.5 * (model.mesh.x[static_cast<std::size_t>(i)]
                            + model.mesh.x[static_cast<std::size_t>(i + 1)]);
                        y = 0.5 * (model.mesh.y[static_cast<std::size_t>(j)]
                            + model.mesh.y[static_cast<std::size_t>(j + 1)]);
                    }
        std::printf("%.2f K in %s at (%.1f, %.1f) mm", hot.temperature, kStack[static_cast<std::size_t>(layer)].name, x,
            y);
    }

    /// Largest temperature inside one layer, for the two dies of Fig. 7.
    void print_die_peak(const Model& model, const std::vector<double>& field, const char* name, int layer)
    {
        double peak = 0.0;
        const auto& cells = model.mesh.layers[static_cast<std::size_t>(layer)];
        for (int k = 0; k < cells.nz; ++k)
            for (int j = 0; j < model.mesh.ny; ++j)
                for (int i = 0; i < model.mesh.nx; ++i) {
                    const int id = model.mesh.id(layer, i, j, k);
                    if (id >= 0)
                        peak = std::max(peak, field[static_cast<std::size_t>(id)]);
                }
        std::printf("  %s %.2f K", name, peak);
    }

    void print_difference(const Model& model, const std::vector<double>& a, const std::vector<double>& b)
    {
        double worst = 0.0;
        double squares = 0.0;
        for (int c = 0; c < model.mesh.n; ++c) {
            const double delta = std::abs(a[static_cast<std::size_t>(c)] - b[static_cast<std::size_t>(c)]);
            worst = std::max(worst, delta);
            squares += delta * delta;
        }
        std::printf("max %.4f K  rms %.4f K", worst, std::sqrt(squares / static_cast<double>(model.mesh.n)));
    }

    /// One transient march: 200 s of Sec. III-B, `chip1_first` picks the case.
    struct March {
        std::array<double, 3> peak {};
        std::array<int, 3> peak_step {};
        double hotspot = 0.0;
        int hotspot_step = 0;
        double seconds = 0.0;
        long long solves = 0;
        std::vector<double> field;
    };

    March march(Model& model, const Settings& settings, const std::vector<int>& observer, Hdi* hdi, bool chip1_first,
        bool verbose)
    {
        March out;
        const int steps = settings.steps;
        const int half = steps / 2;
        std::vector<double> field;

        const auto start = std::chrono::steady_clock::now();
        for (int step = 1; step <= steps; ++step) {
            const bool first = step <= half;
            set_workload(model, { first == chip1_first, first != chip1_first });
            RunInfo info;
            if (hdi != nullptr)
                field = hdi->step(field, settings.tolerance, settings.max_sweeps, info);
            else
                field = solve_reference(model, field, settings.dt, settings.tolerance, settings.max_sweeps, info);
            out.solves += info.solves;
            for (std::size_t i = 0; i < observer.size(); ++i) {
                const double value = field[static_cast<std::size_t>(observer[i])];
                if (value > out.peak[i]) {
                    out.peak[i] = value;
                    out.peak_step[i] = step;
                }
            }
            const double hot = hotspot(model, field).temperature;
            if (hot > out.hotspot) {
                out.hotspot = hot;
                out.hotspot_step = step;
            }
            if (verbose)
                std::printf("    step %3d  sweeps %2d  %5.2f s  peak %.2f K\n", step, info.sweeps, info.seconds, hot);
        }
        out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        out.field = std::move(field);
        return out;
    }

    void print_march(const char* driver, const March& result)
    {
        std::printf("  %-9s %5.2f s  solves %6lld  P1 %.2f K @%3d s  P2 %.2f K @%3d s  P3 %.2f K @%3d s  peak %.2f K "
                    "@%3d s\n",
            driver, result.seconds, result.solves, result.peak[0], result.peak_step[0], result.peak[1],
            result.peak_step[1], result.peak[2], result.peak_step[2], result.hotspot, result.hotspot_step);
    }

} // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        Settings settings;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc)
                    throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--level")
                settings.level = std::stoi(next());
            else if (arg == "--interface-step")
                settings.interface_step = std::stod(next());
            else if (arg == "--dt")
                settings.dt = std::stod(next());
            else if (arg == "--steps")
                settings.steps = std::stoi(next());
            else if (arg == "--tolerance")
                settings.tolerance = std::stod(next());
            else if (arg == "--linear-tolerance")
                settings.linear_tolerance = std::stod(next());
            else if (arg == "--sweeps")
                settings.max_sweeps = std::stoi(next());
            else if (arg == "--steady")
                settings.transient = false;
            else if (arg == "--transient")
                settings.steady = false;
            else if (arg == "--verbose")
                settings.verbose = true;
            else if (arg == "--hdi")
                settings.hdi = true;
            else
                throw std::invalid_argument("unknown option " + arg);
        }
        if (settings.level < 0 || settings.level >= static_cast<int>(kLevels.size()))
            throw std::invalid_argument("level out of range");

        const Level& level = kLevels[static_cast<std::size_t>(settings.level)];
        Model model;
        build_model(model, level);
        const Assembled assembled = assemble(model, {}, 0.0);
        std::printf("level %s  in-plane %.4f mm  target %ld\n", level.name, level.inplane, level.target);
        std::printf("  cells %d", model.mesh.n);
        for (int r = 0; r < kRegionCount; ++r)
            std::printf("  %s %zu", kRegionNames[static_cast<std::size_t>(r)], model.region_cells[r].size());
        std::printf("\n");
        std::size_t top = 0;
        for (const InterfaceFace& face : assembled.interfaces)
            top += face.linear_region == kLinearTop ? 1 : 0;
        double total = 0.0;
        for (const double p : model.power)
            total += p;
        std::printf("  interface faces %zu (top %zu, bottom %zu)  power %.3f W\n", assembled.interfaces.size(), top,
            assembled.interfaces.size() - top, total);

        std::vector<int> observer;
        for (const Observation& point : kObservation)
            observer.push_back(nearest_cell(model.mesh, point.layer, point.x, point.y));

        if (settings.steady) {
            RunInfo reference_info;
            const std::vector<double> expected
                = solve_reference(model, {}, 0.0, settings.tolerance, settings.max_sweeps, reference_info);
            std::printf("[steady] ref   sweeps %2d  %6.2f s  solves %6lld  hotspot ", reference_info.sweeps,
                reference_info.seconds, reference_info.solves);
            print_hotspot(model, expected);
            print_die_peak(model, expected, "chip1", 5);
            print_die_peak(model, expected, "chip2", 3);
            std::printf("\n");
            if (settings.hdi) {
                Hdi hdi(model, 0.0, settings.interface_step, settings.linear_tolerance);
                std::printf("[steady] hdi   presim %.2f s  blocks %d\n", hdi.presimulation_seconds(),
                    hdi.interface_blocks());
                RunInfo info;
                const std::vector<double> field = hdi.steady(settings.tolerance, settings.max_sweeps, info);
                std::printf("[steady] hdi   sweeps %2d  %6.2f s  solves %6lld  hotspot ", info.sweeps, info.seconds,
                    info.solves);
                print_hotspot(model, field);
                std::printf("\n[steady] hdi vs ref  ");
                print_difference(model, field, expected);
                std::printf("\n");
            }
        }

        if (settings.transient) {
            std::printf("[trans]  dt %.3f s  steps %d\n", settings.dt, settings.steps);
            for (int case_index = 0; case_index < 2; ++case_index) {
                const bool chip1_first = case_index == 0;
                std::printf("[trans]  case %c  chip1 first: %s\n", chip1_first ? 'a' : 'b',
                    chip1_first ? "yes" : "no");
                const March expected = march(model, settings, observer, nullptr, chip1_first, settings.verbose);
                print_march("reference", expected);
                if (settings.hdi) {
                    Hdi hdi(model, settings.dt, settings.interface_step, settings.linear_tolerance);
                    std::printf("[trans]  hdi   presim %.2f s  blocks %d\n", hdi.presimulation_seconds(),
                        hdi.interface_blocks());
                    const March hdi_result = march(model, settings, observer, &hdi, chip1_first, settings.verbose);
                    print_march("hdi", hdi_result);
                    std::printf("  difference at %d s: ", settings.steps);
                    print_difference(model, hdi_result.field, expected.field);
                    std::printf("\n");
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "fatal: %s\n", error.what());
        return 1;
    }
}
