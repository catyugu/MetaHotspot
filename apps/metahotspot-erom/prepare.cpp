// Build the native operators and observation maps for the standalone extractor.
#include "compiler/model_compiler.hpp"
#include "model/project.hpp"
#include "pta/io.hpp"
#include "solver/assembler.hpp"
#include "solver/thermal_reconstruction.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace {
    using namespace mhs;
    using namespace mhs::model;
    using Weights = std::map<CellId, double>;

    class Writer {
    public:
        explicit Writer(const char* path) : output_(path, std::ios::binary)
        {
            output_.exceptions(std::ios::badbit | std::ios::failbit);
            output_.write("MHPREP01", 8);
        }

        void count(std::size_t value)
        {
            if (value > std::numeric_limits<std::uint32_t>::max())
                throw std::length_error("EROM extraction input exceeds 32-bit counts");
            integer(static_cast<std::uint32_t>(value));
        }

        void scalar(double value) { integer(std::bit_cast<std::uint64_t>(value)); }

        void name(std::string_view value)
        {
            count(value.size());
            output_.write(value.data(), static_cast<std::streamsize>(value.size()));
        }

        void weights(const Weights& weights, double scale)
        {
            count(weights.size());
            for (const auto& [cell, value] : weights) {
                count(cell);
                scalar(value * scale);
            }
        }

        void close() { output_.close(); }

    private:
        template <typename T> void integer(T value)
        {
            std::array<std::uint8_t, sizeof(T)> bytes {};
            for (std::size_t i = 0; i < bytes.size(); ++i)
                bytes[i] = static_cast<std::uint8_t>(value >> (i * 8));
            output_.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }

        std::ofstream output_;
    };

    double constant(const expression::CompiledExpression& e, std::string_view name)
    {
        if (!e.is_constant() || !std::isfinite(e.constant_value()))
            throw std::invalid_argument(std::string(name) + " must be constant and finite for EROM extraction");
        return e.constant_value();
    }

    template <typename T> T number(std::string_view value)
    {
        if constexpr (std::is_same_v<T, double>) {
            std::size_t consumed = 0;
            const auto parsed = std::stod(std::string(value), &consumed);
            if (consumed != value.size())
                throw std::invalid_argument("invalid number: " + std::string(value));
            return parsed;
        }
        else {
            T result;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
            if (error != std::errc {} || end != value.data() + value.size())
                throw std::invalid_argument("invalid number: " + std::string(value));
            return result;
        }
    }

    struct Selection {
        std::map<SurfaceId, EromHtc> htc;
        std::set<std::string> probes;
        bool all_probes = true;
        bool source_temperatures = true;
    };

    Selection parse(int argc, char** argv)
    {
        Selection result;
        for (int i = 3; i < argc; ++i) {
            const std::string_view option = argv[i];
            if (option == "--htc" && i + 3 < argc) {
                const auto id = number<SurfaceId>(argv[++i]);
                const auto min = number<double>(argv[++i]), max = number<double>(argv[++i]);
                if (!std::isfinite(min) || !std::isfinite(max) || min <= 0 || max < min)
                    throw std::invalid_argument("HTC range requires 0 < minimum <= maximum");
                if (!result.htc.emplace(id, EromHtc {min, max}).second)
                    throw std::invalid_argument("duplicate HTC surface");
            }
            else if (option == "--probe" && i + 1 < argc) {
                result.all_probes = false;
                result.probes.emplace(argv[++i]);
            }
            else if (option == "--no-point-probes")
                result.all_probes = false;
            else if (option == "--no-source-temperatures")
                result.source_temperatures = false;
            else
                throw std::invalid_argument("invalid EROM extraction option: " + std::string(option));
        }
        if (result.htc.empty())
            throw std::invalid_argument("select at least one HTC training surface");
        return result;
    }

    struct Source {
        EromSource channel;
        Weights volume;
        double total_volume = 0;
    };

    struct Output {
        EromOutput channel;
        Weights weights;
    };

    Output project_probe(const Model& model, const Selection& selection, OutputId id, core::Vector<double>& state)
    {
        const auto& probe = model.point_probes[id];
        if (probe.cell == core::invalid_id)
            throw std::invalid_argument("cannot retain point probe outside the FOM: " + probe.name);
        const auto facets = model.discrete.incident_facets(probe.cell);
        const auto& mesh = model.discrete;
        const auto center = mesh.cells[probe.cell].center();
        const bool on_boundary = std::ranges::any_of(
            facets, [&](FacetId f) { return mesh.cell_facets[f].exterior() && mesh.facets[f].contains(probe.point); });
        std::set<CellId> support {probe.cell};
        for (const auto f : facets) {
            const auto& cells = mesh.cell_facets[f];
            const auto& facet = mesh.facets[f];
            if (!cells.exterior()) {
                support.insert(cells.minus == probe.cell ? cells.plus : cells.minus);
                continue;
            }
            const auto axis = static_cast<std::size_t>(facet.axis);
            if (selection.htc.contains(model.facet_thermal_condition[f])
                && (on_boundary ? facet.contains(probe.point) : probe.point[axis] != center[axis]))
                throw std::invalid_argument("probe " + probe.name
                    + " depends directly on HTC boundary temperature and cannot be represented by L");
        }
        auto sample = [&] { return solver::sample_point(model, state, 0, probe.cell, probe.point); };
        const double baseline = sample();
        Output output {{id, probe.name, EromOutputKind::PointTemperature, baseline, probe.point}, {}};
        for (const auto cell : support)
            state[cell] = 0;
        const double zero = sample();
        for (const auto cell : support) {
            state[cell] = 1;
            output.weights[cell] = sample() - zero;
            state[cell] = 0;
        }
        for (const auto cell : support)
            state[cell] = model.reference_temperature;
        return output;
    }

    void prepare(const Project& project, double initial_temperature, Selection selection, Writer& writer)
    {
        if (project.components().size() != 1
            || !std::holds_alternative<std::shared_ptr<const Model>>(project.components().front().asset))
            throw std::invalid_argument("EROM extraction currently requires a single FOM project");
        const auto& model = *std::get<std::shared_ptr<const Model>>(project.components().front().asset);
        const auto& mesh = model.discrete;
        if (mesh.cells.empty() || !model.fluid.fluid_to_global.empty())
            throw std::invalid_argument("EROM extraction requires a nonempty solid FOM");

        const std::set<MaterialId> used_materials(model.cell_material.begin(), model.cell_material.end());
        for (const auto id : used_materials) {
            const auto& material = model.material_table[id];
            for (const auto* property : {&material.kx, &material.ky, &material.kz, &material.rho, &material.c})
                if (constant(*property, "Material property") <= 0)
                    throw std::invalid_argument(
                        "EROM extraction requires positive material properties on active cells");
        }

        for (const auto& group : selection.htc)
            if (const auto id = group.first; id >= model.surfaces.size() || model.surfaces[id].facets.empty())
                throw std::invalid_argument("unknown or empty HTC surface " + std::to_string(id));

        core::Vector<const Surface*> surfaces;
        for (const auto& surface : model.surfaces) {
            if (surface.facets.empty())
                continue;
            surfaces.push_back(&surface);
            const auto training = selection.htc.find(surface.id);
            const auto& condition = model.boundary_conditions[surface.id];
            if (const auto* htc = std::get_if<Convection>(&condition)) {
                const double h = constant(htc->coefficient, "HTC");
                if (h < 0 || (h > 0 && training == selection.htc.end()))
                    throw std::invalid_argument(
                        "surface " + std::to_string(surface.id) + " has an untrained or negative HTC");
                if (training != selection.htc.end()) {
                    training->second.coefficient = h;
                    training->second.ambient = constant(htc->ambient_temperature, "Ambient temperature");
                }
            }
            else if (const auto* flux = std::get_if<Neumann>(&condition);
                !flux || constant(flux->heat_flux, "Neumann heat flux") != 0)
                throw std::invalid_argument("surface " + std::to_string(surface.id)
                    + " has a Dirichlet or nonzero Neumann condition unsupported by EROM extraction");
            else if (training != selection.htc.end())
                training->second.ambient = model.reference_temperature;
        }

        std::map<SourceId, Source> sources;
        std::map<SourceId, double> source_density;
        for (const auto& term : model.source_terms) {
            const auto id = term.source;
            auto [density, inserted] = source_density.try_emplace(id);
            if (inserted)
                density->second = constant(model.heat_source_table[id], "Volumetric source");
            // PTA emits a zero-valued source entry even for an unpowered block.
            if (density->second == 0)
                continue;
            if (!sources.contains(id)) {
                auto name = model.source_names.empty() ? std::string {} : model.source_names[id];
                if (name.empty())
                    name = "source " + std::to_string(id);
                sources.emplace(id, Source {{id, std::move(name), 0}, {}, 0});
            }
            auto& source = sources.at(id);
            source.volume[term.cell] += term.weight;
            source.total_volume += term.weight;
        }
        for (auto& [id, source] : sources)
            source.channel.default_power = source_density.at(id) * source.total_volume;

        core::Vector<Output> outputs;
        core::Vector<double> state(mesh.cells.size(), model.reference_temperature);
        for (std::size_t i = 0; i < model.point_probes.size(); ++i) {
            const auto& probe = model.point_probes[i];
            if (selection.all_probes || selection.probes.contains(probe.name))
                outputs.push_back(project_probe(model, selection, static_cast<OutputId>(i), state));
        }
        for (const auto& requested : selection.probes)
            if (!std::ranges::any_of(model.point_probes, [&](const auto& probe) { return probe.name == requested; }))
                throw std::invalid_argument("unknown point probe: " + requested);

        solver::ThermalAssembler assembler(model);
        const auto& operators = assembler.assemble_internal(state, 0);
        writer.count(operators.K.rows());
        writer.count(operators.K.nnz());
        writer.count(surfaces.size());
        writer.count(sources.size());
        writer.count(outputs.size() + (selection.source_temperatures ? sources.size() : 0));
        writer.scalar(model.reference_temperature);
        writer.scalar(initial_temperature);
        for (const auto v : operators.K.row_ptr())
            writer.count(v);
        for (const auto v : operators.K.col_idx())
            writer.count(v);
        for (const auto v : operators.K.values())
            writer.scalar(v);
        for (const auto v : operators.C.values())
            writer.scalar(v);

        for (const auto* surface : surfaces) {
            writer.count(surface->id);
            writer.name(surface->name);
            const auto range = selection.htc.find(surface->id);
            writer.count(range != selection.htc.end());
            if (range != selection.htc.end())
                for (const auto value :
                    {range->second.minimum, range->second.maximum, range->second.coefficient, range->second.ambient})
                    writer.scalar(value);
            writer.count(surface->facets.size());
            for (const auto id : surface->facets) {
                const auto& f = mesh.facets[id];
                writer.count(static_cast<std::uint32_t>(f.axis));
                writer.count(f.normal > 0);
                writer.scalar(f.coordinate);
                for (const auto x : f.bounds)
                    writer.scalar(x);
                if (range != selection.htc.end()) {
                    writer.count(mesh.cell_facets[id].owner());
                    writer.scalar(assembler.resistance(id));
                }
            }
        }
        for (const auto& [id, source] : sources) {
            writer.count(id);
            writer.name(source.channel.name);
            writer.scalar(source.channel.default_power);
            writer.weights(source.volume, 1 / source.total_volume);
        }
        auto write_output = [&](const EromOutput& output, const Weights& weights, double scale) {
            writer.count(output.id);
            writer.name(output.name);
            writer.count(static_cast<std::uint32_t>(output.kind));
            writer.scalar(output.offset);
            if (output.point)
                for (const auto x : *output.point)
                    writer.scalar(x);
            else
                writer.count(output.source);
            writer.weights(weights, scale);
        };
        for (const auto& output : outputs)
            write_output(output.channel, output.weights, 1);
        if (selection.source_temperatures) {
            std::size_t id = model.point_probes.size();
            for (const auto& [source_id, source] : sources)
                write_output(
                    {static_cast<OutputId>(id++), source.channel.name + " temperature",
                        EromOutputKind::SourceTemperature, model.reference_temperature, std::nullopt, source_id},
                    source.volume, 1 / source.total_volume);
        }
    }
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc < 3)
            throw std::invalid_argument("usage: metahotspot-erom-prepare PTA.xml OUTPUT [--htc ID MIN MAX]...");
        const auto input = mhs::app::pta::read_xml(argv[1]);
        auto model = std::make_shared<const Model>(mhs::compiler::build_model(input.model, input.mesh));
        if (std::string_view(argv[2]) == "--list-surfaces") {
            for (const auto& surface : model->surfaces)
                if (!surface.facets.empty())
                    std::cout << surface.id << '\t' << surface.name << '\t' << surface.facets.size() << " facets\n";
            return 0;
        }
        const auto project = compile_project({{{0, {}, std::move(model), {}, {}}}, {}});
        Writer writer(argv[2]);
        prepare(project, input.study.initial_temperature, parse(argc, argv), writer);
        writer.close();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "EROM extraction: " << error.what() << '\n';
        return 1;
    }
}
