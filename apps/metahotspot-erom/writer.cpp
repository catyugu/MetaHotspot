#include "erom/extraction.hpp"
#include "model/erom_codec.hpp"

#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace mhs::app::erom {
    namespace {
        class Writer {
        public:
            explicit Writer(const std::filesystem::path& path) : output_(path, std::ios::binary)
            {
                output_.exceptions(std::ios::badbit | std::ios::failbit);
                output_.write(
                    model::erom_codec::magic.data(), static_cast<std::streamsize>(model::erom_codec::magic.size()));
            }
            void count(std::size_t value)
            {
                if (value > std::numeric_limits<std::uint32_t>::max())
                    throw std::length_error("EROM exceeds 32-bit counts");
                scalar(static_cast<std::uint32_t>(value));
            }
            void name(std::string_view value)
            {
                count(value.size());
                output_.write(value.data(), static_cast<std::streamsize>(value.size()));
            }
            template <typename T> void scalar(const T value) { values(std::span(&value, 1)); }
            template <typename T> void values(std::span<const T> data)
            {
                model::erom_codec::transfer(
                    data, [this](const char* p, std::size_t n) { output_.write(p, static_cast<std::streamsize>(n)); });
            }
            void close() { output_.close(); }

        private:
            std::ofstream output_;
        };

        struct Temporary {
            std::filesystem::path path;
            ~Temporary()
            {
                std::error_code error;
                std::filesystem::remove(path, error);
            }
        };
    } // namespace

    void write_erom(const std::filesystem::path& path, const model::Erom& e)
    {
        model::validate_erom(e);
        std::random_device random;
        auto temporary = path;
        do {
            temporary = path;
            temporary += ".tmp-" + std::to_string(random()) + "-" + std::to_string(random());
        } while (std::filesystem::exists(temporary));
        const Temporary cleanup {temporary};
        Writer writer(temporary);
        writer.count(model::erom_codec::version);
        for (const auto count :
            {e.K.rows(), e.sources.size(), e.outputs.size(), e.Vb.rows(), e.facets.size(), e.surfaces.size()})
            writer.count(count);
        writer.scalar(e.reference_temperature);
        for (const auto* matrix : {&e.K, &e.C, &e.B, &e.L, &e.Vb})
            writer.values(model::erom_codec::elements(*matrix));
        writer.values(std::span<const double>(e.R));
        writer.values(std::span<const double>(e.uniform_temperature_mode));
        for (const auto& facet : e.facets) {
            writer.count(static_cast<std::uint32_t>(facet.axis));
            writer.count(facet.normal > 0);
            writer.scalar(facet.coordinate);
            writer.values(std::span<const double>(facet.bounds));
        }
        core::Vector<core::Vector<model::FacetId>> facets_by_surface(e.surfaces.size());
        for (std::size_t id = 0; id < e.facet_surface.size(); ++id)
            facets_by_surface[e.facet_surface[id]].push_back(static_cast<model::FacetId>(id));
        for (std::size_t i = 0; i < e.surfaces.size(); ++i) {
            const auto& surface = e.surfaces[i];
            writer.count(surface.id);
            writer.name(surface.name);
            const auto& h = surface.training;
            writer.count(h.has_value());
            if (h)
                for (const double value : {h->minimum, h->maximum, h->coefficient, h->ambient})
                    writer.scalar(value);
            writer.count(facets_by_surface[i].size());
            for (const auto id : facets_by_surface[i])
                writer.count(id);
        }
        for (const auto& source : e.sources) {
            writer.count(source.id);
            writer.name(source.name);
            writer.scalar(source.default_power);
        }
        for (const auto& output : e.outputs) {
            writer.count(output.id);
            writer.name(output.name);
            if (const auto* point = std::get_if<model::Vec3>(&output.location)) {
                writer.count(core::invalid_id);
                writer.values(std::span<const double>(*point));
            }
            else
                writer.count(std::get<model::SourceId>(output.location));
        }
        writer.close();
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "replace .mherom output");
#else
        std::filesystem::rename(temporary, path);
#endif
    }
} // namespace mhs::app::erom
