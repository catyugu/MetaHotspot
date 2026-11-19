#include "extraction.hpp"

#include <bit>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
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
                output_.write("MHEROM\r\n", 8);
            }
            void count(std::size_t value)
            {
                if (value > std::numeric_limits<std::uint32_t>::max())
                    throw std::length_error("EROM exceeds 32-bit counts");
                integer(static_cast<std::uint32_t>(value));
            }
            void scalar(double value) { integer(std::bit_cast<std::uint64_t>(value)); }
            void name(std::string_view value)
            {
                count(value.size());
                output_.write(value.data(), static_cast<std::streamsize>(value.size()));
            }
            void values(std::span<const double> values)
            {
                if constexpr (std::endian::native == std::endian::little) {
                    if (!values.empty())
                        output_.write(reinterpret_cast<const char*>(values.data()),
                            static_cast<std::streamsize>(values.size_bytes()));
                }
                else
                    for (const double value : values)
                        scalar(value);
            }
            void matrix(const core::Matrix<double>& matrix) { values({matrix.data(), matrix.rows() * matrix.cols()}); }
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
        writer.count(1);
        for (const auto count :
            {e.K.rows(), e.sources.size(), e.outputs.size(), e.Vb.rows(), e.facets.size(), e.surfaces.size()})
            writer.count(count);
        writer.scalar(e.reference_temperature);
        writer.scalar(e.initial_temperature);
        writer.scalar(e.training_tolerance);
        writer.count(!e.uniform_temperature_mode.empty());
        for (const auto* matrix : {&e.K, &e.C, &e.B, &e.L, &e.Vb})
            writer.matrix(*matrix);
        writer.values(e.R);
        writer.values(e.initial_state);
        writer.values(e.uniform_temperature_mode);
        for (const auto& facet : e.facets) {
            writer.count(static_cast<std::uint32_t>(facet.axis));
            writer.count(facet.normal > 0);
            writer.scalar(facet.coordinate);
            writer.values(facet.bounds);
        }
        for (std::size_t i = 0; i < e.surfaces.size(); ++i) {
            const auto& surface = e.surfaces[i];
            writer.count(surface.id);
            writer.name(surface.name);
            const auto& h = e.surface_training[i];
            writer.count(h.has_value());
            if (h)
                for (const double value : {h->minimum, h->maximum, h->coefficient, h->ambient})
                    writer.scalar(value);
            writer.count(surface.facets.size());
            for (const auto id : surface.facets)
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
            writer.count(static_cast<std::uint32_t>(output.kind));
            writer.scalar(output.offset);
            if (output.point)
                writer.values(*output.point);
            else
                writer.count(output.source);
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
