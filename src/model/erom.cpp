#include "model/erom.hpp"
#include "erom_codec.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>

namespace mhs::model {
    namespace {
        void require(bool valid, const char* message)
        {
            if (!valid)
                throw std::invalid_argument(std::string(".mherom: ") + message);
        }

        bool finite(std::span<const double> values)
        {
            return std::ranges::all_of(values, [](double x) { return std::isfinite(x); });
        }

        class Reader {
        public:
            explicit Reader(const std::filesystem::path& path) : input_(path, std::ios::binary)
            {
                if (!input_)
                    throw std::runtime_error("cannot open .mherom file: " + path.string());
                input_.seekg(0, std::ios::end);
                const auto length = input_.tellg();
                require(length >= 0, "cannot determine file length");
                remaining_ = static_cast<std::uint64_t>(length);
                input_.seekg(0);
            }

            void check_count(std::uint64_t count, std::uint64_t unit) const
            {
                require(count <= remaining_ / unit, "field exceeds remaining file size");
            }

            std::string bytes(std::size_t count)
            {
                check_count(count, 1);
                std::string value(count, '\0');
                read(value.data(), count);
                return value;
            }

            std::uint32_t u32() { return scalar<std::uint32_t>(); }
            double f64() { return std::bit_cast<double>(scalar<std::uint64_t>()); }

            std::string name() { return bytes(u32()); }

            core::Matrix<double> matrix(std::uint32_t rows, std::uint32_t cols)
            {
                check_count(static_cast<std::uint64_t>(rows) * cols, 8);
                core::Matrix<double> value(rows, cols);
                // The on-disk representation is IEEE binary64, little-endian, row-major.
                doubles({value.data(), static_cast<std::size_t>(rows) * cols});
                return value;
            }

            core::Vector<double> vector(std::uint32_t count)
            {
                check_count(count, 8);
                core::Vector<double> value(count);
                doubles(value);
                return value;
            }

            void finish() const { require(remaining_ == 0, "unexpected trailing data"); }

        private:
            void doubles(std::span<double> values)
            {
                if constexpr (std::endian::native == std::endian::little) {
                    if (!values.empty())
                        read(reinterpret_cast<char*>(values.data()), values.size_bytes());
                }
                else {
                    for (auto& x : values)
                        x = f64();
                }
            }

            void read(char* buffer, std::size_t count)
            {
                require(static_cast<bool>(input_.read(buffer, static_cast<std::streamsize>(count))), "truncated file");
                remaining_ -= count;
            }

            template <typename T> T scalar()
            {
                check_count(1, sizeof(T));
                T value;
                read(reinterpret_cast<char*>(&value), sizeof(T));
                return erom_codec::little_endian(value);
            }

            std::ifstream input_;
            std::uint64_t remaining_;
        };
    } // namespace

    void validate_erom(const Erom& e)
    {
        const auto n = e.K.rows(), nf = e.Vb.rows();
        require(n > 0 && e.K.cols() == n && e.C.rows() == n && e.C.cols() == n, "invalid K/C dimensions");
        require(e.B.rows() == n && e.B.cols() == e.sources.size() && e.L.rows() == e.outputs.size() && e.L.cols() == n
                && e.Vb.cols() == n && e.R.size() == nf && nf <= e.facets.size(),
            "invalid B/L/Vb/R dimensions");
        for (const auto* matrix : {&e.K, &e.C, &e.B, &e.L, &e.Vb})
            for (std::size_t i = 0; i < matrix->rows(); ++i)
                require(finite(matrix->row(i)), "nonfinite matrix entry");
        for (const auto* matrix : {&e.K, &e.C})
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < i; ++j)
                    require(std::abs((*matrix)(i, j) - (*matrix)(j, i))
                            <= 1e-12 * std::max({1.0, std::abs((*matrix)(i, j)), std::abs((*matrix)(j, i))}),
                        "K and C must be symmetric");
        for (std::size_t i = 0; i < n; ++i)
            require(e.C(i, i) > 0 && e.K(i, i) >= 0, "invalid capacity or stiffness diagonal");
        using Dense = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        const auto order = static_cast<Eigen::Index>(n);
        const Eigen::Map<const Dense> C(e.C.data(), order, order), K(e.K.data(), order, order);
        require(C.llt().info() == Eigen::Success, "C must be positive definite");
        Dense shifted = K;
        const double scale = K.cwiseAbs().maxCoeff();
        if (scale > 0) {
            shifted.diagonal().array() += 1e-10 * scale;
            require(shifted.llt().info() == Eigen::Success, "K must be positive semidefinite");
        }
        require(std::isfinite(e.reference_temperature) && e.uniform_temperature_mode.size() == n
                && finite(e.uniform_temperature_mode),
            "invalid temperature initialization");
        require(finite(e.R) && std::ranges::all_of(e.R, [](double r) { return r > 0; }),
            "half-cell resistance must be positive and finite");
        for (const auto& facet : e.facets)
            require(static_cast<unsigned>(facet.axis) < 3 && (facet.normal == -1 || facet.normal == 1)
                    && std::isfinite(facet.coordinate) && finite(facet.bounds) && facet.bounds[0] < facet.bounds[1]
                    && facet.bounds[2] < facet.bounds[3] && std::isfinite(facet.area()) && facet.area() > 0,
                "invalid exterior facet geometry");

        std::set<SurfaceId> surfaces;
        core::Vector<bool> seen(e.facets.size(), false);
        for (const auto& s : e.surfaces) {
            require(s.id != core::invalid_id && surfaces.insert(s.id).second && !s.name.empty() && !s.facets.empty(),
                "invalid or duplicate surface");
            const auto& h = s.training;
            if (h)
                require(std::isfinite(h->minimum) && std::isfinite(h->maximum) && std::isfinite(h->coefficient)
                        && std::isfinite(h->ambient) && h->minimum >= 0 && h->maximum > 0 && h->maximum >= h->minimum
                        && h->coefficient >= 0,
                    "invalid HTC values");
            for (auto id : s.facets) {
                require(id < e.facets.size() && !seen[id] && (id < nf) == h.has_value(),
                    "invalid or duplicate surface facet index");
                seen[id] = true;
            }
        }
        require(std::ranges::all_of(seen, [](bool x) { return x; }), "facet missing from surfaces");
        std::set<SourceId> sources;
        for (const auto& s : e.sources)
            require(s.id != core::invalid_id && sources.insert(s.id).second && !s.name.empty()
                    && std::isfinite(s.default_power),
                "invalid or duplicate source");
        std::set<OutputId> outputs;
        for (const auto& o : e.outputs)
            require(o.id != core::invalid_id && outputs.insert(o.id).second && !o.name.empty()
                    && (o.point ? finite(*o.point) && o.source == core::invalid_id : sources.contains(o.source)),
                "invalid or duplicate temperature output");
    }

    std::shared_ptr<const Erom> read_erom(const std::filesystem::path& path)
    {
        Reader reader(path);
        require(reader.bytes(erom_codec::magic.size()) == erom_codec::magic, "invalid magic");
        require(reader.u32() == erom_codec::version, "unsupported version");
        const auto n = reader.u32(), ns = reader.u32(), no = reader.u32(), nb = reader.u32(), nf = reader.u32(),
                   ng = reader.u32();
        require(n > 0 && nb <= nf, "invalid record counts");
        auto e = std::make_shared<Erom>();
        e->reference_temperature = reader.f64();
        e->K = reader.matrix(n, n);
        e->C = reader.matrix(n, n);
        e->B = reader.matrix(n, ns);
        e->L = reader.matrix(no, n);
        e->Vb = reader.matrix(nb, n);
        e->R = reader.vector(nb);
        e->uniform_temperature_mode = reader.vector(n);
        reader.check_count(nf, 48);
        for (std::size_t i = 0; i < nf; ++i) {
            const auto axis = reader.u32(), normal = reader.u32();
            require(axis < 3 && normal <= 1, "invalid facet axis or normal");
            Facet f;
            f.axis = static_cast<Axis>(axis);
            f.normal = normal ? 1 : -1;
            f.coordinate = reader.f64();
            for (auto& x : f.bounds)
                x = reader.f64();
            e->facets.push_back(f);
        }
        reader.check_count(ng, 16);
        for (std::size_t i = 0; i < ng; ++i) {
            EromSurface surface {{reader.u32(), reader.name(), {}}, std::nullopt};
            const auto trained = reader.u32();
            require(trained <= 1, "invalid surface training flag");
            if (trained)
                surface.training = EromSurfaceTraining {reader.f64(), reader.f64(), reader.f64(), reader.f64()};
            const auto count = reader.u32();
            reader.check_count(count, 4);
            for (std::size_t f = 0; f < count; ++f)
                surface.facets.push_back(reader.u32());
            e->surfaces.push_back(std::move(surface));
        }
        reader.check_count(ns, 16);
        for (std::size_t i = 0; i < ns; ++i)
            e->sources.push_back({reader.u32(), reader.name(), reader.f64()});
        reader.check_count(no, 12);
        for (std::size_t i = 0; i < no; ++i) {
            EromOutput output {reader.u32(), reader.name(), std::nullopt, reader.u32()};
            if (output.source == core::invalid_id)
                output.point = Vec3 {reader.f64(), reader.f64(), reader.f64()};
            e->outputs.push_back(std::move(output));
        }
        reader.finish();
        validate_erom(*e);
        return e;
    }
} // namespace mhs::model
