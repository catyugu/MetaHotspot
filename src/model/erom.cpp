#include "model/erom.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mhs::model {
    namespace {
        void require(bool valid, const char* message)
        {
            if (!valid)
                throw std::invalid_argument(std::string("EROM: ") + message);
        }

        bool finite(std::span<const double> values)
        {
            return std::ranges::all_of(values, [](double x) { return std::isfinite(x); });
        }
    } // namespace

    void validate_erom(const Erom& e)
    {
        const auto n = e.K.rows(), nf = e.Vb.rows();
        require(n > 0 && e.K.cols() == n && e.C.rows() == n && e.C.cols() == n, "invalid K/C dimensions");
        require(e.B.rows() == n && e.B.cols() == e.sources.size() && e.L.rows() == e.outputs.size() && e.L.cols() == n
                && e.Vb.cols() == n && nf == e.interfaces.size(),
            "invalid B/L/Vb dimensions");
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
        require(e.f.size() == n && finite(e.f), "invalid fixed load");
        require(e.output_offset.size() == e.outputs.size() && finite(e.output_offset), "invalid output offset");
        auto validate_facet = [](const Facet& facet) {
            require(static_cast<unsigned>(facet.axis) < 3 && (facet.normal == -1 || facet.normal == 1)
                    && std::isfinite(facet.coordinate) && finite(facet.bounds) && positive_rect(facet.bounds)
                    && std::isfinite(facet.area()) && facet.area() > 0,
                "invalid exterior facet geometry");
        };
        core::Vector<bool> used(e.boundary_groups.size(), false);
        for (const auto& face : e.interfaces) {
            validate_facet(face.facet);
            require(std::isfinite(face.resistance) && face.resistance > 0,
                "half-cell resistance must be positive and finite");
            require(face.boundary_group < e.boundary_groups.size(), "invalid interface surface");
            require(std::isfinite(face.default_coefficient) && face.default_coefficient >= 0
                    && std::isfinite(face.default_ambient),
                "invalid default convection parameters");
            used[face.boundary_group] = true;
        }
        require(std::ranges::all_of(used, [](bool x) { return x; }), "empty surface");
        for (const auto& h : e.boundary_groups)
            require(!h.name.empty() && std::isfinite(h.minimum) && std::isfinite(h.maximum) && h.minimum > 0
                    && h.maximum >= h.minimum,
                "invalid HTC surface");
        for (const auto& group : e.outline) {
            require(!group.name.empty() && !group.facets.empty(), "invalid outline group");
            for (const auto& facet : group.facets)
                validate_facet(facet);
        }
        for (const auto& s : e.sources)
            require(!s.name.empty(), "invalid source");
        for (const auto& o : e.outputs)
            require(!o.name.empty() && (!o.point || finite(*o.point)), "invalid temperature output");
    }
} // namespace mhs::model
