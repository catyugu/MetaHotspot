#include "model/discrete_mesh.hpp"

namespace mhs::model {
    Vec3 FacePatch::center(const std::array<double, 4>& b) const
    {
        Vec3 p {};
        const auto n = static_cast<std::size_t>(axis);
        p[n] = coordinate;
        p[n == 0 ? 1 : 0] = (b[0] + b[1]) / 2;
        p[n == 2 ? 1 : 2] = (b[2] + b[3]) / 2;
        return p;
    }

    bool FacePatch::contains(const Vec3& p, const std::array<double, 4>& b, double eps) const
    {
        const auto n = static_cast<std::size_t>(axis);
        const auto a = n == 0 ? 1 : 0, t = n == 2 ? 1 : 2;
        return std::abs(p[n] - coordinate) <= eps && p[a] >= b[0] - eps && p[a] <= b[1] + eps && p[t] >= b[2] - eps
            && p[t] <= b[3] + eps;
    }
} // namespace mhs::model
