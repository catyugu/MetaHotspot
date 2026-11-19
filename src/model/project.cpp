#include "model/project.hpp"
#include <stdexcept>

namespace mhs::model {
    namespace {
        Vec3 translate(Vec3 point, const Vec3& offset)
        {
            for (std::size_t i = 0; i < point.size(); ++i)
                point[i] += offset[i];
            return point;
        }
    } // namespace

    std::optional<Vec3> Project::output_position(ComponentId component, OutputId output) const
    {
        for (const auto& c : components())
            if (c.id == component) {
                if (const auto* model = std::get_if<std::shared_ptr<const Fom>>(&c.asset)) {
                    if (output < (*model)->point_probes.size())
                        return translate((*model)->point_probes[output].point, c.offset);
                }
                else
                    for (const auto& probe : std::get<std::shared_ptr<const Erom>>(c.asset)->outputs)
                        if (probe.id == output)
                            return probe.point ? std::optional {translate(*probe.point, c.offset)} : std::nullopt;
                break;
            }
        throw std::out_of_range("unknown component output");
    }
} // namespace mhs::model
