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

    std::span<const std::size_t> Project::facet_patches(std::size_t component, FacetId facet) const
    {
        const auto& partitions = facet_patches_[component];
        const auto it = partitions.find(facet);
        return it == partitions.end() ? std::span<const std::size_t> {} : std::span<const std::size_t> {it->second};
    }

    std::optional<Vec3> Project::output_position(ComponentId component, OutputId output) const
    {
        for (const auto& c : components())
            if (c.id == component) {
                if (const auto* model = std::get_if<std::shared_ptr<const Fom>>(&c.asset)) {
                    if (output < (*model)->point_probes.size())
                        return translate((*model)->point_probes[output].point, c.offset);
                }
                else if (const auto& outputs = std::get<std::shared_ptr<const Erom>>(c.asset)->outputs;
                    output < outputs.size()) {
                    const auto* point = std::get_if<Vec3>(&outputs[output].location);
                    return point ? std::optional {translate(*point, c.offset)} : std::nullopt;
                }
                break;
            }
        throw std::out_of_range("unknown component output");
    }
} // namespace mhs::model
