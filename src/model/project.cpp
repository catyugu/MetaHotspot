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

    const FacePartition* Project::face_partition(std::size_t component_index, ComponentFace face) const
    {
        const auto& partitions = face_partitions_.at(component_index);
        const auto it = partitions.find(face);
        return it == partitions.end() ? nullptr : &it->second;
    }

    std::optional<Vec3> Project::output_position(std::size_t component_index, OutputId output) const
    {
        const auto& c = components_.at(component_index);
        if (const auto* fom = std::get_if<FomInstance>(&c.model)) {
            if (output < fom->asset->point_probes.size())
                return translate(fom->asset->point_probes[output].point, c.offset);
        }
        else if (const auto& outputs = std::get<EromInstance>(c.model).asset->outputs; output < outputs.size()) {
            const auto& point = outputs[output].point;
            return point ? std::optional {translate(*point, c.offset)} : std::nullopt;
        }
        throw std::out_of_range("unknown component output");
    }
} // namespace mhs::model
