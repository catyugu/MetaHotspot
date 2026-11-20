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

    std::span<const FacePatch> Project::face_patches(std::size_t component_index, core::EntityIndex face_index) const
    {
        const auto& faces = face_patches_.at(component_index);
        const auto it = faces.find(face_index);
        return it == faces.end() ? std::span<const FacePatch> {} : std::span<const FacePatch> {it->second};
    }

    std::optional<Vec3> Project::output_position(std::size_t component_index, OutputId output) const
    {
        const auto& c = components_.at(component_index);
        if (const auto* model = std::get_if<std::shared_ptr<const Fom>>(&c.asset)) {
            if (output < (*model)->point_probes.size())
                return translate((*model)->point_probes[output].point, c.offset);
        }
        else if (const auto& outputs = std::get<std::shared_ptr<const Erom>>(c.asset)->outputs;
            output < outputs.size()) {
            const auto& point = outputs[output].point;
            return point ? std::optional {translate(*point, c.offset)} : std::nullopt;
        }
        throw std::out_of_range("unknown component output");
    }
} // namespace mhs::model
