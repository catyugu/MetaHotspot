#include "compiler/project_compiler.hpp"

#include "core/csr_matrix.hpp"
#include "logging/logger.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace mhs::compiler {
    using namespace model;
    namespace {
        using core::Vector;

        Rect translate(Rect r, Axis axis, const Vec3& offset)
        {
            const auto [u, v] = tangential_axes(axis);
            r[0] += offset[u];
            r[1] += offset[u];
            r[2] += offset[v];
            r[3] += offset[v];
            return r;
        }

        struct Face {
            std::size_t component_index;
            core::EntityIndex face_index;
            const Facet* facet;
            core::EntityIndex group_index; // boundary group, or diagnostic outline group
            std::string_view name;
            bool outline;
            Rect rect;
        };

        // Index the positive faces on one plane by their two tangential bounds.
        class Plane {
        public:
            Vector<Face> faces;
            void index() { build(faces); }

            template <typename Match> void search(const Rect& rect, const Match& match, std::size_t index = 0) const
            {
                const auto& node = nodes_[index];
                if (!rect_intersection(rect, node.rect))
                    return;
                if (node.right) {
                    search(rect, match, index + 1);
                    search(rect, match, node.right);
                    return;
                }
                for (const auto& face : node.leaf)
                    if (const auto hit = rect_intersection(rect, face.rect))
                        match(face, *hit);
            }

        private:
            struct Node {
                Rect rect;
                std::span<const Face> leaf;
                std::size_t right = 0;
            };

            std::size_t build(std::span<Face> range)
            {
                auto rect = range.front().rect;
                for (const auto& face : range) {
                    rect[0] = std::min(rect[0], face.rect[0]);
                    rect[1] = std::max(rect[1], face.rect[1]);
                    rect[2] = std::min(rect[2], face.rect[2]);
                    rect[3] = std::max(rect[3], face.rect[3]);
                }
                const auto index = nodes_.size();
                nodes_.push_back({rect, {}});
                if (range.size() <= 8) {
                    nodes_[index].leaf = range;
                    return index;
                }
                const int t = rect[1] - rect[0] >= rect[3] - rect[2] ? 0 : 2;
                const auto half = range.size() / 2;
                std::nth_element(range.begin(), range.begin() + half, range.end(), [t](const auto& a, const auto& b) {
                    return std::midpoint(a.rect[t], a.rect[t + 1]) < std::midpoint(b.rect[t], b.rect[t + 1]);
                });
                build(range.first(half));
                const auto right = build(range.subspan(half));
                nodes_[index].right = right;
                return index;
            }

            Vector<Node> nodes_;
        };

        void cover(Vector<Rect>& exposed, const Rect& cut)
        {
            Vector<Rect> remainder;
            auto append = [&](const Rect& r) {
                if (positive_rect(r))
                    remainder.push_back(r);
            };
            double removed = 0;
            for (const auto& r : exposed) {
                if (const auto x = rect_intersection(r, cut)) {
                    removed += rect_area(*x);
                    append({r[0], (*x)[0], r[2], r[3]});
                    append({(*x)[1], r[1], r[2], r[3]});
                    append({(*x)[0], (*x)[1], r[2], (*x)[2]});
                    append({(*x)[0], (*x)[1], (*x)[3], r[3]});
                }
                else
                    remainder.push_back(r);
            }
            if (removed < rect_area(cut) * (1 - 1e-8))
                throw std::invalid_argument("overlapping thermal contacts on one component facet");
            exposed = std::move(remainder);
        }
    } // namespace

    Project compile_project(ProjectDefinition definition)
    {
        Project project;
        project.components_ = std::move(definition.components);
        const auto components = project.components();
        project.face_patches_.resize(components.size());
        std::set<ComponentId> ids;
        std::set<const Erom*> validated;
        std::array<std::map<double, Plane>, 3> positive;
        std::array<Vector<Face>, 3> negative;
        for (std::size_t i = 0; i < components.size(); ++i) {
            const auto& component = components[i];
            if (component.id == core::invalid_id || !ids.insert(component.id).second)
                throw std::invalid_argument("component IDs must be valid and unique");
            if (!std::ranges::all_of(component.offset, [](double x) { return std::isfinite(x); }))
                throw std::invalid_argument("component translation must be finite");
            std::visit(
                [&](const auto& asset) {
                    if (!asset)
                        throw std::invalid_argument("component asset is null");
                    using Asset = typename std::decay_t<decltype(asset)>::element_type;
                    constexpr bool reduced = std::is_same_v<Asset, const Erom>;
                    std::size_t size;
                    if constexpr (reduced) {
                        if (validated.insert(asset.get()).second)
                            validate_erom(*asset);
                        size = asset->K.rows();
                        for (const auto& [id, schedule] : component.powers)
                            if (!schedule || id >= asset->sources.size())
                                throw std::invalid_argument("unknown EROM source ID or empty power schedule");
                    }
                    else {
                        size = asset->discrete.cells.size();
                        if (!component.powers.empty())
                            throw std::invalid_argument("FOM source laws belong to the model");
                    }
                    project.dof_offsets_.push_back(core::detail::checked_sparse_size(project.dof_count() + size));
                    for (const auto& [id, law] : component.boundaries)
                        if (id >= asset->boundary_groups.size())
                            throw std::invalid_argument("unknown or non-connectable component boundary group ID");
                    if (components.size() <= 1)
                        return;
                    auto append = [&](core::EntityIndex face_index, const Facet& f, core::EntityIndex group_index,
                                      std::string_view name, bool outline = false) {
                        const auto axis = static_cast<std::size_t>(f.axis);
                        const auto plane = f.coordinate + component.offset[axis];
                        const auto rect = translate(f.bounds, f.axis, component.offset);
                        if (!std::isfinite(plane)
                            || !std::ranges::all_of(rect, [](double x) { return std::isfinite(x); })
                            || !positive_rect(rect))
                            throw std::invalid_argument("component translation exceeds facet precision");
                        auto& faces = f.normal > 0 ? positive[axis][plane].faces : negative[axis];
                        faces.push_back({i, face_index, &f, group_index, name, outline, rect});
                    };
                    if constexpr (reduced) {
                        for (core::EntityIndex id = 0; id < asset->interfaces.size(); ++id) {
                            const auto& f = asset->interfaces[id];
                            append(id, f.facet, f.boundary_group, asset->boundary_groups[f.boundary_group].name);
                        }
                        for (std::size_t group = 0; group < asset->outline.size(); ++group)
                            for (const auto& f : asset->outline[group].facets)
                                append(core::invalid_id, f, group, asset->outline[group].name, true);
                    }
                    else {
                        for (FacetId id = 0; id < asset->facet_boundary_group.size(); ++id)
                            if (const auto surface = asset->facet_boundary_group[id]; surface != core::invalid_id)
                                append(id, asset->discrete.facets[id], surface, asset->boundary_groups[surface].name);
                    }
                },
                component.asset);
        }

        const auto& resistances = definition.contact_resistance;
        for (const auto& [pair, r] : resistances)
            if (pair.first >= pair.second || !ids.contains(pair.first) || !ids.contains(pair.second)
                || !std::isfinite(r) || r < 0)
                throw std::invalid_argument(
                    "contact resistance requires two valid sorted component IDs and a finite nonnegative value");

        std::set<std::tuple<ComponentId, bool, core::EntityIndex, ComponentId, bool, core::EntityIndex>> warned;
        Vector<std::map<core::EntityIndex, Vector<Rect>>> exposed(components.size());
        for (std::size_t axis = 0; axis < positive.size(); ++axis) {
            auto& index = positive[axis];
            for (auto& [coordinate, group] : index)
                group.index();
            for (const auto& right : negative[axis]) {
                const auto plane = right.facet->coordinate + components[right.component_index].offset[axis];
                for (auto it = index.lower_bound(plane - geometry_eps);
                    it != index.end() && it->first <= plane + geometry_eps; ++it)
                    it->second.search(right.rect, [&](const Face& left, const Rect& cut) {
                        if (left.component_index == right.component_index)
                            return;
                        const auto l = components[left.component_index].id, r = components[right.component_index].id;
                        if (left.outline || right.outline) {
                            const auto key
                                = std::tuple {l, left.outline, left.group_index, r, right.outline, right.group_index};
                            if (warned.insert(key).second)
                                MHS_LOG_WARN("Components {} ({}) and {} ({}): EROM surface is not connectable; thermal "
                                             "contact is ignored",
                                    l, left.name, r, right.name);
                            return;
                        }
                        for (const auto* side : {&left, &right}) {
                            auto& remainders = exposed[side->component_index];
                            auto [it, inserted] = remainders.try_emplace(side->face_index);
                            if (inserted)
                                it->second.push_back(side->rect);
                            cover(it->second, cut);
                            project.face_patches_[side->component_index][side->face_index].push_back(
                                project.patches_.size());
                        }
                        const auto resistance = resistances.find(std::minmax(l, r));
                        project.patches_.push_back({cut,
                            SurfacePatch::Contact {{left.component_index, left.face_index},
                                {right.component_index, right.face_index},
                                resistance == resistances.end() ? 0.0 : resistance->second}});
                    });
            }
        }
        for (std::size_t i = 0; i < exposed.size(); ++i)
            for (const auto& [facet, remainders] : exposed[i]) {
                auto& indices = project.face_patches_[i].at(facet);
                for (const auto& bounds : remainders) {
                    indices.push_back(project.patches_.size());
                    project.patches_.push_back({bounds, std::nullopt});
                }
            }
        return project;
    }

} // namespace mhs::compiler
