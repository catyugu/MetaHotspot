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
            std::optional<ComponentFace> face; // absent for diagnostic outlines
            double coordinate; // world plane
            std::size_t group_index; // boundary group, or diagnostic outline group
            std::string_view name;
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
        project.face_partitions_.resize(components.size());
        std::array<std::map<double, Plane>, 3> positive;
        std::array<Vector<Face>, 3> negative;
        for (std::size_t i = 0; i < components.size(); ++i) {
            const auto& component = components[i];
            if (!std::ranges::all_of(component.offset, [](double x) { return std::isfinite(x); }))
                throw std::invalid_argument("component translation must be finite");
            auto append = [&](std::optional<ComponentFace> face, const Facet& f, std::size_t group_index,
                              std::string_view name) {
                const auto axis = static_cast<std::size_t>(f.axis);
                const auto plane = f.coordinate + component.offset[axis];
                const auto rect = translate(f.bounds, f.axis, component.offset);
                if (!std::isfinite(plane) || !std::ranges::all_of(rect, [](double x) { return std::isfinite(x); })
                    || !positive_rect(rect))
                    throw std::invalid_argument("component translation exceeds facet precision");
                auto& faces = f.normal > 0 ? positive[axis][plane].faces : negative[axis];
                faces.push_back({i, face, plane, group_index, name, rect});
            };
            std::size_t size;
            if (const auto* fom = std::get_if<FomInstance>(&component.model)) {
                if (!fom->asset)
                    throw std::invalid_argument("FOM asset is null");
                const auto& asset = *fom->asset;
                size = asset.discrete.cells.size();
                if (fom->boundaries.size() > asset.boundary_groups.size())
                    throw std::invalid_argument("unknown FOM boundary group index");
                for (const auto& probe : asset.point_probes)
                    if (probe.cell >= size || !asset.discrete.cells[probe.cell].contains(probe.point))
                        throw std::invalid_argument("invalid owning cell for FOM probe: " + probe.name);
                if (components.size() > 1)
                    for (FacetId id = 0; id < asset.facet_boundary_group.size(); ++id)
                        if (const auto group = asset.facet_boundary_group[id]; group != core::invalid_id)
                            append(ComponentFace {id}, asset.discrete.facets[id], group,
                                asset.boundary_groups[group].name);
            }
            else {
                const auto& erom = std::get<EromInstance>(component.model);
                const auto& asset = *erom.asset;
                size = asset.K.rows();
                if (erom.powers.size() > asset.sources.size())
                    throw std::invalid_argument("unknown EROM source index");
                if (erom.htc.size() > asset.boundary_groups.size())
                    throw std::invalid_argument("unknown EROM HTC group index");
                for (const auto& law : erom.htc)
                    if (law
                        && ((law->coefficient.is_constant()
                                && !(std::isfinite(law->coefficient.constant_value())
                                    && law->coefficient.constant_value() >= 0))
                            || (law->ambient_temperature.is_constant()
                                && !std::isfinite(law->ambient_temperature.constant_value()))))
                        throw std::invalid_argument(
                            "EROM HTC requires a finite nonnegative coefficient and finite ambient temperature");
                if (components.size() > 1) {
                    for (core::EntityIndex id = 0; id < asset.interfaces.size(); ++id) {
                        const auto& f = asset.interfaces[id];
                        append(ComponentFace {EromInterfaceId {id}}, f.facet, f.boundary_group,
                            asset.boundary_groups[f.boundary_group].name);
                    }
                    for (std::size_t group = 0; group < asset.outline.size(); ++group)
                        for (const auto& f : asset.outline[group].facets)
                            append(std::nullopt, f, group, asset.outline[group].name);
                }
            }
            project.dof_offsets_.push_back(core::detail::checked_sparse_size(project.dof_count() + size));
        }

        const auto& resistances = definition.contact_resistance;
        for (const auto& [pair, r] : resistances)
            if (pair.first >= pair.second || pair.second >= components.size() || !std::isfinite(r) || r < 0)
                throw std::invalid_argument(
                    "contact resistance requires two valid sorted component indices and a finite nonnegative value");

        std::set<std::tuple<std::size_t, bool, std::size_t, std::size_t, bool, std::size_t>> warned;
        for (std::size_t axis = 0; axis < positive.size(); ++axis) {
            auto& index = positive[axis];
            for (auto& [coordinate, group] : index)
                group.index();
            for (const auto& right : negative[axis]) {
                const auto plane = right.coordinate;
                for (auto it = index.lower_bound(plane - geometry_eps);
                    it != index.end() && it->first <= plane + geometry_eps; ++it)
                    it->second.search(right.rect, [&](const Face& left, const Rect& cut) {
                        if (left.component_index == right.component_index)
                            return;
                        const auto l = left.component_index, r = right.component_index;
                        if (!left.face || !right.face) {
                            const auto key
                                = std::tuple {l, !left.face, left.group_index, r, !right.face, right.group_index};
                            if (warned.insert(key).second)
                                MHS_LOG_WARN("Components {} ({}) and {} ({}): EROM surface is not connectable; thermal "
                                             "contact is ignored",
                                    l, left.name, r, right.name);
                            return;
                        }
                        for (const auto* side : {&left, &right}) {
                            auto& partitions = project.face_partitions_[side->component_index];
                            auto [it, inserted] = partitions.try_emplace(*side->face);
                            if (inserted)
                                it->second.exposed.push_back(side->rect);
                            cover(it->second.exposed, cut);
                            it->second.contacts.push_back(project.contacts_.size());
                        }
                        const auto resistance = resistances.find(std::minmax(l, r));
                        project.contacts_.push_back(
                            {{left.component_index, *left.face}, {right.component_index, *right.face}, cut,
                                resistance == resistances.end() ? 0.0 : resistance->second});
                    });
            }
        }
        // Finish exposed rectangles in component coordinates once, before solving.
        for (std::size_t i = 0; i < components.size(); ++i)
            for (auto& [face, partition] : project.face_partitions_[i]) {
                const auto& model = components[i].model;
                const auto* id = std::get_if<FacetId>(&face);
                const auto& facet = id
                    ? std::get<FomInstance>(model).asset->discrete.facets[*id]
                    : std::get<EromInstance>(model).asset->interface_at(std::get<EromInterfaceId>(face)).facet;
                const auto [u, v] = tangential_axes(facet.axis);
                for (auto& rect : partition.exposed)
                    for (std::size_t a = 0; a < rect.size(); ++a) {
                        const auto shift = components[i].offset[a < 2 ? u : v];
                        rect[a] = rect[a] == facet.bounds[a] + shift ? facet.bounds[a] : rect[a] - shift;
                    }
            }
        return project;
    }

} // namespace mhs::compiler
