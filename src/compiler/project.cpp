#include "model/project.hpp"

#include "core/csr_matrix.hpp"
#include "logging/logger.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace mhs::model {
    namespace {
        using Rect = std::array<double, 4>;
        using core::Vector;

        std::optional<Rect> overlap(const Rect& a, const Rect& b)
        {
            Rect r {std::max(a[0], b[0]), std::min(a[1], b[1]), std::max(a[2], b[2]), std::min(a[3], b[3])};
            if (r[1] > r[0] && r[3] > r[2])
                return r;
            return std::nullopt;
        }

        Vec3 translate(Vec3 point, const Vec3& offset)
        {
            for (std::size_t i = 0; i < point.size(); ++i)
                point[i] += offset[i];
            return point;
        }

        Rect translate(Rect r, Axis axis, const Vec3& offset)
        {
            r[0] += offset[axis == Axis::X ? 1 : 0];
            r[1] += offset[axis == Axis::X ? 1 : 0];
            r[2] += offset[axis == Axis::Z ? 1 : 2];
            r[3] += offset[axis == Axis::Z ? 1 : 2];
            return r;
        }

        struct Face {
            std::size_t component;
            FacetId id;
            const Surface* surface;
            bool connectable;
            double plane;
            Rect rect;
        };

        // Index the positive faces on one plane by their two tangential bounds.
        class Plane {
        public:
            explicit Plane(Vector<Face>& faces) { build(faces); }

            template <typename Match> void search(const Rect& rect, const Match& match) const
            {
                search(0, rect, match);
            }

        private:
            struct Node {
                Rect rect;
                std::span<const Face> leaf;
                std::size_t right = 0;
            };

            std::size_t build(std::span<Face> faces)
            {
                auto rect = faces.front().rect;
                for (const auto& face : faces) {
                    rect[0] = std::min(rect[0], face.rect[0]);
                    rect[1] = std::max(rect[1], face.rect[1]);
                    rect[2] = std::min(rect[2], face.rect[2]);
                    rect[3] = std::max(rect[3], face.rect[3]);
                }
                const auto index = nodes_.size();
                nodes_.push_back({rect, {}});
                if (faces.size() <= 8) {
                    nodes_[index].leaf = faces;
                    return index;
                }
                const int t = rect[1] - rect[0] >= rect[3] - rect[2] ? 0 : 2;
                const auto half = faces.size() / 2;
                std::nth_element(faces.begin(), faces.begin() + half, faces.end(), [t](const auto& a, const auto& b) {
                    return std::midpoint(a.rect[t], a.rect[t + 1]) < std::midpoint(b.rect[t], b.rect[t + 1]);
                });
                build(faces.first(half));
                const auto right = build(faces.subspan(half));
                nodes_[index].right = right;
                return index;
            }

            template <typename Match> void search(std::size_t index, const Rect& rect, const Match& match) const
            {
                const auto& node = nodes_[index];
                if (!overlap(rect, node.rect))
                    return;
                if (node.right) {
                    search(index + 1, rect, match);
                    search(node.right, rect, match);
                    return;
                }
                for (const auto& face : node.leaf)
                    if (const auto hit = overlap(rect, face.rect))
                        match(face, *hit);
            }

            Vector<Node> nodes_;
        };

        std::span<const Facet> facets(const Component& component)
        {
            if (const auto* model = std::get_if<std::shared_ptr<const Model>>(&component.asset))
                return (*model)->discrete.facets;
            return std::get<std::shared_ptr<const Erom>>(component.asset)->facets;
        }

        void subtract(Vector<Facet>& exposed, const Rect& cut, const Facet& original)
        {
            Vector<Facet> remainder;
            auto append = [&](const Rect& r) {
                if (r[1] > r[0] && r[3] > r[2]) {
                    auto patch = original;
                    patch.bounds = r;
                    remainder.push_back(patch);
                }
            };
            double removed = 0;
            for (const auto& patch : exposed) {
                const auto& r = patch.bounds;
                if (const auto x = overlap(r, cut)) {
                    removed += ((*x)[1] - (*x)[0]) * ((*x)[3] - (*x)[2]);
                    append({r[0], (*x)[0], r[2], r[3]});
                    append({(*x)[1], r[1], r[2], r[3]});
                    append({(*x)[0], (*x)[1], r[2], (*x)[2]});
                    append({(*x)[0], (*x)[1], (*x)[3], r[3]});
                }
                else
                    remainder.push_back(patch);
            }
            const double area = (cut[1] - cut[0]) * (cut[3] - cut[2]);
            if (removed < area * (1 - 1e-8))
                throw std::invalid_argument("overlapping thermal contacts on one component facet");
            exposed = std::move(remainder);
        }
    } // namespace

    Project compile_project(ProjectDefinition definition)
    {
        Project project;
        project.definition_ = std::move(definition);
        const auto components = project.components();
        project.exposed_.resize(components.size());
        std::set<ComponentId> ids;
        std::set<const Erom*> validated;
        std::array<std::map<double, Vector<Face>>, 3> positive;
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
                            if (!schedule
                                || !std::ranges::any_of(asset->sources, [id](const auto& s) { return s.id == id; }))
                                throw std::invalid_argument("unknown EROM source ID or empty power schedule");
                    }
                    else {
                        size = asset->discrete.cells.size();
                        if (!component.powers.empty())
                            throw std::invalid_argument("FOM source laws belong to the model");
                    }
                    project.dof_offsets_.push_back(core::detail::checked_sparse_size(project.dof_count() + size));
                    auto is_open = [&](std::size_t s) {
                        if constexpr (reduced)
                            return asset->htc[s].has_value();
                        else
                            return true;
                    };
                    for (const auto& [id, law] : component.boundaries) {
                        bool found = false;
                        for (std::size_t s = 0; s < asset->surfaces.size(); ++s)
                            found |= asset->surfaces[s].id == id && is_open(s);
                        if (!found)
                            throw std::invalid_argument("unknown or non-connectable component surface ID");
                    }
                    if (components.size() <= 1)
                        return;
                    const auto geometry = facets(component);
                    for (std::size_t s = 0; s < asset->surfaces.size(); ++s)
                        for (const auto id : asset->surfaces[s].facets) {
                            const auto& f = geometry[id];
                            const auto axis = static_cast<std::size_t>(f.axis);
                            const auto plane = f.coordinate + component.offset[axis];
                            const auto rect = translate(f.bounds, f.axis, component.offset);
                            if (!std::isfinite(plane)
                                || !std::ranges::all_of(rect, [](double x) { return std::isfinite(x); })
                                || rect[0] >= rect[1] || rect[2] >= rect[3])
                                throw std::invalid_argument("component translation exceeds facet precision");
                            Face face {i, id, &asset->surfaces[s], is_open(s), plane, rect};
                            if (f.normal > 0)
                                positive[axis][plane].push_back(face);
                            else
                                negative[axis].push_back(face);
                        }
                },
                component.asset);
        }

        const auto& resistances = project.definition_.contact_resistance;
        for (const auto& [pair, r] : resistances)
            if (pair.first >= pair.second || !ids.contains(pair.first) || !ids.contains(pair.second)
                || !std::isfinite(r) || r < 0)
                throw std::invalid_argument(
                    "contact resistance requires two valid sorted component IDs and a finite nonnegative value");

        std::set<std::tuple<ComponentId, SurfaceId, ComponentId, SurfaceId>> warned;
        auto cover = [&](const Face& side, const Rect& cut) {
            const auto& original = facets(components[side.component])[side.id];
            const auto& offset = components[side.component].offset;
            auto local = translate(cut, original.axis, {-offset[0], -offset[1], -offset[2]});
            for (std::size_t i = 0; i < local.size(); ++i)
                if (cut[i] == side.rect[i])
                    local[i] = original.bounds[i];
            auto [it, inserted] = project.exposed_[side.component].try_emplace(side.id);
            if (inserted)
                it->second.push_back(original);
            subtract(it->second, local, original);
        };
        for (std::size_t axis = 0; axis < positive.size(); ++axis) {
            std::map<double, Plane> index;
            for (auto& [coordinate, group] : positive[axis])
                index.try_emplace(coordinate, group);
            for (const auto& right : negative[axis])
                for (auto it = index.lower_bound(right.plane - geometry_eps);
                    it != index.end() && it->first <= right.plane + geometry_eps; ++it)
                    it->second.search(right.rect, [&](const Face& left, const Rect& cut) {
                        if (left.component == right.component)
                            return;
                        const auto l = components[left.component].id, r = components[right.component].id;
                        if (!left.connectable || !right.connectable) {
                            const auto key = std::tuple {l, left.surface->id, r, right.surface->id};
                            if (warned.insert(key).second) {
                                auto message = "Components " + std::to_string(l) + " (" + left.surface->name + ") and "
                                    + std::to_string(r) + " (" + right.surface->name
                                    + "): EROM surface is not connectable; thermal contact is ignored";
                                MHS_LOG_WARN("{}", message);
                                project.warnings_.push_back(std::move(message));
                            }
                            return;
                        }
                        cover(left, cut);
                        cover(right, cut);
                        const auto resistance = resistances.find(std::minmax(l, r));
                        const double area = (cut[1] - cut[0]) * (cut[3] - cut[2]);
                        project.connections_.push_back({left.component, right.component, left.id, right.id, cut, area,
                            resistance == resistances.end() ? 0.0 : resistance->second});
                    });
        }
        return project;
    }

    std::optional<Vec3> Project::output_position(ComponentId component, OutputId output) const
    {
        for (const auto& c : components())
            if (c.id == component) {
                if (const auto* model = std::get_if<std::shared_ptr<const Model>>(&c.asset)) {
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
