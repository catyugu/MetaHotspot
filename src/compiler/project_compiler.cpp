#include "compiler/project_compiler.hpp"
#include "compiler/geometry_compiler.hpp"

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
        using Rect = std::array<double, 4>;
        using core::Vector;

        std::optional<Rect> overlap(const Rect& a, const Rect& b)
        {
            Rect r {std::max(a[0], b[0]), std::min(a[1], b[1]), std::max(a[2], b[2]), std::min(a[3], b[3])};
            if (r[1] > r[0] && r[3] > r[2])
                return r;
            return std::nullopt;
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
            const Facet* facet;
            const EromSurface* surface;
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
                if (!overlap(rect, node.rect))
                    return;
                if (node.right) {
                    search(rect, match, index + 1);
                    search(rect, match, node.right);
                    return;
                }
                for (const auto& face : node.leaf)
                    if (const auto hit = overlap(rect, face.rect))
                        match(face, *hit);
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

            Vector<Node> nodes_;
        };

        void cover(FacetContact& record, const Face& side, const Rect& cut, const Vec3& offset, std::size_t contact)
        {
            const auto& original = *side.facet;
            auto local = translate(cut, original.axis, {-offset[0], -offset[1], -offset[2]});
            for (std::size_t i = 0; i < local.size(); ++i)
                if (cut[i] == side.rect[i])
                    local[i] = original.bounds[i];
            if (record.contacts.empty())
                record.exposed.push_back(original);
            Vector<Facet> remainder;
            auto append = [&](const Rect& r) {
                if (r[1] > r[0] && r[3] > r[2]) {
                    auto patch = original;
                    patch.bounds = r;
                    remainder.push_back(patch);
                }
            };
            double removed = 0;
            for (const auto& patch : record.exposed) {
                const auto& r = patch.bounds;
                if (const auto x = overlap(r, local)) {
                    removed += ((*x)[1] - (*x)[0]) * ((*x)[3] - (*x)[2]);
                    append({r[0], (*x)[0], r[2], r[3]});
                    append({(*x)[1], r[1], r[2], r[3]});
                    append({(*x)[0], (*x)[1], r[2], (*x)[2]});
                    append({(*x)[0], (*x)[1], (*x)[3], r[3]});
                }
                else
                    remainder.push_back(patch);
            }
            const double area = (local[1] - local[0]) * (local[3] - local[2]);
            if (removed < area * (1 - 1e-8))
                throw std::invalid_argument("overlapping thermal contacts on one component facet");
            record.exposed = std::move(remainder);
            record.contacts.push_back(contact);
        }
    } // namespace

    Project compile_project(ProjectDefinition definition)
    {
        Project project;
        project.components_ = std::move(definition.components);
        const auto components = project.components();
        project.facet_contacts_.resize(components.size());
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
                    for (const auto& [id, law] : component.boundaries) {
                        const bool known = [&] {
                            if constexpr (reduced)
                                return std::ranges::any_of(asset->surfaces,
                                    [id](const auto& surface) { return surface.id == id && surface.training; });
                            else
                                return id < asset->surfaces.size();
                        }();
                        if (!known)
                            throw std::invalid_argument("unknown or non-connectable component surface ID");
                    }
                    if (components.size() <= 1)
                        return;
                    auto append = [&](FacetId id, const Facet& f, const EromSurface* surface = nullptr) {
                        const auto axis = static_cast<std::size_t>(f.axis);
                        const auto plane = f.coordinate + component.offset[axis];
                        const auto rect = translate(f.bounds, f.axis, component.offset);
                        if (!std::isfinite(plane)
                            || !std::ranges::all_of(rect, [](double x) { return std::isfinite(x); })
                            || rect[0] >= rect[1] || rect[2] >= rect[3])
                            throw std::invalid_argument("component translation exceeds facet precision");
                        auto& faces = f.normal > 0 ? positive[axis][plane].faces : negative[axis];
                        faces.push_back({i, id, &f, surface, rect});
                    };
                    if constexpr (reduced) {
                        for (const auto& surface : asset->surfaces)
                            for (const auto id : surface.facets)
                                append(id, asset->facets[id], &surface);
                    }
                    else {
                        for (const auto& facets : group_exterior_facets(*asset))
                            for (const auto id : facets)
                                append(id, asset->discrete.facets[id]);
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

        std::set<std::tuple<ComponentId, SurfaceId, ComponentId, SurfaceId>> warned;
        auto surface_info = [&](const Face& side) -> std::pair<SurfaceId, std::string_view> {
            if (side.surface)
                return {side.surface->id, side.surface->name};
            const auto& fom = *std::get<std::shared_ptr<const Fom>>(components[side.component].asset);
            const auto id = fom.facet_surface[side.id];
            return {id, fom.surfaces[id].name};
        };
        for (std::size_t axis = 0; axis < positive.size(); ++axis) {
            auto& index = positive[axis];
            for (auto& [coordinate, group] : index)
                group.index();
            for (const auto& right : negative[axis]) {
                const auto plane = right.facet->coordinate + components[right.component].offset[axis];
                for (auto it = index.lower_bound(plane - geometry_eps);
                    it != index.end() && it->first <= plane + geometry_eps; ++it)
                    it->second.search(right.rect, [&](const Face& left, const Rect& cut) {
                        if (left.component == right.component)
                            return;
                        const auto l = components[left.component].id, r = components[right.component].id;
                        if ((left.surface && !left.surface->training) || (right.surface && !right.surface->training)) {
                            const auto [left_id, left_name] = surface_info(left);
                            const auto [right_id, right_name] = surface_info(right);
                            const auto key = std::tuple {l, left_id, r, right_id};
                            if (warned.insert(key).second)
                                MHS_LOG_WARN("Components {} ({}) and {} ({}): EROM surface is not connectable; thermal "
                                             "contact is ignored",
                                    l, left_name, r, right_name);
                            return;
                        }
                        for (const auto* side : {&left, &right})
                            cover(project.facet_contacts_[side->component][side->id], *side, cut,
                                components[side->component].offset, project.contacts_.size());
                        const auto resistance = resistances.find(std::minmax(l, r));
                        project.contacts_.push_back({left.component, right.component, left.id, right.id, cut,
                            resistance == resistances.end() ? 0.0 : resistance->second});
                    });
            }
        }
        return project;
    }

} // namespace mhs::compiler
