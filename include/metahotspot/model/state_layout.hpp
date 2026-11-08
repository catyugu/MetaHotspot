#pragma once

#include "core/vector.hpp"
#include "model/types.hpp"
#include <stdexcept>

namespace mhs::model {
    enum class FieldQuantity : std::uint8_t { Temperature, Pressure, Velocity, ReducedCoordinate };
    enum class EntitySupport : std::uint8_t { Cell, Face, Reduced };

    // Maps local field entries to independent geometric entities and global state DOFs.
    // A velocity support need not coincide with flux integration patches.
    struct FieldLayout {
        DomainId domain = core::invalid_id;
        FieldQuantity quantity = FieldQuantity::Temperature;
        EntitySupport support = EntitySupport::Cell;
        std::uint32_t component = 0;
        core::Vector<core::EntityIndex> entities;
        core::Vector<DofId> dofs;
    };

    struct StateLayout {
        std::size_t state_count = 0;
        core::Vector<FieldLayout> fields;

        const FieldLayout& field(FieldId id) const { return fields.at(id); }
        FieldId find(DomainId domain, FieldQuantity quantity, std::uint32_t component = 0) const
        {
            for (std::size_t i = 0; i < fields.size(); ++i)
                if (fields[i].domain == domain && fields[i].quantity == quantity && fields[i].component == component)
                    return static_cast<FieldId>(i);
            throw std::out_of_range("field not present in state layout");
        }
    };
} // namespace mhs::model
