#pragma once

#include "model/types.hpp"
#include <stdexcept>

namespace mhs::model {
    enum class FieldQuantity : std::uint8_t { Temperature, Pressure, Velocity, ReducedCoordinate };
    enum class EntitySupport : std::uint8_t { Cell, Face, Reduced };

    // Canonical algebraic layout: each physical field owns one contiguous
    // global DOF range. Support-entity identity belongs to domain topology.
    struct FieldLayout {
        DomainId domain = core::invalid_id;
        FieldQuantity quantity = FieldQuantity::Temperature;
        DofId offset = 0;
        std::size_t count = 0;

        DofId dof(std::size_t local) const { return static_cast<DofId>(offset + local); }
        EntitySupport support() const
        {
            switch (quantity) {
            case FieldQuantity::Temperature:
            case FieldQuantity::Pressure:
                return EntitySupport::Cell;
            case FieldQuantity::Velocity:
                return EntitySupport::Face;
            case FieldQuantity::ReducedCoordinate:
                return EntitySupport::Reduced;
            }
            throw std::logic_error("unknown field quantity");
        }
    };

    struct StateLayout {
        std::size_t state_count = 0;
        core::Vector<FieldLayout> fields;

        const FieldLayout& field(FieldId id) const { return fields.at(id); }
        FieldId find(DomainId domain, FieldQuantity quantity) const
        {
            for (std::size_t i = 0; i < fields.size(); ++i)
                if (fields[i].domain == domain && fields[i].quantity == quantity)
                    return static_cast<FieldId>(i);
            throw std::out_of_range("field not present in state layout");
        }
    };
} // namespace mhs::model
